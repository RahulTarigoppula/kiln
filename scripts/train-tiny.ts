/**
 * Train a char-level GPT on TinyShakespeare on the CPU within a wall-clock budget and ship the result to
 * `web/public/checkpoints/shakespeare-tiny.{safetensors,json}` for the Playground page.
 *
 * Usage: `npm run train:tiny -- [--minutes 25] [--preset auto|A|B|C] [--out web/public/checkpoints/shakespeare-tiny.safetensors]`
 *
 * The script calibrates throughput first and picks the largest config that fits enough optimizer steps into
 * the budget (A: nLayer 4 / nEmbd 128, B: 3 / 96, C: 2 / 64). The cosine schedule's horizon is re-estimated
 * from the measured ms/step as training proceeds so the learning rate still decays to lrMin by the deadline.
 */
import { readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { Tensor, fromArray, manualSeed, tidy } from "../src/tensor/index";
import { CharTokenizer } from "../src/data/tokenizer/char";
import { TokenDataset } from "../src/data/dataset";
import { GPT, type GPTConfig } from "../src/models/gpt";
import { saveGPTNode } from "../src/models/io-node";
import { configureAdamW } from "../src/optim/adamw";
import { cosineWithWarmup } from "../src/optim/schedulers";
import { Trainer } from "../src/train/trainer";

interface Candidate {
  name: string;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  blockSize: number;
  batchSize: number;
  /** Minimum number of optimizer steps the budget must afford for this candidate to be chosen. */
  minSteps: number;
}

const CANDIDATES: Candidate[] = [
  { name: "A", nLayer: 4, nHead: 4, nEmbd: 128, blockSize: 128, batchSize: 16, minSteps: 700 },
  { name: "B", nLayer: 3, nHead: 3, nEmbd: 96, blockSize: 128, batchSize: 16, minSteps: 900 },
  { name: "C", nLayer: 2, nHead: 2, nEmbd: 64, blockSize: 64, batchSize: 16, minSteps: 0 },
];

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] !== undefined ? (process.argv[i + 1] as string) : def;
}

const minutes = Number(arg("minutes", "25"));
const presetArg = arg("preset", "auto");
const outPath = resolve(arg("out", "web/public/checkpoints/shakespeare-tiny.safetensors"));
const budgetMs = minutes * 60_000;
const LR_MAX = 1e-3;
const LR_MIN = 1e-4;
const WARMUP = 50;
const SEED = 1337;

const t0 = performance.now();
const elapsed = (): number => performance.now() - t0;
const log = (msg: string): void => console.log(`[${(elapsed() / 1000).toFixed(0).padStart(5)}s] ${msg}`);

function buildModel(c: Candidate, vocabSize: number): GPT {
  const cfg: GPTConfig = { vocabSize, blockSize: c.blockSize, nLayer: c.nLayer, nHead: c.nHead, nEmbd: c.nEmbd, dropout: 0, bias: true };
  manualSeed(SEED);
  return new GPT(cfg);
}

/** ms per optimizer step for a candidate (median of a few steps after a warm-up step). */
function calibrate(c: Candidate, train: TokenDataset, vocabSize: number): number {
  const model = buildModel(c, vocabSize);
  const opt = configureAdamW(model, { lr: LR_MAX });
  const ds = new TokenDataset(train.ids, c.blockSize);
  const times: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = ds.getBatch(c.batchSize, () => Math.random());
    const xs = fromArray(x, [c.batchSize, c.blockSize], { dtype: "i32" });
    const ys = fromArray(y, [c.batchSize, c.blockSize], { dtype: "i32" });
    const s = performance.now();
    tidy(() => {
      const { loss } = model.forward(xs, ys);
      (loss as Tensor).backward();
    });
    opt.step();
    opt.zeroGrad();
    if (i > 0) times.push(performance.now() - s);
    xs.free();
    ys.free();
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)] as number;
}

async function main(): Promise<void> {
  const text = await readFile(resolve("data/tinyshakespeare.txt"), "utf8");
  const tok = CharTokenizer.fromText(text);
  const ids = tok.encode(text);
  log(`corpus: ${text.length} chars, vocab ${tok.vocabSize}; budget ${minutes} min`);

  // ---- pick a config
  let chosen: Candidate | null = null;
  let msPerStep = 0;
  const probe = new TokenDataset(ids, 128);
  for (const c of CANDIDATES) {
    if (presetArg !== "auto" && c.name !== presetArg) continue;
    const ms = calibrate(c, probe, tok.vocabSize);
    const projected = Math.floor((budgetMs - elapsed()) / ms);
    log(`candidate ${c.name} (L${c.nLayer} H${c.nHead} C${c.nEmbd} T${c.blockSize} B${c.batchSize}): ${ms.toFixed(0)} ms/step → ~${projected} steps in budget`);
    if (projected >= c.minSteps || presetArg !== "auto") {
      chosen = c;
      msPerStep = ms;
      break;
    }
  }
  if (!chosen) throw new Error("no candidate fits the budget");
  log(`chosen config ${chosen.name}`);

  // ---- data + model
  const data = new TokenDataset(ids, chosen.blockSize);
  const { train, val } = data.trainValSplit(0.9);
  const model = buildModel(chosen, tok.vocabSize);
  const optimizer = configureAdamW(model, { lr: LR_MAX, weightDecay: 0.1, betas: [0.9, 0.95] });
  log(`model: ${model.numParameters()} params (non-embedding), ${model.flopsPerToken(chosen.blockSize).toExponential(2)} FLOPs/token`);

  // ---- schedule horizon, re-estimated as we learn the real throughput
  const trainStart = elapsed();
  let totalSteps = Math.max(WARMUP + 10, Math.floor((budgetMs - trainStart) / msPerStep));
  const schedule = (step: number): number => cosineWithWarmup(step, { warmup: WARMUP, total: totalSteps, lrMax: LR_MAX, lrMin: LR_MIN });

  let lastVal = NaN;
  const trainer = new Trainer({
    model, optimizer, train, val, tokenizer: tok,
    batchSize: chosen.batchSize, maxSteps: 1_000_000, lrSchedule: schedule,
    evalInterval: 50, evalIters: 5, clipNorm: 1, lossReadbackInterval: 1, seed: SEED,
    callbacks: {
      onStep: (s) => {
        if (s.step % 50 === 0) {
          log(`step ${s.step}: loss ${s.loss.toFixed(4)} lr ${s.lr.toExponential(2)} ${s.msPerStep.toFixed(0)} ms/step ${s.tokensPerSec.toFixed(0)} tok/s (horizon ${totalSteps})`);
        }
        if (s.step % 25 === 0 && s.msPerStep > 0) {
          const remaining = budgetMs - elapsed();
          const projected = s.step + Math.floor(remaining / s.msPerStep);
          totalSteps = Math.max(s.step + 1, projected);
        }
        if (elapsed() >= budgetMs) trainer.stop();
      },
      onEval: (e) => {
        lastVal = e.valLoss;
        log(`eval @ ${e.step}: train ${e.trainLoss.toFixed(4)} val ${e.valLoss.toFixed(4)}`);
      },
    },
  });

  const state = await trainer.run();
  log(`training ${state.status} at step ${state.step}`);

  // ---- final eval + sample
  const finalVal = await trainer.evaluate(val, 20);
  const finalTrain = await trainer.evaluate(train, 20);
  log(`final: train loss ${finalTrain.toFixed(4)} val loss ${finalVal.toFixed(4)} (last periodic val ${Number.isNaN(lastVal) ? "n/a" : lastVal.toFixed(4)})`);
  model.eval();
  const prompt = fromArray(tok.encode("\n"), [1, 1], { dtype: "i32" });
  const out = await model.generate(prompt, 500, { temperature: 0.8, topK: 40, seed: 7 });
  const sample = tok.decode((await out.data()).subarray(1));
  console.log("\n----- sample (temperature 0.8, top-k 40) -----\n" + sample + "\n----- end sample -----\n");

  // ---- save (weights only, to stay well under 5 MB)
  const trainingLog = state.trainingLog;
  const last = trainingLog[trainingLog.length - 1];
  if (last) last.valLoss = finalVal;
  const saved = await saveGPTNode(outPath, model, tok.toJSON(), trainingLog, {
    step: state.step,
    extra: { finalValLoss: finalVal, finalTrainLoss: finalTrain, sample, config: chosen.name, minutes, cpu: true },
  });
  const metaSize = (await stat(saved.metaPath)).size;
  log(`saved ${saved.path} (${(saved.bytes / 1e6).toFixed(2)} MB) + ${saved.metaPath} (${(metaSize / 1e3).toFixed(0)} KB)`);
}

main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
