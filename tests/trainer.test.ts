import { describe, expect, it } from "vitest";
import { manualSeed } from "../src/tensor/random";
import { CharTokenizer } from "../src/data/tokenizer/char";
import { TokenDataset } from "../src/data/dataset";
import { GPT, gptConfig } from "../src/models/gpt";
import { configureAdamW } from "../src/optim/adamw";
import { makeCosineWithWarmup } from "../src/optim/schedulers";
import { Trainer, type StepInfo } from "../src/train/trainer";

const text = "abcdefgh".repeat(60);
const tok = CharTokenizer.fromText(text);
const data = new TokenDataset(tok.encode(text), 8);
const { train, val } = data.trainValSplit(0.8);

function makeModel(): { model: GPT; optimizer: ReturnType<typeof configureAdamW> } {
  manualSeed(11);
  const model = new GPT(gptConfig("nano", { vocabSize: tok.vocabSize, blockSize: 8 }));
  return { model, optimizer: configureAdamW(model, { lr: 2e-3 }) };
}

describe("Trainer", () => {
  it("runs 20 steps with callbacks firing, schedule applied, metrics and log populated", async () => {
    const { model, optimizer } = makeModel();
    const steps: StepInfo[] = [];
    const evals: number[] = [];
    const samples: string[] = [];
    const trainer = new Trainer({
      model, optimizer, train, val, tokenizer: tok, batchSize: 8, maxSteps: 20,
      lrSchedule: makeCosineWithWarmup({ warmup: 5, total: 20, lrMax: 2e-3, lrMin: 2e-4 }),
      evalInterval: 10, evalIters: 2, clipNorm: 1, sampleInterval: 10, sampleTokens: 5, samplePrompt: "ab",
      lossReadbackInterval: 4, seed: 3,
      callbacks: {
        onStep: (s) => steps.push(s),
        onEval: (e) => evals.push(e.valLoss),
        onSample: (s) => samples.push(s.text),
      },
    });
    expect(trainer.state.status).toBe("idle");
    const final = await trainer.run();
    expect(final.status).toBe("done");
    expect(final.step).toBe(20);
    expect(steps.map((s) => s.step)).toEqual(Array.from({ length: 20 }, (_, i) => i + 1));
    expect(steps.every((s) => Number.isFinite(s.loss) && s.loss > 0)).toBe(true);
    expect(steps.every((s) => typeof s.gradNorm === "number")).toBe(true);
    expect(steps[0]!.lr).toBeCloseTo(2e-3 / 6, 10);
    expect(steps[5]!.lr).toBeCloseTo(2e-3, 10);
    expect(steps[19]!.lr).toBeLessThan(steps[10]!.lr);
    expect(steps.some((s) => s.tokensPerSec > 0 && s.msPerStep > 0)).toBe(true);
    expect(evals.length).toBe(2); // steps 10 and 20
    expect(evals.every((v) => Number.isFinite(v))).toBe(true);
    expect(samples.length).toBe(2);
    expect(samples.every((s) => s.length === 5)).toBe(true);
    expect(final.trainingLog.length).toBe(20);
    expect(final.trainingLog.filter((e) => e.valLoss !== undefined).map((e) => e.step)).toEqual([10, 20]);
    expect(final.lastLoss).toBe(steps[19]!.loss);
    expect(optimizer.stepCount).toBe(20);
    // The loss should trend down on this trivially periodic data.
    expect(steps[19]!.loss).toBeLessThan(steps[0]!.loss);
  });

  it("pause/resume suspends the loop and stop ends it early", async () => {
    const { model, optimizer } = makeModel();
    const trainer = new Trainer({ model, optimizer, train, batchSize: 4, maxSteps: 50, callbacks: { onStep: (s) => { if (s.step === 3) trainer.pause(); } } });
    const done = trainer.run();
    while (trainer.state.status !== "paused") await new Promise((r) => setTimeout(r, 1));
    const pausedStep = trainer.state.step;
    expect(pausedStep).toBe(3);
    await new Promise((r) => setTimeout(r, 20));
    expect(trainer.state.step).toBe(3);
    expect(trainer.state.status).toBe("paused");
    trainer.resume();
    expect(trainer.state.status).toBe("running");
    while (trainer.state.step < 6) await new Promise((r) => setTimeout(r, 1));
    trainer.stop();
    const final = await done;
    expect(final.status).toBe("stopped");
    expect(final.step).toBeGreaterThanOrEqual(6);
    expect(final.step).toBeLessThan(50);
    expect(final.trainingLog.length).toBe(final.step);
  });

  it("resumes from a checkpoint: step count, optimizer state and log continue", async () => {
    const first = makeModel();
    const t1 = new Trainer({ model: first.model, optimizer: first.optimizer, train, tokenizer: tok, batchSize: 4, maxSteps: 10, seed: 5 });
    await t1.run();
    const ckpt = await t1.checkpoint();
    expect(ckpt.meta.trainingLog.length).toBe(10);
    expect((ckpt.meta.extra as { step: number }).step).toBe(10);
    const optimNames = ["optim.transformer.wte.weight.m", "optim.transformer.wte.weight.v", "optim.lm_head.weight.m"];
    const header = new TextDecoder().decode(ckpt.bytes.subarray(8, 8 + Number(new DataView(ckpt.bytes.buffer, ckpt.bytes.byteOffset).getBigUint64(0, true))));
    expect(header).toContain(optimNames[0]);
    expect(header).toContain(optimNames[1]);
    expect(header).not.toContain(optimNames[2]); // tied weight: state stored once under the canonical name

    const second = makeModel();
    const steps: number[] = [];
    const t2 = new Trainer({ model: second.model, optimizer: second.optimizer, train, tokenizer: tok, batchSize: 4, maxSteps: 15, seed: 5, callbacks: { onStep: (s) => steps.push(s.step) } });
    t2.loadCheckpoint(ckpt.bytes, ckpt.meta);
    expect(t2.state.step).toBe(10);
    expect(second.optimizer.stepCount).toBe(10);
    // weights identical after load
    const a = first.model.transformer.wte.weight.dataSync();
    const b = second.model.transformer.wte.weight.dataSync();
    expect(Array.from(b)).toEqual(Array.from(a));
    const m1 = first.optimizer.stateTensors().get(first.model.transformer.wte.weight)!.m!.dataSync();
    const m2 = second.optimizer.stateTensors().get(second.model.transformer.wte.weight)!.m!.dataSync();
    expect(Array.from(m2)).toEqual(Array.from(m1));
    const final = await t2.run();
    expect(steps).toEqual([11, 12, 13, 14, 15]);
    expect(final.step).toBe(15);
    expect(second.optimizer.stepCount).toBe(15);
    expect(final.trainingLog.map((e) => e.step)).toEqual(Array.from({ length: 15 }, (_, i) => i + 1));
  });
});

describe("Trainer: resume edge cases", () => {
  it("a finished trainer can load a checkpoint and continue with a larger maxSteps; loading while running throws", async () => {
    const { model, optimizer } = makeModel();
    const t = new Trainer({ model, optimizer, train, val, tokenizer: tok, batchSize: 4, maxSteps: 4, evalInterval: 2, evalIters: 1, seed: 3 });
    await t.run();
    expect(t.state.status).toBe("done");
    const ckpt = await t.checkpoint();
    expect(ckpt.meta.trainingLog.some((e) => typeof e.valLoss === "number")).toBe(true);

    const second = makeModel();
    const t2 = new Trainer({ model: second.model, optimizer: second.optimizer, train, val, tokenizer: tok, batchSize: 4, maxSteps: 6, seed: 3 });
    await t2.run();
    expect(t2.state.status).toBe("done");
    t2.loadCheckpoint(ckpt.bytes, ckpt.meta);
    expect(t2.state.status).toBe("idle");
    expect(t2.state.step).toBe(4);
    expect(t2.state.lastValLoss).not.toBeNull();
    const final = await t2.run();
    expect(final.step).toBe(6);
    expect(final.trainingLog.map((e) => e.step)).toEqual([1, 2, 3, 4, 5, 6]);

    const third = makeModel();
    const t3 = new Trainer({ model: third.model, optimizer: third.optimizer, train, batchSize: 4, maxSteps: 50, seed: 3 });
    const running = t3.run();
    expect(() => t3.loadCheckpoint(ckpt.bytes, ckpt.meta)).toThrow(/stop the run/);
    t3.stop();
    await running;
    expect(() => t3.loadCheckpoint(ckpt.bytes, { ...ckpt.meta, format: "nope" } as unknown as typeof ckpt.meta)).toThrow(/kiln.v1/);
  });
});
