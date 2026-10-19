/**
 * Train a tiny character-level GPT with Kiln's public API and sample from it.
 *
 *   npm run example            (CPU, about a minute)
 *
 * The same code runs on WebGPU in a browser: `await initWebGpu()` from `src/backend/webgpu/register.ts`, then pass
 * `"webgpu"` as the device.
 */
import { manualSeed, tidy, Tensor, CharTokenizer, TokenDataset, GPT, Trainer, optim } from "../src/index";

const text = "To be, or not to be, that is the question: ".repeat(400);
const tok = CharTokenizer.fromText(text);
const { train, val } = new TokenDataset(tok.encode(text), 32).trainValSplit(0.9);

manualSeed(42);
const device = "cpu" as const;
const model = new GPT({ vocabSize: tok.vocabSize, blockSize: 32, nLayer: 2, nHead: 2, nEmbd: 64, dropout: 0.0, bias: true }, device);
const optimizer = optim.configureAdamW(model, { lr: 3e-3, weightDecay: 0.1 });

const trainer = new Trainer({
  model, optimizer, train, val, tokenizer: tok, device,
  batchSize: 16, maxSteps: 150, clipNorm: 1,
  lrSchedule: optim.makeCosineWithWarmup({ warmup: 10, total: 150, lrMax: 3e-3, lrMin: 3e-4 }),
  evalInterval: 50, evalIters: 4,
  callbacks: {
    onStep: (s) => { if (s.step % 25 === 0) console.log(`step ${s.step}  loss ${s.loss.toFixed(3)}  ${s.tokensPerSec.toFixed(0)} tok/s`); },
    onEval: (e) => console.log(`eval @ ${e.step}: val ${e.valLoss.toFixed(3)}`),
  },
});
await trainer.run();

// One forward pass by hand: every op is eager and memory is scoped by tidy().
const x = Tensor.fromArray(tok.encode("To be, or"), [1, 9], { dtype: "i32" });
const nextChar = tidy(() => model.eval().forward(x).logits.argmax(-1)).dataSync()[0] as number;
console.log(`after "To be, or" the model predicts ${JSON.stringify(tok.decode([nextChar]))}`);

const out = await model.generate(x, 60, { temperature: 0.7, topK: 10, seed: 1 });
console.log(tok.decode(await out.data()));
