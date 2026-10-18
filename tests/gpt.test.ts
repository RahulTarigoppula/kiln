import { describe, expect, it } from "vitest";
import { Tensor, fromArray, manualSeed, tidy } from "../src/tensor/index";
import { Random } from "../src/tensor/random";
import { GPT, GPT_PRESETS, gptConfig, sampleLogits } from "../src/models/gpt";
import { loadGPT, saveGPT } from "../src/models/io";
import { configureAdamW } from "../src/optim/adamw";

const ids = (data: number[], shape: number[]): Tensor => fromArray(Int32Array.from(data), shape, { dtype: "i32" });

describe("GPT", () => {
  it("forward shapes with and without targets", () => {
    const gpt = new GPT(gptConfig("nano", { vocabSize: 16, blockSize: 8 }));
    const x = ids(Array.from({ length: 16 }, (_, i) => i % 16), [2, 8]);
    const y = ids(Array.from({ length: 16 }, (_, i) => (i + 1) % 16), [2, 8]);
    const withTargets = gpt.forward(x, y);
    expect([...withTargets.logits.shape]).toEqual([2, 8, 16]);
    expect([...(withTargets.loss as Tensor).shape]).toEqual([]);
    expect(withTargets.loss!.requiresGrad).toBe(true);
    const lastOnly = gpt.forward(x);
    expect([...lastOnly.logits.shape]).toEqual([2, 1, 16]);
    expect(lastOnly.loss).toBeUndefined();
    expect([...gpt.forward(x, undefined, { allPositions: true }).logits.shape]).toEqual([2, 8, 16]);
    // last-position logits agree with the full forward
    const full = gpt.forward(x, undefined, { allPositions: true }).logits.slice(1, 7, 8).contiguous().dataSync();
    const last = lastOnly.logits.dataSync();
    for (let i = 0; i < last.length; i++) expect(Math.abs((last[i] as number) - (full[i] as number))).toBeLessThan(1e-4);
    expect(() => gpt.forward(ids(Array.from({ length: 9 }, () => 0), [1, 9]))).toThrow(/blockSize/);
    expect(GPT_PRESETS.small.nEmbd).toBe(256);
    expect(gpt.flopsPerToken()).toBeGreaterThan(6 * gpt.numParameters());
  });

  it("initial loss is close to ln(vocab)", () => {
    manualSeed(1);
    const V = 64;
    const gpt = new GPT(gptConfig("micro", { vocabSize: V, blockSize: 16 }));
    const rng = new Random(3);
    const x = ids(Array.from({ length: 4 * 16 }, () => rng.int(V)), [4, 16]);
    const y = ids(Array.from({ length: 4 * 16 }, () => rng.int(V)), [4, 16]);
    const loss = (gpt.forward(x, y).loss as Tensor).itemSync();
    expect(Math.abs(loss - Math.log(V))).toBeLessThan(0.3);
  });

  it("generate crops to blockSize, is deterministic with a seed / greedy, and streams tokens", async () => {
    manualSeed(5);
    const gpt = new GPT(gptConfig("nano", { vocabSize: 12, blockSize: 4 }));
    gpt.eval();
    const prompt = ids([1, 2, 3], [1, 3]);
    const seen: number[] = [];
    const out = await gpt.generate(prompt, 10, { temperature: 0, onToken: (id, probs) => { seen.push(id); expect(probs.length).toBe(12); } });
    expect([...out.shape]).toEqual([1, 13]);
    const arr = Array.from(out.dataSync());
    expect(arr.slice(0, 3)).toEqual([1, 2, 3]);
    expect(seen).toEqual(arr.slice(3));
    const again = await gpt.generate(prompt, 10, { temperature: 0 });
    expect(Array.from(again.dataSync())).toEqual(arr);
    const s1 = await gpt.generate(prompt, 6, { temperature: 1, topK: 5, seed: 99 });
    const s2 = await gpt.generate(prompt, 6, { temperature: 1, topK: 5, seed: 99 });
    expect(Array.from(s1.dataSync())).toEqual(Array.from(s2.dataSync()));
    const batched = await gpt.generate(ids([1, 2, 3, 4, 5, 6], [2, 3]), 3, { topP: 0.9 });
    expect([...batched.shape]).toEqual([2, 6]);
    for (const v of batched.dataSync()) expect(v >= 0 && v < 12).toBe(true);
  });

  it("sampleLogits honours temperature, top-k and top-p", () => {
    const logits = Float32Array.from([0, 1, 2, 10]);
    const rng = new Random(1);
    const greedy = sampleLogits(logits, rng, { temperature: 0 });
    expect(greedy.id).toBe(3);
    const k1 = sampleLogits(logits, rng, { topK: 1 });
    expect(k1.id).toBe(3);
    expect(k1.probs[3]).toBeCloseTo(1, 6);
    const k2 = sampleLogits(logits, rng, { topK: 2 });
    expect(k2.probs[0]).toBe(0);
    expect(k2.probs[1]).toBe(0);
    expect((k2.probs[2] as number) + (k2.probs[3] as number)).toBeCloseTo(1, 6);
    const p = sampleLogits(Float32Array.from([0, 0, 0, 5]), rng, { topP: 0.5 });
    expect(p.id).toBe(3);
    const sum = Array.from(sampleLogits(logits, rng, { temperature: 0.5 }).probs).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it("training gate: nano GPT learns to copy a 4-token sequence (loss < 0.3 in ≤ 300 steps)", () => {
    manualSeed(123);
    const V = 8;
    const L = 4;
    const T = 2 * L - 1;
    const gpt = new GPT(gptConfig("nano", { vocabSize: V, blockSize: T }));
    const opt = configureAdamW(gpt, { lr: 3e-3, weightDecay: 0.01 });
    const rng = new Random(7);
    const B = 32;
    const t0 = performance.now();
    let loss = Infinity;
    let steps = 0;
    for (steps = 1; steps <= 300; steps++) {
      const x = new Int32Array(B * T);
      const y = new Int32Array(B * T);
      for (let b = 0; b < B; b++) {
        const seq: number[] = [];
        for (let i = 0; i < L; i++) seq.push(rng.int(V));
        const full = [...seq, ...seq]; // length 2L
        for (let t = 0; t < T; t++) {
          x[b * T + t] = full[t] as number;
          y[b * T + t] = t < L - 1 ? -1 : (full[t + 1] as number); // only the copied half is predictable
        }
      }
      const xs = ids(Array.from(x), [B, T]);
      const ys = ids(Array.from(y), [B, T]);
      const l = tidy(() => {
        const out = gpt.forward(xs, ys);
        (out.loss as Tensor).backward();
        return (out.loss as Tensor).detach();
      });
      opt.step();
      opt.zeroGrad();
      loss = l.itemSync();
      l.free();
      xs.free();
      ys.free();
      if (loss < 0.3) break;
    }
    const secs = (performance.now() - t0) / 1000;
    console.log(`copy task: loss ${loss.toFixed(3)} after ${steps} steps in ${secs.toFixed(1)}s`);
    expect(loss).toBeLessThan(0.3);
    expect(secs).toBeLessThan(60);
  });

  it("checkpoint save → load gives identical logits and keeps config/tokenizer/log", async () => {
    manualSeed(9);
    const gpt = new GPT(gptConfig("nano", { vocabSize: 20, blockSize: 6, dropout: 0.1 }));
    gpt.eval();
    const x = ids([1, 5, 9, 2, 0, 19], [1, 6]);
    const before = gpt.forward(x, undefined, { allPositions: true }).logits.dataSync();
    const tok = { type: "char", chars: ["a", "b"] };
    const log = [{ step: 1, loss: 3, valLoss: 3.1 }];
    const opt = configureAdamW(gpt);
    const { bytes, meta } = await saveGPT(gpt, tok, log, { optimizer: opt, step: 1 });
    expect(meta.config).toMatchObject({ vocabSize: 20, blockSize: 6, nLayer: 2 });
    expect(meta.tokenizer).toEqual(tok);
    expect(meta.trainingLog).toEqual(log);
    expect((meta.extra as { tied: Record<string, string> }).tied).toEqual({ "lm_head.weight": "transformer.wte.weight" });
    const loaded = loadGPT(bytes, meta, "cpu");
    loaded.model.eval();
    const after = loaded.model.forward(x, undefined, { allPositions: true }).logits.dataSync();
    expect(Array.from(after)).toEqual(Array.from(before));
    expect(loaded.step).toBe(1);
    expect(loaded.model.lm_head.weight).toBe(loaded.model.transformer.wte.weight);
    expect(loaded.model.numParameters()).toBe(gpt.numParameters());
    // the optimizer state is not in the model records
    expect(loaded.records.some((r) => r.name.startsWith("optim."))).toBe(false); // state empty before any step()
    expect(bytes.byteLength).toBeGreaterThan(4 * gpt.numParameters());
  });
});
