import { describe, expect, it } from "vitest";
import { Tensor, fromArray, manualSeed } from "../src/tensor/index";
import { Random } from "../src/tensor/random";
import { Module, ModuleList } from "../src/nn/module";
import { Linear } from "../src/nn/linear";
import { LayerNorm } from "../src/nn/layernorm";
import { Embedding } from "../src/nn/embedding";
import { CausalSelfAttention } from "../src/nn/attention";
import { Block } from "../src/nn/transformer";
import { Sequential } from "../src/nn/sequential";
import { GELU } from "../src/nn/activations";
import { GPT, GPT_PRESETS, gptConfig } from "../src/models/gpt";

function expectClose(t: Tensor, expected: ArrayLike<number>, tol = 1e-5): void {
  const got = t.dataSync();
  expect(got.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((got[i] as number) - (expected[i] as number))).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected[i] as number)));
  }
}

function setValues(t: Tensor, values: number[]): void {
  t.copy_(fromArray(Float32Array.from(values), [...t.shape]));
}

describe("Linear", () => {
  it("computes x·Wᵀ + b against hand-computed values and respects bias=false", () => {
    const lin = new Linear(3, 2);
    setValues(lin.weight, [1, 2, 3, 4, 5, 6]); // W = [[1,2,3],[4,5,6]]
    setValues(lin.bias as Tensor, [0.5, -0.5]);
    const x = fromArray(Float32Array.from([1, 0, -1, 2, 2, 2]), [2, 3]);
    const y = lin.forward(x);
    // row0: [1-3, 4-6] + b = [-1.5, -2.5]; row1: [12, 30] + b = [12.5, 29.5]
    expectClose(y, [-1.5, -2.5, 12.5, 29.5]);
    expect([...y.shape]).toEqual([2, 2]);
    const nb = new Linear(3, 2, { bias: false });
    expect(nb.bias).toBeNull();
    expect(nb.parameters().length).toBe(1);
    expect(nb.numParameters()).toBe(6);
  });

  it("initialises with std 0.02 (scaled for residual projections)", () => {
    manualSeed(7);
    const lin = new Linear(256, 256);
    const w = lin.weight.dataSync();
    let s2 = 0;
    for (let i = 0; i < w.length; i++) s2 += (w[i] as number) ** 2;
    expect(Math.sqrt(s2 / w.length)).toBeCloseTo(0.02, 3);
    const scaled = new Linear(256, 256, { scaleInit: 8 });
    const w2 = scaled.weight.dataSync();
    s2 = 0;
    for (let i = 0; i < w2.length; i++) s2 += (w2[i] as number) ** 2;
    expect(Math.sqrt(s2 / w2.length)).toBeCloseTo(0.02 / Math.sqrt(16), 3);
    expect(Array.from((lin.bias as Tensor).dataSync()).every((v) => v === 0)).toBe(true);
  });
});

describe("LayerNorm", () => {
  it("matches a hand-computed normalisation", () => {
    const ln = new LayerNorm(4, { eps: 1e-5 });
    setValues(ln.weight, [1, 2, 1, 2]);
    setValues(ln.bias as Tensor, [0, 0, 1, 1]);
    const x = fromArray(Float32Array.from([1, 2, 3, 4, 2, 2, 2, 2]), [2, 4]);
    const y = ln.forward(x);
    const expected: number[] = [];
    const rows = [[1, 2, 3, 4], [2, 2, 2, 2]];
    for (const r of rows) {
      const mean = r.reduce((a, b) => a + b, 0) / 4;
      const v = r.reduce((a, b) => a + (b - mean) ** 2, 0) / 4;
      const rstd = 1 / Math.sqrt(v + 1e-5);
      const gamma = [1, 2, 1, 2];
      const beta = [0, 0, 1, 1];
      r.forEach((xi, i) => expected.push((xi - mean) * rstd * (gamma[i] as number) + (beta[i] as number)));
    }
    expectClose(y, expected, 1e-4);
    const nb = new LayerNorm(4, { bias: false });
    expect(nb.parameters().length).toBe(1);
    expect(nb.stateDict().size).toBe(1); // zero beta is a non-persistent buffer
    expectClose(nb.forward(x), expected.map((e, i) => {
      const r = rows[Math.floor(i / 4)] as number[];
      const mean = r.reduce((a, b) => a + b, 0) / 4;
      const v = r.reduce((a, b) => a + (b - mean) ** 2, 0) / 4;
      return ((r[i % 4] as number) - mean) / Math.sqrt(v + 1e-5);
    }), 1e-4);
  });
});

describe("Embedding", () => {
  it("looks up rows", () => {
    const emb = new Embedding(4, 2);
    setValues(emb.weight, [0, 1, 10, 11, 20, 21, 30, 31]);
    const ids = fromArray(Int32Array.from([3, 0, 2, 2]), [2, 2], { dtype: "i32" });
    const out = emb.forward(ids);
    expect([...out.shape]).toEqual([2, 2, 2]);
    expectClose(out, [30, 31, 0, 1, 20, 21, 20, 21]);
  });
});

describe("CausalSelfAttention", () => {
  it("matches a naive loop reference (B=1, T=4, nHead=2)", () => {
    const C = 4;
    const nh = 2;
    const hd = C / nh;
    const T = 4;
    const attn = new CausalSelfAttention({ nEmbd: C, nHead: nh, blockSize: 8, dropout: 0 });
    attn.eval();
    const rng = new Random(42);
    const rnd = (n: number): number[] => Array.from({ length: n }, () => rng.uniformRange(-1, 1));
    const Wqkv = rnd(3 * C * C);
    const bqkv = rnd(3 * C);
    const Wp = rnd(C * C);
    const bp = rnd(C);
    const xv = rnd(T * C);
    setValues(attn.c_attn.weight, Wqkv);
    setValues(attn.c_attn.bias as Tensor, bqkv);
    setValues(attn.c_proj.weight, Wp);
    setValues(attn.c_proj.bias as Tensor, bp);
    const x = fromArray(Float32Array.from(xv), [1, T, C]);
    attn.captureAttention = true;
    const y = attn.forward(x);

    // Reference.
    const qkv: number[][] = [];
    for (let t = 0; t < T; t++) {
      const row: number[] = [];
      for (let o = 0; o < 3 * C; o++) {
        let s = bqkv[o] as number;
        for (let i = 0; i < C; i++) s += (xv[t * C + i] as number) * (Wqkv[o * C + i] as number);
        row.push(s);
      }
      qkv.push(row);
    }
    const merged: number[][] = Array.from({ length: T }, () => new Array<number>(C).fill(0));
    const attRef: number[][][] = [];
    for (let h = 0; h < nh; h++) {
      const att: number[][] = [];
      for (let i = 0; i < T; i++) {
        const scores: number[] = [];
        for (let j = 0; j < T; j++) {
          if (j > i) {
            scores.push(-Infinity);
            continue;
          }
          let s = 0;
          for (let d = 0; d < hd; d++) s += (qkv[i]![h * hd + d] as number) * (qkv[j]![C + h * hd + d] as number);
          scores.push(s / Math.sqrt(hd));
        }
        const m = Math.max(...scores);
        const e = scores.map((s) => (s === -Infinity ? 0 : Math.exp(s - m)));
        const z = e.reduce((a, b) => a + b, 0);
        att.push(e.map((v) => v / z));
      }
      attRef.push(att);
      for (let i = 0; i < T; i++) {
        for (let d = 0; d < hd; d++) {
          let s = 0;
          for (let j = 0; j < T; j++) s += (att[i]![j] as number) * (qkv[j]![2 * C + h * hd + d] as number);
          merged[i]![h * hd + d] = s;
        }
      }
    }
    const expected: number[] = [];
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < C; o++) {
        let s = bp[o] as number;
        for (let i = 0; i < C; i++) s += (merged[t]![i] as number) * (Wp[o * C + i] as number);
        expected.push(s);
      }
    }
    expect([...y.shape]).toEqual([1, T, C]);
    expectClose(y, expected, 1e-4);
    const captured = attn.lastAttention as Tensor;
    expect([...captured.shape]).toEqual([1, nh, T, T]);
    expectClose(captured, attRef.flat(2), 1e-4);
    // future positions are exactly zero
    const cap = captured.dataSync();
    for (let h = 0; h < nh; h++) for (let i = 0; i < T; i++) for (let j = i + 1; j < T; j++) expect(cap[(h * T + i) * T + j]).toBe(0);
  });

  it("rejects sequences longer than blockSize", () => {
    const attn = new CausalSelfAttention({ nEmbd: 4, nHead: 2, blockSize: 3 });
    expect(() => attn.forward(Tensor.zeros([1, 4, 4]))).toThrow(/blockSize/);
  });
});

describe("Block / Sequential", () => {
  it("keeps the shape and is differentiable", () => {
    const block = new Block({ nEmbd: 8, nHead: 2, blockSize: 16, dropout: 0.1, nLayer: 2 });
    const x = Tensor.randn([2, 5, 8], { requiresGrad: true });
    const y = block.forward(x);
    expect([...y.shape]).toEqual([2, 5, 8]);
    y.sum().backward();
    expect(x.grad).not.toBeNull();
    for (const p of block.parameters()) expect(p.grad).not.toBeNull();
    const seq = new Sequential(new Linear(8, 16), new GELU(), new Linear(16, 4));
    expect([...seq.forward(Tensor.randn([3, 8])).shape]).toEqual([3, 4]);
    expect([...seq.namedParameters().map(([n]) => n)]).toEqual(["0.weight", "0.bias", "2.weight", "2.bias"]);
  });
});

describe("Module", () => {
  it("names parameters like nanoGPT, counts tied weights once and lists both in stateDict", () => {
    const cfg = gptConfig("nano", { vocabSize: 16, blockSize: 8, bias: true });
    const gpt = new GPT(cfg);
    const names = gpt.namedParameters().map(([n]) => n);
    expect(names).toContain("transformer.wte.weight");
    expect(names).toContain("transformer.wpe.weight");
    expect(names).toContain("transformer.h.0.attn.c_attn.weight");
    expect(names).toContain("transformer.h.1.mlp.c_proj.bias");
    expect(names).toContain("transformer.ln_f.bias");
    expect(names).not.toContain("lm_head.weight"); // tied → counted once
    expect(gpt.lm_head.weight).toBe(gpt.transformer.wte.weight);
    const sd = gpt.stateDict();
    expect(sd.get("lm_head.weight")).toBe(sd.get("transformer.wte.weight"));
    expect(sd.has("transformer.h.0.attn.mask")).toBe(false); // non-persistent buffer
    const { nLayer, nEmbd: C } = GPT_PRESETS.nano;
    const perLayer = 12 * C * C + 13 * C;
    const expected = 16 * C + 8 * C + nLayer * perLayer + 2 * C;
    expect(gpt.numParameters(false)).toBe(expected);
    expect(gpt.numParameters()).toBe(expected - 8 * C);
    expect(gpt.parameters().length).toBe(new Set(gpt.parameters()).size);
  });

  it("round-trips stateDict and reports mismatches", () => {
    const a = new Linear(3, 2);
    const b = new Linear(3, 2);
    setValues(a.weight, [1, 2, 3, 4, 5, 6]);
    setValues(a.bias as Tensor, [7, 8]);
    b.loadStateDict(a.stateDict());
    expectClose(b.weight, [1, 2, 3, 4, 5, 6]);
    expectClose(b.bias as Tensor, [7, 8]);
    expect(b.weight).not.toBe(a.weight);
    expect(() => new Linear(3, 2, { bias: false }).loadStateDict(a.stateDict())).toThrow(/unexpected keys: bias/);
    expect(() => b.loadStateDict(new Map([["weight", a.weight]]))).toThrow(/missing keys: bias/);
    expect(() => new Linear(2, 2).loadStateDict(a.stateDict())).toThrow(/shape mismatch/);
    const result = new Linear(3, 2, { bias: false }).loadStateDict(a.stateDict(), false);
    expect(result.unexpected).toEqual(["bias"]);
  });

  it("to('cpu') is a no-op, train/eval propagate, zeroGrad frees grads, apply visits children first", () => {
    const block = new Block({ nEmbd: 4, nHead: 2, blockSize: 4 });
    const w = block.attn.c_attn.weight;
    expect(block.to("cpu")).toBe(block);
    expect(block.attn.c_attn.weight).toBe(w);
    block.eval();
    expect(block.attn.attn_dropout.training).toBe(false);
    block.train();
    expect(block.mlp.dropout.training).toBe(true);
    block.forward(Tensor.randn([1, 3, 4])).sum().backward();
    expect(w.grad).not.toBeNull();
    block.zeroGrad();
    expect(w.grad).toBeNull();
    const visited: string[] = [];
    block.apply((m) => visited.push(m.constructor.name));
    expect(visited[visited.length - 1]).toBe("Block");
    expect(visited.indexOf("Linear")).toBeLessThan(visited.indexOf("CausalSelfAttention"));
    expect(block.toString()).toContain("CausalSelfAttention");
  });

  it("auto-discovers module/tensor properties and ModuleList children", () => {
    class Custom extends Module {
      scale = Tensor.ones([2]);
      inner = new Linear(2, 2, { bias: false });
      list = new ModuleList([new Linear(1, 1, { bias: false }), new Linear(1, 1, { bias: false })]);
      constructor() {
        super();
        this.scale.requiresGrad = true;
      }
    }
    const m = new Custom();
    const names = m.namedParameters().map(([n]) => n);
    expect(names).toEqual(["scale", "inner.weight", "list.0.weight", "list.1.weight"]);
    expect(m.list.length).toBe(2);
    expect([...m.list].length).toBe(2);
  });
});
