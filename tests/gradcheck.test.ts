import { describe, expect, it } from "vitest";
import { Tensor, fromArray, ones, tensor, zeros } from "../src/tensor/tensor";
import { Random } from "../src/tensor/random";
import { noGrad } from "../src/autograd/graph";
import * as ops from "../src/autograd/ops";

const EPS = 1e-3;
const RTOL = 1e-2;
const ATOL = 1e-3;

type Fn = (...inputs: Tensor[]) => Tensor;

/** Deterministic input generator: values in [lo, hi) kept at least `margin` away from `avoid`. */
function mk(shape: number[], seed: number, lo = -1, hi = 1, avoid?: number, margin = 0.15): Tensor {
  const r = new Random(seed);
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = r.uniformRange(lo, hi);
    if (avoid !== undefined && Math.abs(v - avoid) < margin) v = avoid + (v >= avoid ? margin : -margin);
    data[i] = v;
  }
  return fromArray(data, shape, { requiresGrad: true });
}

/** Weights that probe every output element of `shape`. */
function probe(shape: readonly number[], seed: number): Tensor {
  const r = new Random(seed);
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = r.uniformRange(-1, 1) + (r.uniform() < 0.5 ? -0.5 : 0.5);
  return fromArray(data, shape);
}

/**
 * Central-difference gradient check of `fn` w.r.t. every input (loss = Σ fn(inputs) ⊙ w).
 * Returns the maximum violation ratio |analytic − numeric| / (atol + rtol·|numeric|) (≤ 1 passes).
 */
function gradcheck(fn: Fn, inputs: Tensor[], eps = EPS, rtol = RTOL, atol = ATOL): number {
  for (const t of inputs) {
    t.requiresGrad = true;
    t.grad = null;
  }
  const out = fn(...inputs);
  const w = probe(out.shape, 4242);
  const lossOf = (o: Tensor): Tensor => ops.sum(ops.mul(o, w));
  const loss = lossOf(out);
  expect(loss.requiresGrad).toBe(true);
  loss.backward();
  let worst = 0;
  for (let i = 0; i < inputs.length; i++) {
    const x = inputs[i] as Tensor;
    const analytic = x.grad;
    expect(analytic, `input ${i} received no gradient`).not.toBeNull();
    expect([...(analytic as Tensor).shape]).toEqual([...x.shape]);
    const a = (analytic as Tensor).dataSync();
    const data = x.dataSync() as Float32Array;
    for (let j = 0; j < data.length; j++) {
      const orig = data[j] as number;
      data[j] = orig + eps;
      const hp = data[j] as number;
      const fp = noGrad(() => lossOf(fn(...inputs)).itemSync());
      data[j] = orig - eps;
      const hm = data[j] as number;
      const fm = noGrad(() => lossOf(fn(...inputs)).itemSync());
      data[j] = orig;
      const numeric = (fp - fm) / (hp - hm);
      const ratio = Math.abs((a[j] as number) - numeric) / (atol + rtol * Math.abs(numeric));
      if (ratio > worst) worst = ratio;
      if (ratio > 1) {
        throw new Error(`gradcheck failed: input ${i} element ${j}: analytic ${a[j]} vs numeric ${numeric} (ratio ${ratio.toFixed(2)})`);
      }
    }
  }
  return worst;
}

const ids = (...v: number[]): Tensor => fromArray(new Int32Array(v));

describe("gradcheck: arithmetic", () => {
  it("add/sub/mul/div with broadcasting", () => {
    gradcheck((a, b) => ops.add(a, b), [mk([2, 3], 1), mk([3], 2)]);
    gradcheck((a, b) => ops.sub(a, b), [mk([2, 1], 3), mk([1, 3], 4)]);
    gradcheck((a, b) => ops.mul(a, b), [mk([2, 3], 5), mk([2, 3], 6)]);
    gradcheck((a, b) => ops.mul(a, b), [mk([4], 7), mk([2, 1, 4], 8)]);
    gradcheck((a, b) => ops.div(a, b), [mk([2, 3], 9), mk([3], 10, 0.5, 2)]);
    gradcheck((a, b) => ops.div(a, b), [mk([2, 3], 9), mk([], 10, 0.5, 2)]);
  });

  it("scalar overloads", () => {
    gradcheck((a) => a.add(2.5), [mk([5], 11)]);
    gradcheck((a) => a.sub(1.5), [mk([5], 12)]);
    gradcheck((a) => ops.sub(3, a), [mk([5], 13)]);
    gradcheck((a) => a.mul(-1.7), [mk([5], 14)]);
    gradcheck((a) => a.div(4), [mk([5], 15)]);
    gradcheck((a) => ops.div(2, a), [mk([5], 16, 0.5, 2)]);
    gradcheck((a) => a.pow(3), [mk([5], 17)]);
    gradcheck((a) => a.pow(0.5), [mk([5], 18, 0.5, 2)]);
    gradcheck((a) => ops.pow(2, a), [mk([5], 19)]);
  });

  it("pow / maximum / minimum (tensor-tensor)", () => {
    gradcheck((a, b) => ops.pow(a, b), [mk([4], 20, 0.5, 2), mk([4], 21, -1, 2)]);
    const a = mk([6], 22);
    const bData = a.dataSync().map((v, i) => v + (i % 2 === 0 ? 0.4 : -0.4));
    gradcheck((x, y) => ops.maximum(x, y), [a, fromArray(bData, [6])]);
    gradcheck((x, y) => ops.minimum(x, y), [mk([6], 23), fromArray(bData, [6])]);
    gradcheck((x) => ops.maximum(x, 0.1), [mk([6], 24, -1, 1, 0.1)]);
  });
});

describe("gradcheck: unary", () => {
  it("smooth ops", () => {
    gradcheck((x) => x.neg(), [mk([5], 30)]);
    gradcheck((x) => x.exp(), [mk([5], 31)]);
    gradcheck((x) => x.log(), [mk([5], 32, 0.5, 2)]);
    gradcheck((x) => x.sqrt(), [mk([5], 33, 0.5, 2)]);
    gradcheck((x) => x.rsqrt(), [mk([5], 34, 0.5, 2)]);
    gradcheck((x) => x.tanh(), [mk([5], 35)]);
    gradcheck((x) => x.sigmoid(), [mk([5], 36)]);
    gradcheck((x) => x.gelu(), [mk([7], 37, -2, 2)]);
    gradcheck((x) => x.square(), [mk([5], 38)]);
    gradcheck((x) => x.reciprocal(), [mk([5], 39, 0.5, 2)]);
    gradcheck((x) => x.sin(), [mk([5], 40)]);
    gradcheck((x) => x.cos(), [mk([5], 41)]);
  });

  it("piecewise ops away from kinks", () => {
    gradcheck((x) => x.relu(), [mk([8], 42, -1, 1, 0)]);
    gradcheck((x) => x.abs(), [mk([8], 43, -1, 1, 0)]);
    const c = mk([8], 44, -1, 1, 0.5);
    const cd = c.dataSync();
    for (let i = 0; i < cd.length; i++) if (Math.abs((cd[i] as number) + 0.5) < 0.15) cd[i] = -0.8;
    gradcheck((x) => x.clamp(-0.5, 0.5), [c]);
    gradcheck((x) => x.clamp(undefined, 0.5), [mk([8], 45, -1, 1, 0.5)]);
  });
});

describe("gradcheck: softmax and reductions", () => {
  it("softmax / logSoftmax along several dims", () => {
    gradcheck((x) => x.softmax(), [mk([2, 4], 50)]);
    gradcheck((x) => x.logSoftmax(), [mk([2, 4], 51)]);
    gradcheck((x) => x.softmax(0), [mk([3, 2], 52)]);
    gradcheck((x) => x.logSoftmax(1), [mk([2, 3, 2], 53)]);
  });

  it("sum / mean / max / min / var / std", () => {
    gradcheck((x) => x.sum(), [mk([2, 3], 60)]);
    gradcheck((x) => x.sum(1), [mk([2, 3], 61)]);
    gradcheck((x) => x.sum(0, true), [mk([2, 3], 62)]);
    gradcheck((x) => x.sum([0, 2]), [mk([2, 3, 2], 63)]);
    gradcheck((x) => x.mean(), [mk([2, 3], 64)]);
    gradcheck((x) => x.mean(-1, true), [mk([2, 3], 65)]);
    gradcheck((x) => x.max(), [mk([2, 3], 66)]);
    gradcheck((x) => x.max(1), [mk([3, 4], 67)]);
    gradcheck((x) => x.min(0, true), [mk([3, 4], 68)]);
    gradcheck((x) => x.var(), [mk([6], 69)]);
    gradcheck((x) => x.var(1, { unbiased: false }), [mk([2, 4], 70)]);
    gradcheck((x) => x.std(1, { keepDim: true }), [mk([2, 4], 71)]);
  });
});

describe("gradcheck: matmul", () => {
  it("2-D, vectors, batched broadcast, fused transposes", () => {
    gradcheck((a, b) => a.matmul(b), [mk([3, 4], 80), mk([4, 2], 81)]);
    gradcheck((a, b) => a.matmul(b), [mk([4], 82), mk([4, 3], 83)]);
    gradcheck((a, b) => a.matmul(b), [mk([3, 4], 84), mk([4], 85)]);
    gradcheck((a, b) => a.matmul(b), [mk([4], 86), mk([4], 87)]);
    gradcheck((a, b) => a.matmul(b), [mk([2, 3, 4], 88), mk([2, 4, 2], 89)]);
    gradcheck((a, b) => a.matmul(b), [mk([2, 1, 3, 4], 90), mk([2, 4, 2], 91)]);
    gradcheck((a, b) => a.matmul(b), [mk([3, 4], 92), mk([2, 4, 2], 93)]);
    // transposed views feed the transA/transB fast paths
    gradcheck((a, b) => a.transpose().matmul(b), [mk([4, 3], 94), mk([4, 2], 95)]);
    gradcheck((a, b) => a.matmul(b.transpose()), [mk([3, 4], 96), mk([2, 4], 97)]);
    gradcheck((a, b) => a.transpose().matmul(b.transpose()), [mk([4, 3], 98), mk([2, 4], 99)]);
    gradcheck((a, b) => a.transpose(1, 2).matmul(b), [mk([2, 4, 3], 100), mk([2, 4, 2], 101)]);
  });
});

describe("gradcheck: views", () => {
  it("transpose/permute/reshape/view/flatten/squeeze/unsqueeze", () => {
    gradcheck((x) => x.transpose().mul(tensor([[1, 2, 3], [4, 5, 6]])), [mk([3, 2], 110)]);
    gradcheck((x) => x.permute(2, 0, 1).exp(), [mk([2, 3, 2], 111)]);
    gradcheck((x) => x.reshape(3, 4).sigmoid(), [mk([2, 6], 112)]);
    gradcheck((x) => x.transpose().reshape(6).mul(tensor([1, 2, 3, 4, 5, 6])), [mk([2, 3], 113)]);
    gradcheck((x) => x.view(-1).tanh(), [mk([2, 3], 114)]);
    gradcheck((x) => x.flatten(1).sum(1), [mk([2, 2, 2], 115)]);
    gradcheck((x) => x.squeeze().mul(2), [mk([1, 3, 1], 116)]);
    gradcheck((x) => x.unsqueeze(1).mul(tensor([[1, 2, 3]])), [mk([2], 117)]);
  });

  it("slice/narrow/cat/stack/expand/contiguous/clone", () => {
    gradcheck((x) => x.slice(0, 1, 3), [mk([4, 2], 120)]);
    gradcheck((x) => x.slice(1, 0, undefined, 2).exp(), [mk([2, 5], 121)]);
    gradcheck((x) => x.slice(0, 1, undefined, 2).slice(1, 1, 3), [mk([5, 4], 122)]);
    gradcheck((x) => x.narrow(1, 1, 2), [mk([2, 4], 123)]);
    gradcheck((a, b) => ops.cat([a, b], 0), [mk([2, 3], 124), mk([1, 3], 125)]);
    gradcheck((a, b, c) => ops.cat([a, b.transpose(), c], 1).square(), [mk([2, 2], 126), mk([3, 2], 127), mk([2, 1], 128)]);
    gradcheck((a, b) => ops.stack([a, b], 1), [mk([2, 2], 129), mk([2, 2], 130)]);
    gradcheck((x) => x.expand(3, 2, 4).sin(), [mk([2, 1], 131)]);
    gradcheck((x) => x.broadcastTo(2, 3), [mk([3], 132)]);
    gradcheck((x) => x.transpose().contiguous().exp(), [mk([2, 3], 133)]);
    gradcheck((x) => x.clone().mul(x), [mk([4], 134)]);
  });
});

describe("gradcheck: indexing", () => {
  it("embedding / indexSelect / gather / scatterAdd", () => {
    gradcheck((t) => t.embedding(ids(2, 0, 2, 3)), [mk([4, 3], 140)]);
    gradcheck((t) => t.embedding(ids(1, 1, 0).reshape(1, 3)).tanh(), [mk([2, 2], 141)]);
    gradcheck((t) => t.indexSelect(1, ids(2, 0)), [mk([3, 3], 142)]);
    gradcheck((t) => t.gather(1, ids(1, 1, 0)).exp(), [mk([2, 2, 2], 143)]);
    gradcheck((t, s) => t.scatterAdd(ids(1, 0, 1), s), [mk([3, 2], 144), mk([3, 2], 145)]);
    gradcheck((t, s) => t.scatterAdd(ids(2, 2).reshape(1, 2), s).square(), [mk([3, 2], 146), mk([1, 2, 2], 147)]);
  });
});

describe("gradcheck: masking", () => {
  it("where / maskedFill / tril / triu", () => {
    const cond = tensor([[1, 0, 1], [0, 0, 1]]);
    gradcheck((a, b) => ops.where(cond, a, b), [mk([2, 3], 150), mk([2, 3], 151)]);
    gradcheck((a, b) => ops.where(cond, a, b), [mk([3], 152), mk([2, 1], 153)]);
    gradcheck((a) => ops.where(cond, a, 0.5), [mk([2, 3], 154)]);
    gradcheck((x) => x.maskedFill(tensor([0, 1, 0]), -2).exp(), [mk([2, 3], 155)]);
    gradcheck((x) => x.tril(), [mk([3, 3], 156)]);
    gradcheck((x) => x.triu(1).mul(3), [mk([2, 3, 3], 157)]);
  });
});

describe("gradcheck: fused ops", () => {
  it("crossEntropy mean/sum/none with ignoreIndex", () => {
    gradcheck((l) => l.crossEntropy(ids(2, 0, 1)), [mk([3, 4], 160)]);
    gradcheck((l) => l.crossEntropy(ids(2, 0, 1), { reduction: "sum" }), [mk([3, 4], 161)]);
    gradcheck((l) => l.crossEntropy(ids(2, 0, 1), { reduction: "none" }), [mk([3, 4], 162)]);
    gradcheck((l) => l.crossEntropy(ids(2, -100, 1)), [mk([3, 4], 163)]);
    gradcheck((l) => l.crossEntropy(ids(3, 0, 9, 1).reshape(2, 2), { ignoreIndex: 9 }), [mk([2, 2, 4], 164)]);
    gradcheck((l) => l.transpose().crossEntropy(ids(0, 1, 2, 3)), [mk([3, 4], 165)]);
  });

  it("layerNorm w.r.t. x, gamma and beta", () => {
    gradcheck((x, g, b) => x.layerNorm(g, b), [mk([2, 5], 170), mk([5], 171, 0.5, 1.5), mk([5], 172)]);
    gradcheck((x, g, b) => x.layerNorm(g, b, 1e-3).mul(x), [mk([2, 2, 3], 173), mk([3], 174, 0.5, 1.5), mk([3], 175)]);
    gradcheck((x, g, b) => x.transpose().layerNorm(g, b), [mk([3, 4], 176), mk([3], 177, 0.5, 1.5), mk([3], 178)]);
  });

  it("dropout with a fixed seed", () => {
    gradcheck((x) => x.dropout(0.4, true, 7), [mk([4, 4], 180)]);
    gradcheck((x) => x.dropout(0.4, false), [mk([4], 181)]);
  });

  it("composite: a causal self-attention head end to end", () => {
    const T = 3, D = 4;
    const mask = ones([T, T]).tril().eq(0);
    gradcheck(
      (x, wq, wk, wv) => {
        const q = x.matmul(wq);
        const k = x.matmul(wk);
        const v = x.matmul(wv);
        const att = q.matmul(k.transpose()).div(Math.sqrt(D)).maskedFill(mask, -Infinity).softmax();
        return att.matmul(v).layerNorm(ones([D]), zeros([D]));
      },
      [mk([T, D], 190), mk([D, D], 191), mk([D, D], 192), mk([D, D], 193)],
    );
  });
});

describe("autograd engine", () => {
  it("accumulates grads across backward calls, handles reuse and retainGrad", () => {
    const x = tensor([1, 2, 3], { requiresGrad: true });
    const y = x.mul(x); // dy/dx = 2x
    const h = y.retainGrad();
    y.sum().backward();
    expect(Array.from(x.grad!.dataSync())).toEqual([2, 4, 6]);
    expect(Array.from(h.grad!.dataSync())).toEqual([1, 1, 1]);
    x.mul(3).sum().backward();
    expect(Array.from(x.grad!.dataSync())).toEqual([5, 7, 9]);
    x.zeroGrad();
    expect(x.grad).toBeNull();
    expect(() => y.sum().backward()).toThrow();
  });

  it("explicit grad for non-scalar roots, leaf grad aliasing is safe", () => {
    const a = tensor([1, 2], { requiresGrad: true });
    const b = tensor([3, 4], { requiresGrad: true });
    const c = a.add(b);
    c.backward(tensor([10, 20]));
    expect(Array.from(a.grad!.dataSync())).toEqual([10, 20]);
    expect(Array.from(b.grad!.dataSync())).toEqual([10, 20]);
    expect(a.grad!.storage).not.toBe(b.grad!.storage);
    expect(() => c.backward()).toThrow();
    expect(() => tensor([1]).backward()).toThrow();
  });

  it("does not record under noGrad and stops at detach", () => {
    const x = tensor([1, 2], { requiresGrad: true });
    const y = x.mul(2).detach().mul(3);
    expect(y.requiresGrad).toBe(false);
    const z = noGrad(() => x.mul(2));
    expect(z.gradFn).toBeNull();
  });
});
