import { describe, expect, it } from "vitest";
import { Tensor, arange, eye, fromArray, full, linspace, ones, tensor, zeros } from "../src/tensor/tensor";
import { noGrad } from "../src/autograd/graph";
import * as ops from "../src/autograd/ops";

/** Assert tensor values (CPU) match `expected` within tolerance, and shape matches when given. */
function expectClose(t: Tensor, expected: ArrayLike<number>, shape?: number[], tol = 1e-5): void {
  if (shape) expect([...t.shape]).toEqual(shape);
  const got = t.dataSync();
  expect(got.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] as number;
    const g = got[i] as number;
    if (Number.isNaN(e)) expect(Number.isNaN(g)).toBe(true);
    else if (!Number.isFinite(e)) expect(g).toBe(e);
    else expect(Math.abs(g - e)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(e)));
  }
}

/** Naive reference matmul for a[M×K] · b[K×N] given as flat row-major arrays. */
function refMatmul(a: ArrayLike<number>, b: ArrayLike<number>, M: number, K: number, N: number): Float32Array {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += (a[i * K + k] as number) * (b[k * N + j] as number);
    out[i * N + j] = s;
  }
  return out;
}

describe("creation", () => {
  it("zeros/ones/full/eye/arange/linspace/fromArray/tensor", () => {
    expectClose(zeros([2, 2]), [0, 0, 0, 0], [2, 2]);
    expectClose(ones([3]), [1, 1, 1], [3]);
    expectClose(full([2], 7), [7, 7]);
    expectClose(eye(3), [1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3]);
    expectClose(eye(2, 3), [1, 0, 0, 0, 1, 0], [2, 3]);
    expectClose(arange(5), [0, 1, 2, 3, 4], [5]);
    expectClose(arange(1, 2, 0.25), [1, 1.25, 1.5, 1.75], [4]);
    expectClose(linspace(0, 1, 5), [0, 0.25, 0.5, 0.75, 1]);
    expectClose(fromArray([1, 2, 3, 4], [2, 2]), [1, 2, 3, 4], [2, 2]);
    expectClose(tensor([[1, 2], [3, 4], [5, 6]]), [1, 2, 3, 4, 5, 6], [3, 2]);
    expect(tensor(3).shape).toEqual([]);
    expect(arange(3, undefined, 1, { dtype: "i32" }).dtype).toBe("i32");
    expect(fromArray(new Int32Array([1, 2])).dtype).toBe("i32");
    expect(() => fromArray([1, 2, 3], [2, 2])).toThrow();
  });

  it("metadata and toString", () => {
    const t = tensor([[1, 2.5], [3, 4]]);
    expect(t.numel).toBe(4);
    expect(t.ndim).toBe(2);
    expect(t.isContiguous).toBe(true);
    expect([...t.strides]).toEqual([2, 1]);
    const s = t.toString();
    expect(s).toContain("1.0000");
    expect(s).toContain("shape=[2, 2]");
    expect(arange(3).toString()).toContain("[0., 1., 2.]");
    expect(arange(2000).toString()).toContain("...");
    expect(t.itemSync).toBeDefined();
    expect(tensor(5).itemSync()).toBe(5);
  });

  it("dataSync returns a live view for contiguous tensors and a copy for views", async () => {
    const t = arange(4);
    const d = t.dataSync();
    d[0] = 99;
    expect(t.dataSync()[0]).toBe(99);
    const tr = tensor([[1, 2], [3, 4]]).transpose();
    expectClose(tr, [1, 3, 2, 4], [2, 2]);
    expect(await tr.item === undefined).toBe(false);
    expect(await tensor(2.5).item()).toBe(2.5);
    expect(Array.from(await tr.data())).toEqual([1, 3, 2, 4]);
    expect(tr.toArray()).toEqual([[1, 3], [2, 4]]);
  });

  it("views share storage and refcounting frees once", () => {
    const t = arange(6);
    const v = t.reshape(2, 3);
    expect(v.storage).toBe(t.storage);
    expect(t.storage.refCount).toBe(2);
    v.free();
    expect(t.storage.isFreed).toBe(false);
    t.free();
    expect(t.storage.isFreed).toBe(true);
  });

  it("detach/clone/copy_/fill_", () => {
    const t = tensor([1, 2, 3], { requiresGrad: true });
    const d = t.detach();
    expect(d.requiresGrad).toBe(false);
    expect(d.storage).toBe(t.storage);
    const c = t.clone();
    expect(c.storage).not.toBe(t.storage);
    expectClose(c, [1, 2, 3]);
    noGrad(() => { c.copy_(tensor([4, 5, 6])); });
    expectClose(c, [4, 5, 6]);
    c.fill_(0);
    expectClose(c, [0, 0, 0]);
    expect(t.to("cpu")).toBe(t);
  });
});

describe("arithmetic", () => {
  const a = tensor([[1, 2, 3], [4, 5, 6]]);
  const b = tensor([10, 20, 30]);

  it("add/sub/mul/div/pow with tensors, scalars and broadcasting", () => {
    expectClose(a.add(b), [11, 22, 33, 14, 25, 36], [2, 3]);
    expectClose(a.add(2), [3, 4, 5, 6, 7, 8]);
    expectClose(ops.add(2, a), [3, 4, 5, 6, 7, 8]);
    expectClose(a.sub(b), [-9, -18, -27, -6, -15, -24]);
    expectClose(a.sub(1), [0, 1, 2, 3, 4, 5]);
    expectClose(ops.sub(10, a), [9, 8, 7, 6, 5, 4]);
    expectClose(a.mul(b), [10, 40, 90, 40, 100, 180]);
    expectClose(a.mul(0.5), [0.5, 1, 1.5, 2, 2.5, 3]);
    expectClose(a.div(b), [0.1, 0.1, 0.1, 0.4, 0.25, 0.2]);
    expectClose(a.div(2), [0.5, 1, 1.5, 2, 2.5, 3]);
    expectClose(ops.div(6, a), [6, 3, 2, 1.5, 1.2, 1]);
    expectClose(a.pow(2), [1, 4, 9, 16, 25, 36]);
    expectClose(ops.pow(2, tensor([0, 1, 3])), [1, 2, 8]);
    expectClose(tensor([2, 3]).pow(tensor([3, 2])), [8, 9]);
    // [2,1] + [1,3] → [2,3]
    expectClose(tensor([[1], [2]]).add(tensor([[10, 20, 30]])), [11, 21, 31, 12, 22, 32], [2, 3]);
    // scalar tensor broadcast
    expectClose(a.mul(tensor(2)), [2, 4, 6, 8, 10, 12]);
    expect(() => a.add(tensor([1, 2]))).toThrow();
  });

  it("works on non-contiguous inputs", () => {
    const at = a.transpose(); // [3,2] strides [1,3]
    expectClose(at.add(tensor([100, 200])), [101, 204, 102, 205, 103, 206], [3, 2]);
    expectClose(at.mul(at), [1, 16, 4, 25, 9, 36]);
    const sl = arange(10).slice(0, 2, 8, 2); // offset 2 → must materialise
    expectClose(sl.add(1), [3, 5, 7]);
    expectClose(sl.add(sl), [4, 8, 12]);
  });

  it("maximum/minimum/comparisons", () => {
    const x = tensor([1, 5, 3]);
    const y = tensor([4, 2, 3]);
    expectClose(x.maximum(y), [4, 5, 3]);
    expectClose(x.minimum(y), [1, 2, 3]);
    expectClose(x.maximum(2), [2, 5, 3]);
    expectClose(x.eq(y), [0, 0, 1]);
    expectClose(x.lt(y), [1, 0, 0]);
    expectClose(x.gt(y), [0, 1, 0]);
    expectClose(x.le(3), [1, 0, 1]);
    expectClose(x.ge(3), [0, 1, 1]);
    expectClose(ops.lt(2, x), [0, 1, 1]);
  });
});

describe("unary", () => {
  const x = tensor([-2, -0.5, 0, 0.5, 2]);
  const p = tensor([0.25, 1, 4, 9]);

  it("golden values", () => {
    expectClose(x.neg(), [2, 0.5, 0, -0.5, -2]);
    expectClose(x.exp(), [0.1353352832, 0.6065306597, 1, 1.6487212707, 7.3890560989]);
    expectClose(p.log(), [-1.3862943611, 0, 1.3862943611, 2.1972245773]);
    expectClose(p.sqrt(), [0.5, 1, 2, 3]);
    expectClose(p.rsqrt(), [2, 1, 0.5, 1 / 3]);
    expectClose(x.tanh(), [-0.9640275801, -0.4621171573, 0, 0.4621171573, 0.9640275801]);
    expectClose(x.sigmoid(), [0.1192029220, 0.3775406688, 0.5, 0.6224593312, 0.8807970780]);
    expectClose(x.relu(), [0, 0, 0, 0.5, 2]);
    // tanh-approximate GELU (PyTorch approximate="tanh")
    expectClose(x.gelu(), [-0.0454022884, -0.1542859972, 0, 0.3457140028, 1.9545977116], undefined, 1e-5);
    expectClose(x.square(), [4, 0.25, 0, 0.25, 4]);
    expectClose(x.abs(), [2, 0.5, 0, 0.5, 2]);
    expectClose(p.reciprocal(), [4, 1, 0.25, 1 / 9]);
    expectClose(x.sin(), [-0.9092974268, -0.4794255386, 0, 0.4794255386, 0.9092974268]);
    expectClose(x.cos(), [-0.4161468365, 0.8775825619, 1, 0.8775825619, -0.4161468365]);
    expectClose(x.clamp(-1, 1), [-1, -0.5, 0, 0.5, 1]);
    expectClose(x.clamp(undefined, 0), [-2, -0.5, 0, 0, 0]);
    expectClose(x.clamp(0), [0, 0, 0, 0.5, 2]);
  });

  it("unary on a transposed view", () => {
    const t = tensor([[1, 4], [9, 16]]).transpose();
    expectClose(t.sqrt(), [1, 3, 2, 4], [2, 2]);
  });
});

describe("softmax", () => {
  it("softmax / logSoftmax along last dim and dim 0", () => {
    const x = tensor([[1, 2, 3], [1, 1, 1]]);
    const e = [0.0900305732, 0.2447284711, 0.6652409558, 1 / 3, 1 / 3, 1 / 3];
    expectClose(x.softmax(), e, [2, 3]);
    expectClose(x.logSoftmax(), e.map(Math.log));
    // dim 0: columns (1,1) (2,1) (3,1)
    expectClose(x.softmax(0), [0.5, 0.7310585786, 0.8807970780, 0.5, 0.2689414214, 0.1192029220], [2, 3]);
    // numerically stable with large values
    expectClose(tensor([1000, 1000]).softmax(), [0.5, 0.5]);
    expectClose(tensor([1000, 0]).logSoftmax(), [0, -1000]);
  });
});

describe("reductions", () => {
  const x = tensor([[1, 5, 3], [4, 2, 6]]);

  it("sum/mean/max/min over axes and keepDim", () => {
    expectClose(x.sum(), [21], []);
    expectClose(x.sum(0), [5, 7, 9], [3]);
    expectClose(x.sum(1), [9, 12], [2]);
    expectClose(x.sum(1, true), [9, 12], [2, 1]);
    expectClose(x.sum([0, 1]), [21], []);
    expectClose(x.mean(), [3.5], []);
    expectClose(x.mean(0), [2.5, 3.5, 4.5]);
    expectClose(x.mean(-1, true), [3, 4], [2, 1]);
    expectClose(x.max(), [6], []);
    expectClose(x.max(1), [5, 6]);
    expectClose(x.max(0), [4, 5, 6]);
    expectClose(x.min(1), [1, 2]);
    expectClose(x.min(0, true), [1, 2, 3], [1, 3]);
    // 3-D reduce over middle axis
    const y = arange(24).reshape(2, 3, 4);
    expectClose(y.sum(1), [12, 15, 18, 21, 48, 51, 54, 57], [2, 4]);
    expectClose(y.sum([0, 2]), [60, 92, 124], [3]);
    expectClose(y.max(2), [3, 7, 11, 15, 19, 23], [2, 3]);
    // reduction on a transposed view
    expectClose(x.transpose().sum(0), [9, 12]);
  });

  it("var/std (unbiased default) and argmax", () => {
    const v = tensor([1, 2, 3, 4]);
    expectClose(v.var(), [1.6666666667], []);
    expectClose(v.var(undefined, { unbiased: false }), [1.25]);
    expectClose(v.std(), [1.2909944487]);
    expectClose(x.var(1), [4, 4]);
    expectClose(x.std(1, { keepDim: true }), [2, 2], [2, 1]);
    const am = x.argmax(1);
    expect(am.dtype).toBe("i32");
    expect(Array.from(am.dataSync())).toEqual([1, 2]);
    expect(Array.from(x.argmax(0).dataSync())).toEqual([1, 0, 1]);
    expect(Array.from(x.argmax().dataSync())).toEqual([5]);
    expect([...x.argmax(1, true).shape]).toEqual([2, 1]);
  });
});

describe("matmul", () => {
  it("2-D, vector and scalar forms", () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    const b = tensor([[7, 8], [9, 10], [11, 12]]);
    expectClose(a.matmul(b), [58, 64, 139, 154], [2, 2]);
    expectClose(tensor([1, 2, 3]).matmul(b), [58, 64], [2]);
    expectClose(a.matmul(tensor([1, 1, 1])), [6, 15], [2]);
    expectClose(tensor([1, 2, 3]).matmul(tensor([4, 5, 6])), [32], []);
    expect(() => a.matmul(a)).toThrow();
  });

  it("transposed views are fused (all four layouts) and match a naive reference", () => {
    const M = 5, K = 7, N = 6;
    const A = Tensor.randn([M, K], { seed: 1 });
    const B = Tensor.randn([K, N], { seed: 2 });
    const ref = refMatmul(A.dataSync(), B.dataSync(), M, K, N);
    const At = A.transpose().contiguous(); // stored K×M
    const Bt = B.transpose().contiguous(); // stored N×K
    expectClose(A.matmul(B), ref, [M, N], 1e-4);
    expectClose(A.matmul(Bt.transpose()), ref, [M, N], 1e-4);
    expectClose(At.transpose().matmul(B), ref, [M, N], 1e-4);
    expectClose(At.transpose().matmul(Bt.transpose()), ref, [M, N], 1e-4);
  });

  it("batched with broadcasting over batch dims", () => {
    const A = Tensor.randn([2, 3, 4], { seed: 3 });
    const B = Tensor.randn([2, 4, 5], { seed: 4 });
    const out = A.matmul(B);
    expect([...out.shape]).toEqual([2, 3, 5]);
    const ad = A.dataSync(), bd = B.dataSync();
    for (let i = 0; i < 2; i++) {
      const ref = refMatmul(ad.subarray(i * 12, i * 12 + 12), bd.subarray(i * 20, i * 20 + 20), 3, 4, 5);
      expectClose(out.slice(0, i, i + 1).squeeze(0), ref, [3, 5], 1e-4);
    }
    // [2,1,3,4] @ [1,3,4,5] → [2,3,3,5]
    const A2 = A.unsqueeze(1);
    const B2 = Tensor.randn([3, 4, 5], { seed: 5 }).unsqueeze(0);
    const out2 = A2.matmul(B2);
    expect([...out2.shape]).toEqual([2, 3, 3, 5]);
    const b2d = B2.dataSync();
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
      const ref = refMatmul(ad.subarray(i * 12, i * 12 + 12), b2d.subarray(j * 20, j * 20 + 20), 3, 4, 5);
      expectClose(out2.slice(0, i, i + 1).slice(1, j, j + 1).reshape(3, 5), ref, [3, 5], 1e-4);
    }
    // broadcast a single matrix against a batch, with a batched transposed view
    const W = Tensor.randn([4, 6], { seed: 6 });
    const Bt = B.transpose(1, 2); // [2,5,4] view
    const out3 = Bt.matmul(W); // [2,5,6]
    const wd = W.dataSync();
    for (let i = 0; i < 2; i++) {
      const bt = Bt.slice(0, i, i + 1).squeeze(0).contiguous().dataSync();
      expectClose(out3.slice(0, i, i + 1).squeeze(0), refMatmul(bt, wd, 5, 4, 6), [5, 6], 1e-4);
    }
  });

  it("larger sizes stay accurate (exercises tiling and unroll remainders)", () => {
    for (const [M, K, N] of [[70, 130, 65], [1, 100, 3], [33, 1, 17]] as const) {
      const A = Tensor.randn([M, K], { seed: M });
      const B = Tensor.randn([K, N], { seed: N });
      const ref = refMatmul(A.dataSync(), B.dataSync(), M, K, N);
      expectClose(A.matmul(B), ref, [M, N], 1e-4);
      expectClose(A.matmul(B.transpose().contiguous().transpose()), ref, [M, N], 1e-4);
      expectClose(A.transpose().contiguous().transpose().matmul(B), ref, [M, N], 1e-4);
    }
  });
});

describe("views", () => {
  const x = arange(24).reshape(2, 3, 4);

  it("transpose/permute/reshape/view/flatten", () => {
    const t = x.transpose(0, 2);
    expect([...t.shape]).toEqual([4, 3, 2]);
    expect(t.isContiguous).toBe(false);
    expect(t.storage).toBe(x.storage);
    expectClose(t.contiguous().slice(0, 0, 1), [0, 12, 4, 16, 8, 20], [1, 3, 2]);
    const p = x.permute(2, 0, 1);
    expect([...p.shape]).toEqual([4, 2, 3]);
    expectClose(p.contiguous().slice(0, 1, 2), [1, 5, 9, 13, 17, 21], [1, 2, 3]);
    expectClose(x.reshape(6, 4).slice(0, 1, 2), [4, 5, 6, 7], [1, 4]);
    expectClose(x.reshape(-1, 12).slice(0, 1, 2), arange(12, 24).dataSync(), [1, 12]);
    expect(() => t.view(24)).toThrow();
    expectClose(t.reshape(24).slice(0, 0, 3), [0, 12, 4]);
    expect([...x.flatten().shape]).toEqual([24]);
    expect([...x.flatten(1).shape]).toEqual([2, 12]);
    expect([...x.flatten(0, 1).shape]).toEqual([6, 4]);
    expect(tensor([[1, 2], [3, 4]]).T.toArray()).toEqual([[1, 3], [2, 4]]);
  });

  it("squeeze/unsqueeze/expand/broadcastTo", () => {
    const s = ones([1, 3, 1]);
    expect([...s.squeeze().shape]).toEqual([3]);
    expect([...s.squeeze(0).shape]).toEqual([3, 1]);
    expect([...s.squeeze(1).shape]).toEqual([1, 3, 1]);
    expect([...s.unsqueeze(0).shape]).toEqual([1, 1, 3, 1]);
    expect([...s.unsqueeze(-1).shape]).toEqual([1, 3, 1, 1]);
    const e = tensor([[1], [2]]).expand(2, 3);
    expect([...e.strides]).toEqual([1, 0]);
    expectClose(e.contiguous(), [1, 1, 1, 2, 2, 2], [2, 3]);
    expectClose(tensor([1, 2]).broadcastTo(2, 2), [1, 2, 1, 2], [2, 2]);
    expectClose(tensor([1, 2]).expand(-1).contiguous(), [1, 2]);
  });

  it("slice/narrow with steps and negatives", () => {
    const v = arange(10);
    expectClose(v.slice(0, 2, 7), [2, 3, 4, 5, 6]);
    expectClose(v.slice(0, 1, undefined, 3), [1, 4, 7]);
    expectClose(v.slice(0, -3), [7, 8, 9]);
    expectClose(v.narrow(0, 4, 3), [4, 5, 6]);
    expectClose(x.slice(2, 1, 3), [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22], [2, 3, 2]);
    expectClose(x.slice(1, 1, 2).slice(2, 0, 2), [4, 5, 16, 17], [2, 1, 2]);
    expect(() => v.narrow(0, 8, 5)).toThrow();
  });

  it("cat/stack along every dim, including i32 and non-contiguous inputs", () => {
    const a = tensor([[1, 2], [3, 4]]);
    const b = tensor([[5, 6]]);
    expectClose(ops.cat([a, b], 0), [1, 2, 3, 4, 5, 6], [3, 2]);
    expectClose(ops.cat([a, a.transpose()], 1), [1, 2, 1, 3, 3, 4, 2, 4], [2, 4]);
    expectClose(ops.cat([a, b.transpose()], -1), [1, 2, 5, 3, 4, 6], [2, 3]);
    expectClose(ops.stack([a, a.mul(10)], 0), [1, 2, 3, 4, 10, 20, 30, 40], [2, 2, 2]);
    expectClose(ops.stack([tensor([1, 2]), tensor([3, 4])], 1), [1, 3, 2, 4], [2, 2]);
    const ids = ops.cat([fromArray(new Int32Array([1, 2])), fromArray(new Int32Array([3]))]);
    expect(ids.dtype).toBe("i32");
    expect(Array.from(ids.dataSync())).toEqual([1, 2, 3]);
    expect(() => ops.cat([a, tensor([1, 2, 3])])).toThrow();
  });
});

describe("indexing", () => {
  it("embedding / indexSelect / gather", () => {
    const table = tensor([[0, 0], [1, 10], [2, 20], [3, 30]]);
    const ids = fromArray(new Int32Array([3, 1, 1]));
    expectClose(table.embedding(ids), [3, 30, 1, 10, 1, 10], [3, 2]);
    expectClose(table.embedding(ids.reshape(1, 3)), [3, 30, 1, 10, 1, 10], [1, 3, 2]);
    expectClose(table.indexSelect(0, ids), [3, 30, 1, 10, 1, 10], [3, 2]);
    expectClose(table.indexSelect(1, fromArray(new Int32Array([1, 0]))), [0, 0, 10, 1, 20, 2, 30, 3], [4, 2]);
    expectClose(table.gather(1, fromArray(new Int32Array([1]))), [0, 10, 20, 30], [4, 1]);
    const x = arange(24).reshape(2, 3, 4);
    expectClose(x.indexSelect(1, fromArray(new Int32Array([2, 0]))), [8, 9, 10, 11, 0, 1, 2, 3, 20, 21, 22, 23, 12, 13, 14, 15], [2, 2, 4]);
    expectClose(x.indexSelect(2, fromArray(new Int32Array([3]))), [3, 7, 11, 15, 19, 23], [2, 3, 1]);
    expect(() => table.embedding(tensor([1, 2]))).toThrow();
  });

  it("scatterAdd accumulates duplicates", () => {
    const target = zeros([3, 2]);
    const ids = fromArray(new Int32Array([2, 0, 2]));
    const src = tensor([[1, 2], [3, 4], [5, 6]]);
    expectClose(target.scatterAdd(ids, src), [3, 4, 0, 0, 6, 8], [3, 2]);
    expectClose(target, [0, 0, 0, 0, 0, 0]); // out-of-place
  });
});

describe("selection and masking", () => {
  it("where/maskedFill/tril/triu", () => {
    const cond = tensor([1, 0, 1]);
    expectClose(ops.where(cond, tensor([1, 2, 3]), tensor([10, 20, 30])), [1, 20, 3]);
    expectClose(ops.where(cond, tensor([[1, 2, 3], [4, 5, 6]]), 0), [1, 0, 3, 4, 0, 6], [2, 3]);
    expectClose(ops.where(fromArray(new Uint32Array([0, 1])), 1, 2), [2, 1]);
    const x = tensor([[1, 2], [3, 4]]);
    expectClose(x.maskedFill(tensor([[0, 1], [1, 0]]), -Infinity), [1, -Infinity, -Infinity, 4]);
    expectClose(x.maskedFill(tensor([0, 1]), 9), [1, 9, 3, 9]);
    const m = arange(9).reshape(3, 3);
    expectClose(m.tril(), [0, 0, 0, 3, 4, 0, 6, 7, 8]);
    expectClose(m.tril(-1), [0, 0, 0, 3, 0, 0, 6, 7, 0]);
    expectClose(m.triu(), [0, 1, 2, 0, 4, 5, 0, 0, 8]);
    expectClose(m.triu(1), [0, 1, 2, 0, 0, 5, 0, 0, 0]);
    expectClose(ops.tril(ones([2, 2, 3])), [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], [2, 2, 3]);
    // causal attention mask pattern: masked_fill(scores, tril==0, -inf) then softmax
    const scores = zeros([3, 3]);
    const causal = ones([3, 3]).tril().eq(0);
    const att = scores.maskedFill(causal, -Infinity).softmax();
    expectClose(att, [1, 0, 0, 0.5, 0.5, 0, 1 / 3, 1 / 3, 1 / 3]);
  });
});

describe("fused losses / normalisation", () => {
  it("crossEntropy golden values with mean/sum/none and ignoreIndex", () => {
    const logits = tensor([[1, 2, 3], [1, 1, 1]]);
    const t = fromArray(new Int32Array([2, 0]));
    const row0 = Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)) - 3;
    const row1 = Math.log(3);
    expectClose(logits.crossEntropy(t), [(row0 + row1) / 2], []);
    expectClose(logits.crossEntropy(t, { reduction: "sum" }), [row0 + row1]);
    expectClose(logits.crossEntropy(t, { reduction: "none" }), [row0, row1], [2]);
    expectClose(logits.crossEntropy(fromArray(new Int32Array([2, -100]))), [row0]);
    expectClose(logits.crossEntropy(fromArray(new Int32Array([2, 7])), { ignoreIndex: 7 }), [row0]);
    // 3-D logits [B,T,C] with [B,T] targets
    const l3 = logits.reshape(1, 2, 3);
    expectClose(l3.crossEntropy(t.reshape(1, 2)), [(row0 + row1) / 2]);
    expectClose(l3.crossEntropy(t.reshape(1, 2), { reduction: "none" }), [row0, row1], [1, 2]);
    // stable with huge logits
    expectClose(tensor([[1000, 0]]).crossEntropy(fromArray(new Int32Array([0]))), [0]);
  });

  it("layerNorm golden values", () => {
    const x = tensor([[1, 2, 3, 4], [2, 2, 2, 2]]);
    const g = ones([4]);
    const b = zeros([4]);
    const r = Math.sqrt(1.25);
    expectClose(x.layerNorm(g, b, 1e-12), [-1.5 / r, -0.5 / r, 0.5 / r, 1.5 / r, 0, 0, 0, 0], [2, 4]);
    expectClose(x.layerNorm(tensor([1, 2, 3, 4]), tensor([10, 10, 10, 10]), 0).slice(0, 0, 1), [10 - 1.5 / r, 10 - 1 / r, 10 + 1.5 / r, 10 + 6 / r]);
    // eps=1 on the first row: rstd = 1/sqrt(1.25 + 1)
    const r2 = Math.sqrt(2.25);
    expectClose(x.layerNorm(g, b, 1).slice(0, 0, 1), [-1.5 / r2, -0.5 / r2, 0.5 / r2, 1.5 / r2]);
  });
});

describe("grad mode", () => {
  it("noGrad disables recording and restores state", () => {
    const x = tensor([1, 2], { requiresGrad: true });
    const y = noGrad(() => x.mul(2));
    expect(y.requiresGrad).toBe(false);
    expect(y.gradFn).toBeNull();
    const z = x.mul(2);
    expect(z.requiresGrad).toBe(true);
    expect(z.gradFn?.name).toBe("mul");
    expect(() => noGrad(() => { throw new Error("boom"); })).toThrow("boom");
    expect(x.mul(3).requiresGrad).toBe(true);
  });
});
