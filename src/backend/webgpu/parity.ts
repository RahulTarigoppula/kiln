/**
 * Backend-level GPU vs CPU parity harness.
 *
 * For every `Backend` method, random inputs (odd sizes, broadcasting, batch, transposes, duplicate ids ...)
 * are run on both backends and compared. Usable from the browser Tests page and from the Playwright
 * script (`scripts/gpu-parity.mjs`) via {@link runParityAndPublish}, which stores the outcome on
 * `globalThis.__kilnParityDone`.
 *
 * Thresholds: 1e-4 by default, 1e-3 for matmul with K > 256, reductions scaled by sqrt(n).
 */

import type { Backend, BinaryOp, Buf, ReduceOp, UnaryOp } from "../types";
import { cpuBackend } from "../cpu/index";
import { Random } from "../../tensor/random";
import { createWebGpuBackend, isWebGpuAvailable } from "./index";

/** One comparison outcome. */
export interface ParityResult {
  op: string;
  case: string;
  maxAbs: number;
  maxRel: number;
  threshold: number;
  pass: boolean;
  /** Wall time of the GPU path including readback, ms. */
  ms: number;
  /** Set when the case threw instead of producing numbers. */
  error?: string;
}

/** Options for {@link runParity}. */
export interface ParityOptions {
  seed?: number;
  /** Called after each case (for live UI updates). */
  onResult?: (r: ParityResult) => void;
  /** Restrict to ops whose name includes one of these substrings. */
  only?: string[];
}

const DEFAULT_THR = 1e-4;

interface Pair {
  c: Buf;
  g: Buf;
}

class Harness {
  readonly results: ParityResult[] = [];
  readonly rng: Random;
  private live: Pair[] = [];

  constructor(readonly cpu: Backend, readonly gpu: Backend, readonly opts: ParityOptions) {
    this.rng = new Random(opts.seed ?? 20241018);
  }

  // ---- data generation
  f32(n: number, lo = -3, hi = 3): Float32Array {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = this.rng.uniformRange(lo, hi);
    return a;
  }
  ints(n: number, lo: number, hi: number): Int32Array {
    const a = new Int32Array(n);
    for (let i = 0; i < n; i++) a[i] = lo + this.rng.int(hi - lo);
    return a;
  }
  intsF32(n: number, lo: number, hi: number): Float32Array {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = lo + this.rng.int(hi - lo);
    return a;
  }

  // ---- buffers
  upload(data: Float32Array | Int32Array | Uint32Array): Pair {
    const c = this.cpu.alloc(data.byteLength);
    const g = this.gpu.alloc(data.byteLength);
    this.cpu.write(c, data);
    this.gpu.write(g, data);
    const p = { c, g };
    this.live.push(p);
    return p;
  }
  alloc(bytes: number): Pair {
    const p = { c: this.cpu.alloc(bytes), g: this.gpu.alloc(bytes) };
    // Make both deterministic (zero) so untouched regions compare equal.
    this.cpu.fill(p.c, bytes / 4, 0);
    this.gpu.fill(p.g, bytes / 4, 0);
    this.live.push(p);
    return p;
  }
  private release(): void {
    for (const p of this.live) {
      this.cpu.free(p.c);
      this.gpu.free(p.g);
    }
    this.live = [];
  }

  /** Run `body` on both backends, then compare each listed output. */
  async run(
    op: string,
    caseName: string,
    body: (b: Backend, pick: (p: Pair) => Buf) => void,
    outputs: { pair: Pair; n: number; dtype?: "f32" | "i32"; threshold?: number; label?: string }[],
  ): Promise<void> {
    if (this.opts.only && !this.opts.only.some((s) => op.includes(s))) {
      this.release();
      return;
    }
    const t0 = performance.now();
    try {
      body(this.cpu, (p) => p.c);
      body(this.gpu, (p) => p.g);
      for (const o of outputs) {
        const thr = o.threshold ?? DEFAULT_THR;
        const [cb, gb] = await Promise.all([this.cpu.read(o.pair.c, o.n * 4), this.gpu.read(o.pair.g, o.n * 4)]);
        const { maxAbs, maxRel } = compare(cb, gb, o.dtype ?? "f32");
        const pass = maxAbs <= thr || maxRel <= thr;
        this.push({ op, case: o.label ? `${caseName} [${o.label}]` : caseName, maxAbs, maxRel, threshold: thr, pass, ms: performance.now() - t0 });
      }
    } catch (err: unknown) {
      this.push({
        op,
        case: caseName,
        maxAbs: NaN,
        maxRel: NaN,
        threshold: DEFAULT_THR,
        pass: false,
        ms: performance.now() - t0,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      this.release();
    }
  }

  private push(r: ParityResult): void {
    this.results.push(r);
    this.opts.onResult?.(r);
  }
}

function compare(a: ArrayBuffer, b: ArrayBuffer, dtype: "f32" | "i32"): { maxAbs: number; maxRel: number } {
  const A = dtype === "f32" ? new Float32Array(a) : new Int32Array(a);
  const B = dtype === "f32" ? new Float32Array(b) : new Int32Array(b);
  let maxAbs = 0;
  let maxMag = 0;
  const n = Math.min(A.length, B.length);
  for (let i = 0; i < n; i++) {
    const x = A[i] as number;
    const y = B[i] as number;
    if (Number.isNaN(x) && Number.isNaN(y)) continue;
    if (x === y) {
      maxMag = Math.max(maxMag, Math.abs(x));
      continue;
    }
    if (!Number.isFinite(x) || !Number.isFinite(y)) return { maxAbs: Infinity, maxRel: Infinity };
    const d = Math.abs(x - y);
    if (d > maxAbs) maxAbs = d;
    maxMag = Math.max(maxMag, Math.abs(x));
  }
  if (A.length !== B.length) return { maxAbs: Infinity, maxRel: Infinity };
  return { maxAbs, maxRel: maxMag > 0 ? maxAbs / maxMag : maxAbs };
}

function strides(shape: number[]): number[] {
  const s = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i] ?? 1;
  }
  return s;
}
function prod(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

const UNARY_OPS: UnaryOp[] = [
  "neg", "exp", "log", "sqrt", "rsqrt", "tanh", "sigmoid", "relu", "gelu", "square", "abs", "recip", "sin", "cos",
  "scale", "addScalar", "powScalar", "geluGrad", "tanhGrad", "sigmoidGrad", "reluGrad", "clamp",
];
const POSITIVE_ONLY = new Set<UnaryOp>(["log", "sqrt", "rsqrt", "recip", "powScalar"]);
const BINARY_OPS: BinaryOp[] = ["add", "sub", "mul", "div", "pow", "max", "min", "eq", "lt", "gt", "le", "ge"];
const REDUCE_OPS: ReduceOp[] = ["sum", "max", "min"];

/** Run the full parity suite. `gpu` defaults to a freshly created WebGPU backend. */
export async function runParity(gpu: Backend, opts: ParityOptions = {}, cpu: Backend = cpuBackend): Promise<ParityResult[]> {
  const H = new Harness(cpu, gpu, opts);

  // ---------------------------------------------------------------- unary
  for (const op of UNARY_OPS) {
    const n = 1001;
    const x = POSITIVE_ONLY.has(op) ? H.f32(n, 0.5, 2.5) : H.f32(n);
    const alpha = op === "scale" ? 1.7 : op === "addScalar" ? -0.3 : op === "powScalar" ? 2.5 : op === "clamp" ? -0.5 : 0;
    const beta = op === "clamp" ? 0.8 : 0;
    const a = H.upload(x);
    const out = H.alloc(n * 4);
    await H.run("unary", op, (b, p) => b.unary(op, p(a), n, p(out), alpha, beta), [{ pair: out, n }]);
  }
  {
    // integer exponent with negative base
    const n = 513;
    const a = H.upload(H.f32(n));
    const out = H.alloc(n * 4);
    await H.run("unary", "powScalar(x, 3) negative base", (b, p) => b.unary("powScalar", p(a), n, p(out), 3), [{ pair: out, n }]);
  }

  // --------------------------------------------------------------- binary
  for (const op of BINARY_OPS) {
    const positive = op === "div" || op === "pow";
    const cmp = op === "eq" || op === "lt" || op === "gt" || op === "le" || op === "ge";
    const gen = (n: number): Float32Array => (cmp ? H.intsF32(n, -2, 3) : positive ? H.f32(n, 0.5, 2) : H.f32(n));
    // same shape
    {
      const shape = [3, 5, 7];
      const n = prod(shape);
      const a = H.upload(gen(n));
      const bb = H.upload(gen(n));
      const out = H.alloc(n * 4);
      await H.run("binary", `${op} same [3,5,7]`, (b, p) => b.binary(op, p(a), p(bb), p(out), shape, strides(shape), strides(shape)), [{ pair: out, n }]);
    }
    // broadcast [4,1,6] ⊕ [1,5,1] -> [4,5,6]
    {
      const outShape = [4, 5, 6];
      const a = H.upload(gen(24));
      const bb = H.upload(gen(5));
      const out = H.alloc(120 * 4);
      await H.run("binary", `${op} broadcast [4,1,6]+[1,5,1]`, (b, p) => b.binary(op, p(a), p(bb), p(out), outShape, [6, 0, 1], [0, 1, 0]), [{ pair: out, n: 120 }]);
    }
    // scalar broadcast + strided (transposed) a
    {
      const outShape = [6, 9];
      const a = H.upload(gen(54)); // stored as [9,6], read transposed
      const bb = H.upload(gen(1));
      const out = H.alloc(54 * 4);
      await H.run("binary", `${op} transposed a + scalar b`, (b, p) => b.binary(op, p(a), p(bb), p(out), outShape, [1, 6], [0, 0]), [{ pair: out, n: 54 }]);
    }
  }
  {
    // rank 5 broadcast
    const outShape = [2, 3, 4, 5, 6];
    const n = prod(outShape);
    const a = H.upload(H.f32(2 * 1 * 4 * 1 * 6));
    const bb = H.upload(H.f32(1 * 3 * 1 * 5 * 1));
    const out = H.alloc(n * 4);
    await H.run("binary", "add rank-5 broadcast", (b, p) => b.binary("add", p(a), p(bb), p(out), outShape, [24, 0, 6, 0, 1], [0, 5, 0, 1, 0]), [{ pair: out, n }]);
  }

  // ---------------------------------------------------------- copyStrided
  {
    const a = H.upload(H.f32(35));
    const out = H.alloc(35 * 4);
    await H.run("copyStrided", "transpose [5,7]", (b, p) => b.copyStrided(p(a), [7, 5], [1, 7], 0, p(out)), [{ pair: out, n: 35 }]);
  }
  {
    const a = H.upload(H.f32(100));
    const out = H.alloc(12 * 4);
    await H.run("copyStrided", "slice rows 3..6 cols 2..6 of [10,10]", (b, p) => b.copyStrided(p(a), [3, 4], [10, 1], 32, p(out)), [{ pair: out, n: 12 }]);
  }
  {
    const a = H.upload(H.f32(2 * 3 * 4 * 5));
    const out = H.alloc(120 * 4);
    await H.run("copyStrided", "permute [2,3,4,5]->[5,2,4,3]", (b, p) => b.copyStrided(p(a), [5, 2, 4, 3], [1, 60, 5, 20], 0, p(out)), [{ pair: out, n: 120 }]);
  }

  // --------------------------------------------------------- where / fill
  {
    const n = 777;
    const cond = H.upload(H.intsF32(n, 0, 2));
    const a = H.upload(H.f32(n));
    const bb = H.upload(H.f32(n));
    const out = H.alloc(n * 4);
    await H.run("where", "f32 cond", (b, p) => b.where(p(cond), p(a), p(bb), n, p(out)), [{ pair: out, n }]);
  }
  {
    const n = 777;
    const condU = new Uint32Array(n);
    for (let i = 0; i < n; i++) condU[i] = H.rng.int(2);
    const cond = H.upload(condU);
    const a = H.upload(H.f32(n));
    const bb = H.upload(H.f32(n));
    const out = H.alloc(n * 4);
    await H.run("where", "u32 cond", (b, p) => b.where(p(cond), p(a), p(bb), n, p(out)), [{ pair: out, n }]);
  }
  {
    const n = 999;
    const mask = H.upload(H.intsF32(n, 0, 2));
    const a = H.upload(H.f32(n));
    const out = H.alloc(n * 4);
    await H.run("maskedFill", "value -1e9", (b, p) => b.maskedFill(p(a), p(mask), -1e9, n, p(out)), [{ pair: out, n }]);
  }
  {
    const n = 1234;
    const out = H.alloc(n * 4);
    await H.run("fill", "3.25", (b, p) => b.fill(p(out), n, 3.25), [{ pair: out, n }]);
  }

  // --------------------------------------------------------------- matmul
  const mmCases: { M: number; K: number; N: number; batch: number; tA: boolean; tB: boolean; bcastA?: boolean; bcastB?: boolean }[] = [
    { M: 1, K: 1, N: 1, batch: 1, tA: false, tB: false },
    { M: 17, K: 33, N: 65, batch: 1, tA: false, tB: false },
    { M: 64, K: 64, N: 64, batch: 1, tA: false, tB: false },
    { M: 17, K: 33, N: 65, batch: 1, tA: true, tB: false },
    { M: 17, K: 33, N: 65, batch: 1, tA: false, tB: true },
    { M: 17, K: 33, N: 65, batch: 1, tA: true, tB: true },
    { M: 70, K: 300, N: 90, batch: 1, tA: false, tB: false },
    { M: 70, K: 300, N: 90, batch: 1, tA: false, tB: true },
    { M: 9, K: 20, N: 11, batch: 3, tA: false, tB: false },
    { M: 9, K: 20, N: 11, batch: 3, tA: true, tB: true },
    { M: 9, K: 20, N: 11, batch: 3, tA: false, tB: false, bcastB: true },
    { M: 9, K: 20, N: 11, batch: 3, tA: false, tB: false, bcastA: true },
    { M: 1, K: 128, N: 257, batch: 2, tA: false, tB: true },
  ];
  for (const c of mmCases) {
    const aMat = c.M * c.K;
    const bMat = c.K * c.N;
    const aBS = c.bcastA ? 0 : aMat;
    const bBS = c.bcastB ? 0 : bMat;
    const a = H.upload(H.f32(c.bcastA ? aMat : aMat * c.batch, -1, 1));
    const bb = H.upload(H.f32(c.bcastB ? bMat : bMat * c.batch, -1, 1));
    const n = c.batch * c.M * c.N;
    const out = H.alloc(n * 4);
    const name = `${c.M}x${c.K}x${c.N} b${c.batch}${c.tA ? " tA" : ""}${c.tB ? " tB" : ""}${c.bcastA ? " bcastA" : ""}${c.bcastB ? " bcastB" : ""}`;
    await H.run("matmul", name, (b, p) => b.matmul(p(a), p(bb), p(out), c.batch, c.M, c.K, c.N, c.tA, c.tB, aBS, bBS), [
      { pair: out, n, threshold: c.K > 256 ? 1e-3 : DEFAULT_THR },
    ]);
  }

  // --------------------------------------------------------------- reduce
  for (const op of REDUCE_OPS) {
    const shape = [4, 5, 6];
    const cases: number[][] = [[2], [0], [1], [0, 2], [1, 2], [0, 1], [0, 1, 2]];
    for (const axes of cases) {
      const n = prod(shape);
      const outN = prod(shape.filter((_, i) => !axes.includes(i)));
      const redN = n / outN;
      const a = H.upload(H.f32(n));
      const out = H.alloc(outN * 4);
      await H.run("reduce", `${op} [4,5,6] axes ${JSON.stringify(axes)}`, (b, p) => b.reduce(op, p(a), shape, axes, p(out)), [
        { pair: out, n: outN, threshold: DEFAULT_THR * Math.sqrt(redN) },
      ]);
    }
    {
      const shape = [1000, 3];
      const a = H.upload(H.f32(3000));
      const out = H.alloc(3 * 4);
      await H.run("reduce", `${op} [1000,3] axis 0`, (b, p) => b.reduce(op, p(a), shape, [0], p(out)), [{ pair: out, n: 3, threshold: DEFAULT_THR * Math.sqrt(1000) }]);
    }
    {
      const shape = [3, 70000];
      const a = H.upload(H.f32(210000));
      const out = H.alloc(3 * 4);
      await H.run("reduce", `${op} [3,70000] axis 1`, (b, p) => b.reduce(op, p(a), shape, [1], p(out)), [{ pair: out, n: 3, threshold: DEFAULT_THR * Math.sqrt(70000) }]);
    }
    {
      // leading-axis reduction over many rows (bias gradients): exercises the two-stage chunked path
      const shape = [4096, 37];
      const a = H.upload(H.f32(4096 * 37));
      const out = H.alloc(37 * 4);
      await H.run("reduce", `${op} [4096,37] axis 0 (chunked)`, (b, p) => b.reduce(op, p(a), shape, [0], p(out)), [{ pair: out, n: 37, threshold: DEFAULT_THR * Math.sqrt(4096) }]);
    }
    {
      // prime row count: no divisor in range, single-pass fallback
      const shape = [1021, 5];
      const a = H.upload(H.f32(1021 * 5));
      const out = H.alloc(5 * 4);
      await H.run("reduce", `${op} [1021,5] axis 0 (prime rows)`, (b, p) => b.reduce(op, p(a), shape, [0], p(out)), [{ pair: out, n: 5, threshold: DEFAULT_THR * Math.sqrt(1021) }]);
    }
    {
      // leading reduction with a non-contiguous kept pair of groups
      const shape = [600, 3, 2, 5];
      const n = prod(shape);
      const a = H.upload(H.f32(n));
      const out = H.alloc(6 * 4);
      await H.run("reduce", `${op} [600,3,2,5] axes [0,3]`, (b, p) => b.reduce(op, p(a), shape, [0, 3], p(out)), [{ pair: out, n: 6, threshold: DEFAULT_THR * Math.sqrt(3000) }]);
    }
  }
  {
    const shape = [2, 1, 3, 1];
    const a = H.upload(H.f32(6));
    const out = H.alloc(6 * 4);
    await H.run("reduce", "sum over size-1 axes (copy)", (b, p) => b.reduce("sum", p(a), shape, [1, 3], p(out)), [{ pair: out, n: 6 }]);
  }

  // --------------------------------------------------------------- argmax
  {
    const rows = 7;
    const cols = 300;
    const a = H.upload(H.f32(rows * cols));
    const out = H.alloc(rows * 4);
    await H.run("argmax", "7x300", (b, p) => b.argmax(p(a), rows, cols, p(out)), [{ pair: out, n: rows, dtype: "i32", threshold: 0 }]);
  }
  {
    const rows = 5;
    const cols = 40;
    const a = H.upload(H.intsF32(rows * cols, 0, 3)); // ties -> first index
    const out = H.alloc(rows * 4);
    await H.run("argmax", "ties first index", (b, p) => b.argmax(p(a), rows, cols, p(out)), [{ pair: out, n: rows, dtype: "i32", threshold: 0 }]);
  }

  // -------------------------------------------------------------- softmax
  for (const log of [false, true]) {
    for (const [rows, cols] of [[5, 1000], [3, 7], [2, 70000]] as const) {
      const a = H.upload(H.f32(rows * cols, -5, 5));
      const out = H.alloc(rows * cols * 4);
      const label = `${log ? "log" : "softmax"} ${rows}x${cols}`;
      await H.run("softmax", label, (b, p) => b.softmax(p(a), rows, cols, p(out), log), [{ pair: out, n: rows * cols }]);
      const y = H.upload(H.f32(rows * cols, -5, 5));
      const yOut = H.alloc(rows * cols * 4);
      const g = H.upload(H.f32(rows * cols));
      const gi = H.alloc(rows * cols * 4);
      await H.run(
        "softmaxBackward",
        label,
        (b, p) => {
          b.softmax(p(y), rows, cols, p(yOut), log);
          b.softmaxBackward(p(yOut), p(g), rows, cols, p(gi), log);
        },
        [{ pair: gi, n: rows * cols, threshold: DEFAULT_THR * Math.sqrt(cols) }],
      );
    }
  }

  // ------------------------------------------------------------ layerNorm
  for (const [rows, cols] of [[6, 300], [33, 17], [2, 1000]] as const) {
    const x = H.upload(H.f32(rows * cols));
    const gamma = H.upload(H.f32(cols, 0.5, 1.5));
    const beta = H.upload(H.f32(cols));
    const out = H.alloc(rows * cols * 4);
    const mr = H.alloc(rows * 2 * 4);
    await H.run("layerNorm", `${rows}x${cols}`, (b, p) => b.layerNorm(p(x), p(gamma), p(beta), rows, cols, 1e-5, p(out), p(mr)), [
      { pair: out, n: rows * cols },
      { pair: mr, n: rows * 2, label: "meanRstd" },
    ]);
    const x2 = H.upload(H.f32(rows * cols));
    const gamma2 = H.upload(H.f32(cols, 0.5, 1.5));
    const beta2 = H.upload(H.f32(cols));
    const out2 = H.alloc(rows * cols * 4);
    const mr2 = H.alloc(rows * 2 * 4);
    const g = H.upload(H.f32(rows * cols));
    const gx = H.alloc(rows * cols * 4);
    const gg = H.alloc(cols * 4);
    const gb = H.alloc(cols * 4);
    await H.run(
      "layerNormBackward",
      `${rows}x${cols}`,
      (b, p) => {
        b.layerNorm(p(x2), p(gamma2), p(beta2), rows, cols, 1e-5, p(out2), p(mr2));
        b.layerNormBackward(p(x2), p(gamma2), p(mr2), p(g), rows, cols, p(gx), p(gg), p(gb));
      },
      [
        { pair: gx, n: rows * cols, label: "gradX", threshold: DEFAULT_THR * Math.sqrt(cols) },
        { pair: gg, n: cols, label: "gradGamma", threshold: DEFAULT_THR * Math.sqrt(rows) },
        { pair: gb, n: cols, label: "gradBeta", threshold: DEFAULT_THR * Math.sqrt(rows) },
      ],
    );
  }

  // ------------------------------------------------------- gather/scatter
  {
    const R = 50;
    const dim = 16;
    const n = 37;
    const table = H.upload(H.f32(R * dim));
    const ids = H.upload(H.ints(n, 0, R));
    const out = H.alloc(n * dim * 4);
    await H.run("gather", "50x16, 37 ids", (b, p) => b.gather(p(table), p(ids), n, dim, p(out)), [{ pair: out, n: n * dim }]);
  }
  for (const [R, dim, n] of [[20, 8, 100], [65, 24, 1500], [7, 3, 5]] as const) {
    const gradOut = H.upload(H.f32(n * dim));
    const ids = H.upload(H.ints(n, 0, R));
    const init = H.f32(R * dim);
    const table = H.upload(init);
    await H.run("scatterAdd", `table ${R}x${dim}, ${n} ids (dups, accumulate)`, (b, p) => b.scatterAdd(p(gradOut), p(ids), n, dim, p(table)), [
      { pair: table, n: R * dim, threshold: DEFAULT_THR * Math.sqrt(n / R + 1) },
    ]);
  }

  // --------------------------------------------------------- crossEntropy
  for (const withGrad of [false, true]) {
    const rows = 9;
    const cols = 50;
    const logits = H.upload(H.f32(rows * cols, -4, 4));
    const t = H.ints(rows, 0, cols);
    t[2] = -1;
    t[5] = -1;
    const targets = H.upload(t);
    const loss = H.alloc(rows * 4);
    const grad = H.alloc(rows * cols * 4);
    const outputs = [{ pair: loss, n: rows, label: "loss" }];
    if (withGrad) outputs.push({ pair: grad, n: rows * cols, label: "grad" });
    await H.run(
      "crossEntropy",
      `${rows}x${cols} ignoreIndex=-1 validCount=7${withGrad ? " +grad" : ""}`,
      (b, p) => b.crossEntropy(p(logits), p(targets), rows, cols, p(loss), withGrad ? p(grad) : undefined, -1, 7),
      outputs,
    );
  }
  {
    const rows = 4;
    const cols = 70000;
    const logits = H.upload(H.f32(rows * cols, -4, 4));
    const targets = H.upload(H.ints(rows, 0, cols));
    const loss = H.alloc(rows * 4);
    const grad = H.alloc(rows * cols * 4);
    await H.run("crossEntropy", `${rows}x${cols} wide rows`, (b, p) => b.crossEntropy(p(logits), p(targets), rows, cols, p(loss), p(grad)), [
      { pair: loss, n: rows, label: "loss", threshold: 1e-3 },
      { pair: grad, n: rows * cols, label: "grad" },
    ]);
  }

  // ---------------------------------------------------------------- adamw
  {
    const n = 777;
    const param = H.upload(H.f32(n));
    const grad = H.upload(H.f32(n, -0.1, 0.1));
    const m = H.upload(H.f32(n, -0.01, 0.01));
    const v = H.upload(H.f32(n, 0, 0.001));
    await H.run("adamw", "n=777 step=3", (b, p) => b.adamw(p(param), p(grad), p(m), p(v), n, 3e-4, 0.9, 0.95, 1e-8, 0.1, 3), [
      { pair: param, n, label: "param" },
      { pair: m, n, label: "m" },
      { pair: v, n, label: "v" },
    ]);
  }

  // ----------------------------------------------------------- sumSquares
  for (const n of [1, 1000, 100003]) {
    const a = H.upload(H.f32(n));
    const out = H.alloc(4);
    await H.run("sumSquares", `n=${n}`, (b, p) => b.sumSquares(p(a), n, p(out)), [{ pair: out, n: 1, threshold: 1e-4 * Math.sqrt(n) * 3 }]);
  }

  // ------------------------------------------------------------------ rng
  {
    const n = 1000;
    const seed = 1234;
    for (const kind of ["rand", "randn", "dropoutMask"] as const) {
      const out = H.alloc(n * 4);
      await H.run(
        "rng",
        `${kind} seed=${seed}`,
        (b, p) => {
          if (kind === "rand") b.rand(p(out), n, seed);
          else if (kind === "randn") b.randn(p(out), n, seed);
          else b.dropoutMask(p(out), n, 0.3, seed);
        },
        [{ pair: out, n, threshold: kind === "randn" ? 1e-4 : 0 }],
      );
    }
  }

  // ----------------------------------------------------------------- copy
  {
    const n = 300;
    const a = H.upload(H.f32(n));
    const out = H.alloc(n * 4);
    await H.run("copy", "offset copy", (b, p) => b.copy(p(a), p(out), 100 * 4, 50 * 4, 20 * 4), [{ pair: out, n }]);
  }

  return H.results;
}

/** Outcome published for the Playwright script. */
export interface ParityPublished {
  results: ParityResult[];
  pass: boolean;
  /** True when no WebGPU backend could be created (the script then exits 0). */
  unavailable: boolean;
  info?: { vendor: string; architecture: string; description: string };
  failures: number;
}

/**
 * Run the suite against a fresh (or given) WebGPU backend and publish the outcome on
 * `globalThis.__kilnParityDone`. Never throws.
 */
export async function runParityAndPublish(gpu?: Backend, opts: ParityOptions = {}): Promise<ParityPublished> {
  let published: ParityPublished;
  try {
    let backend = gpu;
    let info: ParityPublished["info"];
    if (!backend) {
      if (!isWebGpuAvailable()) throw new Error("navigator.gpu missing");
      const b = await createWebGpuBackend();
      info = { vendor: b.info.vendor, architecture: b.info.architecture, description: b.info.description };
      backend = b;
    }
    const results = await runParity(backend, opts);
    const failures = results.filter((r) => !r.pass).length;
    published = { results, pass: failures === 0, unavailable: false, failures };
    if (info) published.info = info;
  } catch (err: unknown) {
    published = {
      results: [{ op: "setup", case: err instanceof Error ? err.message : String(err), maxAbs: NaN, maxRel: NaN, threshold: 0, pass: false, ms: 0 }],
      pass: false,
      unavailable: true,
      failures: 1,
    };
  }
  (globalThis as { __kilnParityDone?: ParityPublished }).__kilnParityDone = published;
  return published;
}
