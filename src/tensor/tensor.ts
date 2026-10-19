/**
 * The Tensor class: a device-agnostic strided view over a refcounted Storage, plus autograd metadata.
 * Op methods delegate to `src/autograd/ops.ts` (the import cycle is resolved at call time).
 */
import type { Backend, Buf, Device, DType } from "../backend/types";
import { allocTypedArray, dtypeOfArray, dtypeSize, makeTypedArray, type TypedArray } from "./dtype";
import { getBackend, getDefaultDevice } from "./device";
import { nextSeed } from "./random";
import {
  contiguousStrides, flattenNested, inferNestedShape, isContiguous, offsetOfFlat, sizeOf, type Nested, type Shape,
} from "./shape";
import { Storage } from "./storage";
import { trackTensor } from "./scope";
import * as ops from "../autograd/ops";
import { backward as runBackward } from "../autograd/graph";

/** Options accepted by every creation function. */
export interface TensorOptions {
  device?: Device;
  dtype?: DType;
  requiresGrad?: boolean;
}

/** A recorded autograd function (implemented by `autograd/graph.ts`'s `Node`). */
export interface GradFn {
  readonly name: string;
  readonly inputs: readonly Tensor[];
  /** Fresh tensors owned by the node for backward (freed by `release()`). */
  readonly saved: readonly Tensor[];
  /** True once `release()` ran (the graph behind this node is consumed). */
  readonly released: boolean;
  backward(gradOut: Tensor): (Tensor | null)[];
  release(): void;
}

/** Number of edge items shown per dimension when printing large tensors. */
const PRINT_EDGE = 3;
const PRINT_THRESHOLD = 1000;

/** N-dimensional strided tensor. */
export class Tensor {
  readonly shape: readonly number[];
  readonly strides: readonly number[];
  readonly offset: number;
  readonly dtype: DType;
  readonly device: Device;
  storage: Storage;
  requiresGrad: boolean;
  grad: Tensor | null = null;
  gradFn: GradFn | null = null;
  /** When true, `backward()` keeps this non-leaf tensor's grad. */
  retainsGrad = false;
  private freed = false;

  constructor(storage: Storage, shape: Shape, strides: Shape, offset: number, requiresGrad = false) {
    this.storage = storage;
    this.shape = shape.slice();
    this.strides = strides.slice();
    this.offset = offset;
    this.dtype = storage.dtype;
    this.device = storage.device;
    this.requiresGrad = requiresGrad;
    trackTensor(this);
  }

  // ---------------------------------------------------------------- creation

  /** Uninitialised contiguous tensor. */
  static empty(shape: Shape, opts: TensorOptions = {}): Tensor {
    const device = opts.device ?? getDefaultDevice();
    const dtype = opts.dtype ?? "f32";
    const backend = getBackend(device);
    const st = Storage.alloc(backend, dtype, sizeOf(shape));
    return new Tensor(st, shape, contiguousStrides(shape), 0, opts.requiresGrad ?? false);
  }

  /** Tensor filled with `value`. */
  static full(shape: Shape, value: number, opts: TensorOptions = {}): Tensor {
    const t = Tensor.empty(shape, opts);
    if (t.dtype === "f32") {
      t.backend.fill(t.buf, t.numel, value);
    } else {
      const arr = allocTypedArray(t.dtype, t.numel);
      arr.fill(value);
      t.backend.write(t.buf, arr);
    }
    return t;
  }

  static zeros(shape: Shape, opts: TensorOptions = {}): Tensor {
    return Tensor.full(shape, 0, opts);
  }

  static ones(shape: Shape, opts: TensorOptions = {}): Tensor {
    return Tensor.full(shape, 1, opts);
  }

  /** Standard-normal tensor from the counter-based hash RNG (seed drawn from the global generator). */
  static randn(shape: Shape, opts: TensorOptions & { seed?: number } = {}): Tensor {
    const t = Tensor.empty(shape, { ...opts, dtype: "f32" });
    t.backend.randn(t.buf, t.numel, opts.seed ?? nextSeed());
    return t;
  }

  /** Uniform [0,1) tensor from the counter-based hash RNG. */
  static rand(shape: Shape, opts: TensorOptions & { seed?: number } = {}): Tensor {
    const t = Tensor.empty(shape, { ...opts, dtype: "f32" });
    t.backend.rand(t.buf, t.numel, opts.seed ?? nextSeed());
    return t;
  }

  /** `arange(end)` or `arange(start, end, step)`. */
  static arange(start: number, end?: number, step = 1, opts: TensorOptions = {}): Tensor {
    let s = start;
    let e = end;
    if (e === undefined) {
      e = s;
      s = 0;
    }
    if (step === 0) throw new Error("arange step must be non-zero");
    const n = Math.max(0, Math.ceil((e - s) / step));
    const dtype = opts.dtype ?? "f32";
    const arr = allocTypedArray(dtype, n);
    for (let i = 0; i < n; i++) arr[i] = s + i * step;
    return Tensor.fromArray(arr, [n], { ...opts, dtype });
  }

  /** `steps` evenly spaced values from start to end inclusive. */
  static linspace(start: number, end: number, steps: number, opts: TensorOptions = {}): Tensor {
    const arr = new Float32Array(steps);
    const d = steps > 1 ? (end - start) / (steps - 1) : 0;
    for (let i = 0; i < steps; i++) arr[i] = start + i * d;
    return Tensor.fromArray(arr, [steps], { ...opts, dtype: "f32" });
  }

  /** Identity matrix nxm. */
  static eye(n: number, m?: number, opts: TensorOptions = {}): Tensor {
    const cols = m ?? n;
    const dtype = opts.dtype ?? "f32";
    const arr = allocTypedArray(dtype, n * cols);
    for (let i = 0; i < Math.min(n, cols); i++) arr[i * cols + i] = 1;
    return Tensor.fromArray(arr, [n, cols], { ...opts, dtype });
  }

  /** Tensor from a flat array (typed arrays set the default dtype). */
  static fromArray(data: ArrayLike<number>, shape?: Shape, opts: TensorOptions = {}): Tensor {
    const dtype = opts.dtype ?? dtypeOfArray(data);
    const shp = shape ?? [data.length];
    const n = sizeOf(shp);
    if (n !== data.length) throw new Error(`fromArray: ${data.length} values do not fit shape [${shp.join(",")}]`);
    const t = Tensor.empty(shp, { ...opts, dtype });
    let src: TypedArray;
    if (
      (dtype === "f32" && data instanceof Float32Array) ||
      (dtype === "i32" && data instanceof Int32Array) ||
      (dtype === "u32" && data instanceof Uint32Array)
    ) {
      src = data;
    } else {
      src = allocTypedArray(dtype, n);
      for (let i = 0; i < n; i++) src[i] = data[i] as number;
    }
    if (n > 0) t.backend.write(t.buf, src);
    return t;
  }

  /** Tensor from nested JS arrays (shape inferred). */
  static tensor(data: Nested, opts: TensorOptions = {}): Tensor {
    const shape = inferNestedShape(data);
    const dtype = opts.dtype ?? "f32";
    const arr = allocTypedArray(dtype, sizeOf(shape));
    flattenNested(data, shape, { set: (i, v) => { arr[i] = v; } });
    return Tensor.fromArray(arr, shape, { ...opts, dtype });
  }

  /** Scalar (0-d) tensor. */
  static scalar(value: number, opts: TensorOptions = {}): Tensor {
    return Tensor.full([], value, opts);
  }

  // ---------------------------------------------------------------- metadata

  get numel(): number {
    return sizeOf(this.shape);
  }

  get ndim(): number {
    return this.shape.length;
  }

  /** True when strides are dense row-major (offset may be non-zero). */
  get isContiguous(): boolean {
    return isContiguous(this.shape, this.strides);
  }

  /** True when the tensor is contiguous *and* starts at element 0 of its buffer - i.e. kernels can read it directly. */
  get isDense(): boolean {
    return this.offset === 0 && this.isContiguous;
  }

  get backend(): Backend {
    return this.storage.backend;
  }

  /** The underlying backend buffer. */
  get buf(): Buf {
    return this.storage.buf;
  }

  /** True for a leaf of the autograd graph. */
  get isLeaf(): boolean {
    return this.gradFn === null;
  }

  /** Size of dimension `dim` (negative allowed). */
  size(dim: number): number {
    const d = dim < 0 ? dim + this.ndim : dim;
    const v = this.shape[d];
    if (v === undefined) throw new Error(`Dimension ${dim} out of range`);
    return v;
  }

  // ---------------------------------------------------------------- memory / views

  /** New tensor handle sharing this storage (refcount +1). */
  viewAs(shape: Shape, strides: Shape, offset: number): Tensor {
    this.storage.retain();
    return new Tensor(this.storage, shape, strides, offset, false);
  }

  /** Release this tensor's reference to its storage (idempotent per handle). */
  free(): void {
    if (this.freed) return;
    this.freed = true;
    this.storage.release();
  }

  /** True once this handle released its storage reference. */
  get isFreed(): boolean {
    return this.freed;
  }

  /** Same data, no autograd history. */
  detach(): Tensor {
    return this.viewAs(this.shape, this.strides, this.offset);
  }

  /** Drop the accumulated gradient (its buffer is freed - the tensor owns its grad). */
  zeroGrad(): void {
    if (this.grad) this.grad.free();
    this.grad = null;
  }

  /** Ask `backward()` to keep this non-leaf tensor's gradient. */
  retainGrad(): this {
    this.retainsGrad = true;
    return this;
  }

  // ---------------------------------------------------------------- data movement

  /** Move to another device (CPU -> GPU is synchronous; GPU -> CPU needs `await t.cpu()`). */
  to(device: Device): Tensor {
    if (device === this.device) return this;
    if (this.device !== "cpu") throw new Error("Use `await t.cpu()` to move data off the GPU");
    const be = getBackend(device);
    const data = this.dataSync();
    const st = Storage.alloc(be, this.dtype, this.numel);
    if (this.numel > 0) be.write(st.buf, data);
    return new Tensor(st, this.shape, contiguousStrides(this.shape), 0, this.requiresGrad);
  }

  /** Copy to the CPU (async; forces pending GPU work). */
  async cpu(): Promise<Tensor> {
    if (this.device === "cpu") return this;
    const data = await this.data();
    return Tensor.fromArray(data, this.shape, { device: "cpu", dtype: this.dtype, requiresGrad: this.requiresGrad });
  }

  /** Contiguous copy of the data as a typed array (async; works on every device). */
  async data(): Promise<TypedArray> {
    if (this.device === "cpu") return this.dataSync().slice() as TypedArray;
    const d = this.isDense ? this : ops.contiguousRaw(this);
    const ab = await d.backend.read(d.buf, d.numel * dtypeSize(d.dtype));
    return makeTypedArray(this.dtype, ab, 0, this.numel);
  }

  /**
   * CPU only. Returns a typed array over the tensor's elements. Contiguous tensors return a live view
   * into storage (mutations write through); non-contiguous ones return a fresh contiguous copy.
   */
  dataSync(): TypedArray {
    if (this.device !== "cpu") throw new Error("dataSync() is only available on CPU tensors");
    if (this.isContiguous) {
      return makeTypedArray(this.dtype, this.buf.handle as ArrayBuffer, this.offset * dtypeSize(this.dtype), this.numel);
    }
    const d = ops.contiguousRaw(this);
    return makeTypedArray(this.dtype, d.buf.handle as ArrayBuffer, 0, this.numel);
  }

  /** Value of a one-element tensor. */
  async item(): Promise<number> {
    if (this.numel !== 1) throw new Error(`item() requires a one-element tensor, got shape [${this.shape.join(",")}]`);
    const d = await this.data();
    return d[0] as number;
  }

  /** CPU-only synchronous `item()`. */
  itemSync(): number {
    if (this.numel !== 1) throw new Error(`item() requires a one-element tensor, got shape [${this.shape.join(",")}]`);
    return this.dataSync()[0] as number;
  }

  /** CPU-only: nested JS arrays of the values. */
  toArray(): Nested {
    const flat = this.dataSync();
    const build = (dim: number, base: number): Nested => {
      if (dim === this.ndim) return flat[base] as number;
      const n = this.shape[dim] as number;
      const stride = sizeOf(this.shape.slice(dim + 1));
      const out: Nested[] = new Array(n);
      for (let i = 0; i < n; i++) out[i] = build(dim + 1, base + i * stride);
      return out;
    };
    return build(0, 0);
  }

  // ---------------------------------------------------------------- in-place (no autograd)

  /** In-place copy of `src` (same shape, same device) into this tensor's storage. */
  copy_(src: Tensor): this {
    if (src.numel !== this.numel) throw new Error("copy_: numel mismatch");
    if (src.device !== this.device) throw new Error("copy_: device mismatch");
    const s = src.isDense ? src : ops.contiguousRaw(src);
    if (this.isDense) {
      this.backend.copy(s.buf, this.buf, this.numel * dtypeSize(this.dtype));
    } else if (this.isContiguous) {
      this.backend.copy(s.buf, this.buf, this.numel * dtypeSize(this.dtype), 0, this.offset * dtypeSize(this.dtype));
    } else {
      throw new Error("copy_ into a non-contiguous tensor is not supported");
    }
    return this;
  }

  /** In-place fill (contiguous tensors only). */
  fill_(value: number): this {
    if (!this.isDense) throw new Error("fill_ requires a dense tensor");
    this.backend.fill(this.buf, this.numel, value);
    return this;
  }

  /** In-place zero. */
  zero_(): this {
    return this.fill_(0);
  }

  // ---------------------------------------------------------------- autograd

  /** Reverse-mode backward from this tensor (scalar unless `grad` is given). */
  backward(grad?: Tensor): void {
    runBackward(this, grad);
  }

  // ---------------------------------------------------------------- op sugar (see autograd/ops.ts)

  add(other: Tensor | number): Tensor { return ops.add(this, other); }
  sub(other: Tensor | number): Tensor { return ops.sub(this, other); }
  mul(other: Tensor | number): Tensor { return ops.mul(this, other); }
  div(other: Tensor | number): Tensor { return ops.div(this, other); }
  pow(other: Tensor | number): Tensor { return ops.pow(this, other); }
  maximum(other: Tensor | number): Tensor { return ops.maximum(this, other); }
  minimum(other: Tensor | number): Tensor { return ops.minimum(this, other); }
  eq(other: Tensor | number): Tensor { return ops.eq(this, other); }
  lt(other: Tensor | number): Tensor { return ops.lt(this, other); }
  gt(other: Tensor | number): Tensor { return ops.gt(this, other); }
  le(other: Tensor | number): Tensor { return ops.le(this, other); }
  ge(other: Tensor | number): Tensor { return ops.ge(this, other); }
  neg(): Tensor { return ops.neg(this); }
  exp(): Tensor { return ops.exp(this); }
  log(): Tensor { return ops.log(this); }
  sqrt(): Tensor { return ops.sqrt(this); }
  rsqrt(): Tensor { return ops.rsqrt(this); }
  tanh(): Tensor { return ops.tanh(this); }
  sigmoid(): Tensor { return ops.sigmoid(this); }
  relu(): Tensor { return ops.relu(this); }
  gelu(): Tensor { return ops.gelu(this); }
  square(): Tensor { return ops.square(this); }
  abs(): Tensor { return ops.abs(this); }
  reciprocal(): Tensor { return ops.reciprocal(this); }
  sin(): Tensor { return ops.sin(this); }
  cos(): Tensor { return ops.cos(this); }
  clamp(min?: number, max?: number): Tensor { return ops.clamp(this, min, max); }
  softmax(dim = -1): Tensor { return ops.softmax(this, dim); }
  logSoftmax(dim = -1): Tensor { return ops.logSoftmax(this, dim); }
  sum(axes?: number | readonly number[], keepDim = false): Tensor { return ops.sum(this, axes, keepDim); }
  mean(axes?: number | readonly number[], keepDim = false): Tensor { return ops.mean(this, axes, keepDim); }
  max(axes?: number | readonly number[], keepDim = false): Tensor { return ops.max(this, axes, keepDim); }
  min(axes?: number | readonly number[], keepDim = false): Tensor { return ops.min(this, axes, keepDim); }
  var(axes?: number | readonly number[], opts?: { unbiased?: boolean; keepDim?: boolean }): Tensor { return ops.variance(this, axes, opts); }
  std(axes?: number | readonly number[], opts?: { unbiased?: boolean; keepDim?: boolean }): Tensor { return ops.std(this, axes, opts); }
  argmax(dim?: number, keepDim = false): Tensor { return ops.argmax(this, dim, keepDim); }
  matmul(other: Tensor): Tensor { return ops.matmul(this, other); }
  transpose(dim0 = -2, dim1 = -1): Tensor { return ops.transpose(this, dim0, dim1); }
  /** Transpose of a 2-D tensor. */
  get T(): Tensor { return ops.transpose(this, 0, 1); }
  permute(...perm: number[]): Tensor { return ops.permute(this, perm); }
  reshape(...shape: number[]): Tensor { return ops.reshape(this, shape); }
  view(...shape: number[]): Tensor { return ops.view(this, shape); }
  flatten(start = 0, end = -1): Tensor { return ops.flatten(this, start, end); }
  squeeze(dim?: number): Tensor { return ops.squeeze(this, dim); }
  unsqueeze(dim: number): Tensor { return ops.unsqueeze(this, dim); }
  slice(dim: number, start?: number, end?: number, step = 1): Tensor { return ops.slice(this, dim, start, end, step); }
  narrow(dim: number, start: number, length: number): Tensor { return ops.narrow(this, dim, start, length); }
  expand(...shape: number[]): Tensor { return ops.expand(this, shape); }
  broadcastTo(...shape: number[]): Tensor { return ops.broadcastTo(this, shape); }
  gather(dim: number, ids: Tensor): Tensor { return ops.gather(this, dim, ids); }
  indexSelect(dim: number, ids: Tensor): Tensor { return ops.indexSelect(this, dim, ids); }
  /** Treat this [V, C] tensor as an embedding table and look up `ids`. */
  embedding(ids: Tensor): Tensor { return ops.embedding(this, ids); }
  scatterAdd(ids: Tensor, src: Tensor): Tensor { return ops.scatterAdd(this, ids, src); }
  /** where(cond, this, other). */
  where(cond: Tensor, other: Tensor | number): Tensor { return ops.where(cond, this, other); }
  maskedFill(mask: Tensor, value: number): Tensor { return ops.maskedFill(this, mask, value); }
  tril(k = 0): Tensor { return ops.tril(this, k); }
  triu(k = 0): Tensor { return ops.triu(this, k); }
  contiguous(): Tensor { return ops.contiguous(this); }
  clone(): Tensor { return ops.clone(this); }
  crossEntropy(targets: Tensor, opts?: ops.CrossEntropyOptions): Tensor { return ops.crossEntropy(this, targets, opts); }
  layerNorm(gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor { return ops.layerNorm(this, gamma, beta, eps); }
  dropout(p: number, training = true, seed?: number): Tensor { return ops.dropout(this, p, training, seed); }

  /** CPU-only dtype conversion (copies). */
  toDtype(dtype: DType): Tensor {
    if (dtype === this.dtype) return this;
    if (this.device !== "cpu") throw new Error("toDtype is only available on CPU tensors");
    const src = this.dataSync();
    const dst = allocTypedArray(dtype, this.numel);
    for (let i = 0; i < this.numel; i++) dst[i] = dtype === "f32" ? (src[i] as number) : Math.trunc(src[i] as number);
    return Tensor.fromArray(dst, this.shape, { device: "cpu", dtype });
  }

  /** CPU-only cast to f32. */
  float(): Tensor { return this.toDtype("f32"); }
  /** CPU-only cast to i32. */
  int(): Tensor { return this.toDtype("i32"); }

  // ---------------------------------------------------------------- printing

  /** PyTorch-style pretty print (CPU tensors print values; GPU tensors print metadata). */
  toString(): string {
    const meta = `shape=[${this.shape.join(", ")}], dtype=${this.dtype}, device=${this.device}` +
      (this.requiresGrad ? ", requiresGrad=true" : "") + (this.gradFn ? `, gradFn=${this.gradFn.name}` : "");
    if (this.device !== "cpu") return `Tensor(<${this.device} data>, ${meta})`;
    const data = this.dataSync();
    const fmt = this.dtype === "f32" ? formatFloats(data, this.numel) : (v: number) => String(v);
    const n = this.numel;
    const summarize = n > PRINT_THRESHOLD;
    const render = (dim: number, base: number, indent: string): string => {
      if (dim === this.ndim) return fmt(data[base] as number);
      const size = this.shape[dim] as number;
      const stride = sizeOf(this.shape.slice(dim + 1));
      const items: string[] = [];
      const push = (i: number): void => { items.push(render(dim + 1, base + i * stride, indent + " ")); };
      if (summarize && size > 2 * PRINT_EDGE) {
        for (let i = 0; i < PRINT_EDGE; i++) push(i);
        items.push("...");
        for (let i = size - PRINT_EDGE; i < size; i++) push(i);
      } else {
        for (let i = 0; i < size; i++) push(i);
      }
      const sep = dim === this.ndim - 1 ? ", " : ",\n" + indent + " ";
      return "[" + items.join(sep) + "]";
    };
    const body = this.ndim === 0 ? fmt(data[0] as number) : render(0, 0, "       ");
    return `tensor(${body}, ${meta})`;
  }
}

function formatFloats(data: ArrayLike<number>, n: number): (v: number) => string {
  let allInt = true;
  let maxAbs = 0;
  for (let i = 0; i < n; i++) {
    const v = data[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v !== Math.trunc(v)) allInt = false;
    maxAbs = Math.max(maxAbs, Math.abs(v));
  }
  if (allInt) return (v) => (Number.isFinite(v) ? `${v}.` : String(v));
  if (maxAbs !== 0 && (maxAbs >= 1e5 || maxAbs < 1e-3)) return (v) => (Number.isFinite(v) ? v.toExponential(4) : String(v));
  return (v) => (Number.isFinite(v) ? v.toFixed(4) : String(v));
}

// -------------------------------------------------------------------- functional creation aliases

/** Uninitialised tensor. */
export const empty = Tensor.empty;
/** Zero-filled tensor. */
export const zeros = Tensor.zeros;
/** One-filled tensor. */
export const ones = Tensor.ones;
/** Constant-filled tensor. */
export const full = Tensor.full;
/** Standard-normal tensor. */
export const randn = Tensor.randn;
/** Uniform [0,1) tensor. */
export const rand = Tensor.rand;
/** Range tensor. */
export const arange = Tensor.arange;
/** Linearly spaced tensor. */
export const linspace = Tensor.linspace;
/** Identity matrix. */
export const eye = Tensor.eye;
/** Tensor from a flat array. */
export const fromArray = Tensor.fromArray;
/** Tensor from nested arrays. */
export const tensor = Tensor.tensor;
/** 0-d tensor. */
export const scalar = Tensor.scalar;

/** Element offset of the i-th row-major element of a view (exported for tests). */
export function elementOffset(t: Tensor, flat: number): number {
  return offsetOfFlat(flat, t.shape, t.strides, t.offset);
}
