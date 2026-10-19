/**
 * Differentiable tensor ops. Every forward and backward is expressed through the `Backend` interface
 * (never touching typed arrays), so the same code drives the CPU and WebGPU backends.
 *
 * Conventions:
 *  - `*Raw` helpers compute without recording history; public ops call `record()` to attach a Node.
 *  - Kernels need dense inputs (contiguous, offset 0); `dense()` materialises views on demand.
 *  - Backward closures return one gradient per input, each shaped exactly like that input.
 */
import type { BinaryOp, Buf, Device, UnaryOp } from "../backend/types";
import { dtypeSize } from "../tensor/dtype";
import { nextSeed } from "../tensor/random";
import {
  broadcastShapes, broadcastShapesN, broadcastStrides, checkPermutation, contiguousStrides, expandStrides,
  inferReshape, insertDim, invertPermutation, normalizeAxes, normalizeAxis, permuteShape, reduceShape, removeDim,
  shapesEqual, sizeOf, sliceStridesOffset, type Shape,
} from "../tensor/shape";
import { Tensor } from "../tensor/tensor";
import { escape } from "../tensor/scope";
import { isGradEnabled, record } from "./graph";

// ============================================================================ helpers

type Scalar = number | Tensor;

function emptyLike(t: Tensor, shape: Shape, dtype = t.dtype): Tensor {
  return Tensor.empty(shape, { device: t.device, dtype });
}

function elemBytes(t: Tensor): number {
  return dtypeSize(t.dtype);
}

/** Contiguous, offset-0 copy of `t` (or `t` itself when it already is). No autograd. */
export function contiguousRaw(t: Tensor): Tensor {
  if (t.isDense) return t;
  const out = emptyLike(t, t.shape);
  if (t.numel > 0) t.backend.copyStrided(t.buf, t.shape.slice(), t.strides.slice(), t.offset, out.buf);
  return out;
}

/** Always-fresh contiguous copy. No autograd. */
export function cloneRaw(t: Tensor): Tensor {
  if (!t.isDense) return contiguousRaw(t);
  const out = emptyLike(t, t.shape);
  if (t.numel > 0) t.backend.copy(t.buf, out.buf, t.numel * elemBytes(t));
  return out;
}

const dense = contiguousRaw;

/** The tensors among `derived` that are fresh copies (not one of `originals`) - owned by a backward node. */
function fresh(derived: readonly Tensor[], originals: readonly Tensor[]): Tensor[] {
  const out: Tensor[] = [];
  for (const d of derived) if (!originals.includes(d) && !out.includes(d)) out.push(d);
  return out;
}

function assertDevice(a: Tensor, b: Tensor, op: string): void {
  if (a.device !== b.device) throw new Error(`${op}: tensors are on different devices (${a.device} vs ${b.device})`);
}

function assertFloat(t: Tensor, op: string): void {
  if (t.dtype !== "f32") throw new Error(`${op}: requires an f32 tensor, got ${t.dtype}`);
}

function scalarTensor(v: number, device: Device): Tensor {
  return Tensor.full([], v, { device, dtype: "f32" });
}

function asTensor(x: Scalar, like: Tensor): Tensor {
  return typeof x === "number" ? scalarTensor(x, like.device) : x;
}

/** Lift a (Tensor|number, Tensor|number) pair to two tensors on the same device. */
function pair(a: Scalar, b: Scalar, op: string): [Tensor, Tensor] {
  if (typeof a === "number") {
    if (typeof b === "number") throw new Error(`${op}: at least one argument must be a Tensor`);
    return [scalarTensor(a, b.device), b];
  }
  return [a, asTensor(b, a)];
}

/** View helper (no autograd). */
function viewRaw(t: Tensor, shape: Shape, strides: Shape, offset: number): Tensor {
  return t.viewAs(shape, strides, offset);
}

function unaryRaw(op: UnaryOp, x: Tensor, alpha?: number, beta?: number): Tensor {
  const xd = dense(x);
  const out = emptyLike(x, x.shape, "f32");
  x.backend.unary(op, xd.buf, x.numel, out.buf, alpha, beta);
  return out;
}

function binaryRaw(op: BinaryOp, a: Tensor, b: Tensor): Tensor {
  assertDevice(a, b, op);
  const outShape = broadcastShapes(a.shape, b.shape);
  const ad = a.offset === 0 ? a : dense(a);
  const bd = b.offset === 0 ? b : dense(b);
  const out = emptyLike(a, outShape, "f32");
  a.backend.binary(
    op, ad.buf, bd.buf, out.buf, outShape,
    broadcastStrides(ad.shape, ad.strides, outShape), broadcastStrides(bd.shape, bd.strides, outShape),
  );
  return out;
}

function reduceRaw(op: "sum" | "max" | "min", x: Tensor, axes: number[]): Tensor {
  const xd = dense(x);
  const out = emptyLike(x, reduceShape(x.shape, axes, false), "f32");
  x.backend.reduce(op, xd.buf, x.shape.slice(), axes, out.buf);
  return out;
}

/** Sum `g` down to `shape` (inverse of broadcasting). No autograd. */
function unbroadcast(g: Tensor, shape: Shape): Tensor {
  if (shapesEqual(g.shape, shape)) return g;
  const lead = g.ndim - shape.length;
  const axes: number[] = [];
  for (let i = 0; i < lead; i++) axes.push(i);
  for (let i = 0; i < shape.length; i++) {
    if (shape[i] === 1 && g.shape[lead + i] !== 1) axes.push(lead + i);
  }
  const r = reduceRaw("sum", g, axes);
  return shapesEqual(r.shape, shape) ? r : viewRaw(r, shape, contiguousStrides(shape), 0);
}

/** Broadcast `g` (shaped like `keepShape`, i.e. reduced dims kept as 1) back to `shape` as a dense tensor. */
function expandBack(g: Tensor, keepShape: Shape, shape: Shape): Tensor {
  const gk = viewRaw(dense(g), keepShape, contiguousStrides(keepShape), 0);
  const e = expandStrides(gk.shape, gk.strides, shape);
  return contiguousRaw(viewRaw(gk, e.shape, e.strides, 0));
}

// ============================================================================ arithmetic

/** a + b (broadcasting; either side may be a number). */
export function add(a: Scalar, b: Scalar): Tensor {
  if (typeof a === "number") return add(b, a);
  if (typeof b === "number") {
    const out = unaryRaw("addScalar", a, b);
    return record(out, "add", [a], (g) => [g]);
  }
  const out = binaryRaw("add", a, b);
  return record(out, "add", [a, b], (g) => [unbroadcast(g, a.shape), unbroadcast(g, b.shape)]);
}

/** a − b. */
export function sub(a: Scalar, b: Scalar): Tensor {
  if (typeof a === "number") {
    if (typeof b === "number") throw new Error("sub: at least one argument must be a Tensor");
    const out = unaryRaw("addScalar", unaryRaw("scale", b, -1), a);
    return record(out, "rsub", [b], (g) => [unaryRaw("scale", g, -1)]);
  }
  if (typeof b === "number") {
    const out = unaryRaw("addScalar", a, -b);
    return record(out, "sub", [a], (g) => [g]);
  }
  const out = binaryRaw("sub", a, b);
  return record(out, "sub", [a, b], (g) => [unbroadcast(g, a.shape), unbroadcast(unaryRaw("scale", g, -1), b.shape)]);
}

/** a · b (elementwise). */
export function mul(a: Scalar, b: Scalar): Tensor {
  if (typeof a === "number") return mul(b, a);
  if (typeof b === "number") {
    const out = unaryRaw("scale", a, b);
    return record(out, "mul", [a], (g) => [unaryRaw("scale", g, b)]);
  }
  const out = binaryRaw("mul", a, b);
  return record(out, "mul", [a, b], (g) => [
    unbroadcast(binaryRaw("mul", g, b), a.shape),
    unbroadcast(binaryRaw("mul", g, a), b.shape),
  ]);
}

/** a / b (elementwise). */
export function div(a: Scalar, b: Scalar): Tensor {
  if (typeof a === "number") {
    if (typeof b === "number") throw new Error("div: at least one argument must be a Tensor");
    const out = unaryRaw("scale", unaryRaw("recip", b), a);
    return record(out, "rdiv", [b], (g) => [unaryRaw("scale", binaryRaw("div", g, binaryRaw("mul", b, b)), -a)]);
  }
  if (typeof b === "number") {
    const out = unaryRaw("scale", a, 1 / b);
    return record(out, "div", [a], (g) => [unaryRaw("scale", g, 1 / b)]);
  }
  const out = binaryRaw("div", a, b);
  return record(out, "div", [a, b], (g) => [
    unbroadcast(binaryRaw("div", g, b), a.shape),
    unbroadcast(unaryRaw("scale", binaryRaw("mul", g, binaryRaw("div", out, b)), -1), b.shape),
  ]);
}

/** a ^ b (elementwise). */
export function pow(a: Scalar, b: Scalar): Tensor {
  if (typeof a === "number") {
    if (typeof b === "number") throw new Error("pow: at least one argument must be a Tensor");
    const ln = Math.log(a);
    const out = unaryRaw("exp", unaryRaw("scale", b, ln));
    return record(out, "rpow", [b], (g) => [unaryRaw("scale", binaryRaw("mul", g, out), ln)]);
  }
  if (typeof b === "number") {
    const out = unaryRaw("powScalar", a, b);
    return record(out, "pow", [a], (g) => [
      b === 0 ? unaryRaw("scale", g, 0) : binaryRaw("mul", g, unaryRaw("scale", unaryRaw("powScalar", a, b - 1), b)),
    ]);
  }
  const out = binaryRaw("pow", a, b);
  return record(out, "pow", [a, b], (g) => [
    unbroadcast(binaryRaw("mul", g, binaryRaw("mul", b, binaryRaw("pow", a, sub(b, 1)))), a.shape),
    unbroadcast(binaryRaw("mul", g, binaryRaw("mul", out, unaryRaw("log", a))), b.shape),
  ]);
}

/** Elementwise maximum (ties route the gradient to `a`). */
export function maximum(a: Scalar, b: Scalar): Tensor {
  const [at, bt] = pair(a, b, "maximum");
  const out = binaryRaw("max", at, bt);
  return record(out, "maximum", [at, bt], (g) => {
    const m = binaryRaw("ge", at, bt);
    return [unbroadcast(binaryRaw("mul", g, m), at.shape), unbroadcast(binaryRaw("mul", g, binaryRaw("lt", at, bt)), bt.shape)];
  });
}

/** Elementwise minimum (ties route the gradient to `a`). */
export function minimum(a: Scalar, b: Scalar): Tensor {
  const [at, bt] = pair(a, b, "minimum");
  const out = binaryRaw("min", at, bt);
  return record(out, "minimum", [at, bt], (g) => {
    const m = binaryRaw("le", at, bt);
    return [unbroadcast(binaryRaw("mul", g, m), at.shape), unbroadcast(binaryRaw("mul", g, binaryRaw("gt", at, bt)), bt.shape)];
  });
}

function compare(op: BinaryOp, a: Scalar, b: Scalar): Tensor {
  const [at, bt] = pair(a, b, op);
  return binaryRaw(op, at, bt);
}

/** a == b -> 1.0/0.0 (no gradient). */
export function eq(a: Scalar, b: Scalar): Tensor { return compare("eq", a, b); }
/** a < b -> 1.0/0.0. */
export function lt(a: Scalar, b: Scalar): Tensor { return compare("lt", a, b); }
/** a > b -> 1.0/0.0. */
export function gt(a: Scalar, b: Scalar): Tensor { return compare("gt", a, b); }
/** a <= b -> 1.0/0.0. */
export function le(a: Scalar, b: Scalar): Tensor { return compare("le", a, b); }
/** a >= b -> 1.0/0.0. */
export function ge(a: Scalar, b: Scalar): Tensor { return compare("ge", a, b); }

// ============================================================================ unary

/** −x. */
export function neg(x: Tensor): Tensor {
  const out = unaryRaw("scale", x, -1);
  return record(out, "neg", [x], (g) => [unaryRaw("scale", g, -1)]);
}

/** eˣ. */
export function exp(x: Tensor): Tensor {
  const out = unaryRaw("exp", x);
  return record(out, "exp", [x], (g) => [binaryRaw("mul", g, out)]);
}

/** Natural log. */
export function log(x: Tensor): Tensor {
  const out = unaryRaw("log", x);
  return record(out, "log", [x], (g) => [binaryRaw("div", g, x)]);
}

/** √x. */
export function sqrt(x: Tensor): Tensor {
  const out = unaryRaw("sqrt", x);
  return record(out, "sqrt", [x], (g) => [binaryRaw("div", unaryRaw("scale", g, 0.5), out)]);
}

/** 1/√x. */
export function rsqrt(x: Tensor): Tensor {
  const out = unaryRaw("rsqrt", x);
  return record(out, "rsqrt", [x], (g) => [
    binaryRaw("mul", g, unaryRaw("scale", unaryRaw("powScalar", out, 3), -0.5)),
  ]);
}

/** tanh(x). */
export function tanh(x: Tensor): Tensor {
  const out = unaryRaw("tanh", x);
  return record(out, "tanh", [x], (g) => [binaryRaw("mul", g, unaryRaw("tanhGrad", x))]);
}

/** Logistic sigmoid. */
export function sigmoid(x: Tensor): Tensor {
  const out = unaryRaw("sigmoid", x);
  return record(out, "sigmoid", [x], (g) => [binaryRaw("mul", g, unaryRaw("sigmoidGrad", x))]);
}

/** max(x, 0). */
export function relu(x: Tensor): Tensor {
  const out = unaryRaw("relu", x);
  return record(out, "relu", [x], (g) => [binaryRaw("mul", g, unaryRaw("reluGrad", x))]);
}

/** GELU (tanh approximation). */
export function gelu(x: Tensor): Tensor {
  const out = unaryRaw("gelu", x);
  return record(out, "gelu", [x], (g) => [binaryRaw("mul", g, unaryRaw("geluGrad", x))]);
}

/** x². */
export function square(x: Tensor): Tensor {
  const out = unaryRaw("square", x);
  return record(out, "square", [x], (g) => [binaryRaw("mul", g, unaryRaw("scale", x, 2))]);
}

/** |x| (gradient 0 at x = 0). */
export function abs(x: Tensor): Tensor {
  const out = unaryRaw("abs", x);
  return record(out, "abs", [x], (g) => [
    binaryRaw("mul", g, binaryRaw("sub", binaryRaw("gt", x, scalarTensor(0, x.device)), binaryRaw("lt", x, scalarTensor(0, x.device)))),
  ]);
}

/** 1/x. */
export function reciprocal(x: Tensor): Tensor {
  const out = unaryRaw("recip", x);
  return record(out, "reciprocal", [x], (g) => [unaryRaw("scale", binaryRaw("mul", g, binaryRaw("mul", out, out)), -1)]);
}

/** sin(x). */
export function sin(x: Tensor): Tensor {
  const out = unaryRaw("sin", x);
  return record(out, "sin", [x], (g) => [binaryRaw("mul", g, unaryRaw("cos", x))]);
}

/** cos(x). */
export function cos(x: Tensor): Tensor {
  const out = unaryRaw("cos", x);
  return record(out, "cos", [x], (g) => [unaryRaw("scale", binaryRaw("mul", g, unaryRaw("sin", x)), -1)]);
}

/** clamp(x, min, max); either bound may be omitted. */
export function clamp(x: Tensor, min?: number, max?: number): Tensor {
  const lo = min ?? -Infinity;
  const hi = max ?? Infinity;
  const out = unaryRaw("clamp", x, lo, hi);
  return record(out, "clamp", [x], (g) => [binaryRaw("mul", g, binaryRaw("eq", out, x))]);
}

// ============================================================================ softmax family

function softmaxImpl(x: Tensor, dim: number, logMode: boolean): Tensor {
  if (x.ndim === 0) throw new Error("softmax requires a tensor with at least one dimension");
  const d = normalizeAxis(dim, x.ndim);
  const last = x.ndim - 1;
  if (d !== last) {
    return transpose(softmaxImpl(transpose(x, d, last), last, logMode), d, last);
  }
  const cols = x.shape[last] as number;
  const rows = cols === 0 ? 0 : x.numel / cols;
  const xd = dense(x);
  const out = emptyLike(x, x.shape, "f32");
  x.backend.softmax(xd.buf, rows, cols, out.buf, logMode);
  return record(out, logMode ? "logSoftmax" : "softmax", [x], (g) => {
    const gd = dense(g);
    const gi = emptyLike(x, x.shape, "f32");
    x.backend.softmaxBackward(out.buf, gd.buf, rows, cols, gi.buf, logMode);
    return [gi];
  });
}

/** Softmax along `dim`. */
export function softmax(x: Tensor, dim = -1): Tensor {
  return softmaxImpl(x, dim, false);
}

/** Log-softmax along `dim`. */
export function logSoftmax(x: Tensor, dim = -1): Tensor {
  return softmaxImpl(x, dim, true);
}

// ============================================================================ reductions

type Axes = number | readonly number[] | undefined;

/** Σ over `axes` (all when omitted). */
export function sum(x: Tensor, axes?: Axes, keepDim = false): Tensor {
  const ax = normalizeAxes(axes, x.ndim);
  if (ax.length === 0) return x;
  const out = reduceRaw("sum", x, ax);
  const keepShape = reduceShape(x.shape, ax, true);
  const res = record(out, "sum", [x], (g) => [expandBack(g, keepShape, x.shape)]);
  return keepDim ? reshape(res, keepShape) : res;
}

/** Arithmetic mean over `axes`. */
export function mean(x: Tensor, axes?: Axes, keepDim = false): Tensor {
  const ax = normalizeAxes(axes, x.ndim);
  if (ax.length === 0) return x;
  let n = 1;
  for (const a of ax) n *= x.shape[a] as number;
  const out = unaryRaw("scale", reduceRaw("sum", x, ax), 1 / n);
  const keepShape = reduceShape(x.shape, ax, true);
  const res = record(out, "mean", [x], (g) => [expandBack(unaryRaw("scale", g, 1 / n), keepShape, x.shape)]);
  return keepDim ? reshape(res, keepShape) : res;
}

function minMaxReduce(op: "max" | "min", x: Tensor, axes: Axes, keepDim: boolean): Tensor {
  const ax = normalizeAxes(axes, x.ndim);
  if (ax.length === 0) return x;
  const out = reduceRaw(op, x, ax);
  const keepShape = reduceShape(x.shape, ax, true);
  const res = record(out, op, [x], (g) => {
    const outK = viewRaw(out, keepShape, contiguousStrides(keepShape), 0);
    const mask = binaryRaw("eq", x, outK);
    const count = viewRaw(reduceRaw("sum", mask, ax), keepShape, contiguousStrides(keepShape), 0);
    const gk = viewRaw(dense(g), keepShape, contiguousStrides(keepShape), 0);
    return [binaryRaw("mul", mask, binaryRaw("div", gk, count))];
  });
  return keepDim ? reshape(res, keepShape) : res;
}

/** Maximum over `axes` (ties share the gradient equally). */
export function max(x: Tensor, axes?: Axes, keepDim = false): Tensor {
  return minMaxReduce("max", x, axes, keepDim);
}

/** Minimum over `axes`. */
export function min(x: Tensor, axes?: Axes, keepDim = false): Tensor {
  return minMaxReduce("min", x, axes, keepDim);
}

/** Variance over `axes` (unbiased by default, like PyTorch). */
export function variance(x: Tensor, axes?: Axes, opts: { unbiased?: boolean; keepDim?: boolean } = {}): Tensor {
  const ax = normalizeAxes(axes, x.ndim);
  let n = 1;
  for (const a of ax) n *= x.shape[a] as number;
  const m = mean(x, ax, true);
  const d = sub(x, m);
  const s = sum(square(d), ax, opts.keepDim ?? false);
  return div(s, Math.max(1, n - ((opts.unbiased ?? true) ? 1 : 0)));
}

/** Standard deviation over `axes`. */
export function std(x: Tensor, axes?: Axes, opts: { unbiased?: boolean; keepDim?: boolean } = {}): Tensor {
  return sqrt(variance(x, axes, opts));
}

/** Index of the maximum along `dim` (flattened when omitted). Returns i32; no gradient. */
export function argmax(x: Tensor, dim?: number, keepDim = false): Tensor {
  if (dim === undefined) {
    const xd = dense(x);
    const out = Tensor.empty(keepDim ? x.shape.map(() => 1) : [], { device: x.device, dtype: "i32" });
    x.backend.argmax(xd.buf, 1, x.numel, out.buf);
    return out;
  }
  const d = normalizeAxis(dim, x.ndim);
  const last = x.ndim - 1;
  const xt = d === last ? x : transpose(x, d, last);
  const xd = dense(xt);
  const cols = xt.shape[last] as number;
  const rows = cols === 0 ? 0 : xt.numel / cols;
  const outShape = removeDim(x.shape, d);
  const out = Tensor.empty(outShape, { device: x.device, dtype: "i32" });
  x.backend.argmax(xd.buf, rows, cols, out.buf);
  return keepDim ? viewRaw(out, insertDim(outShape, d), contiguousStrides(insertDim(outShape, d)), 0) : out;
}

// ============================================================================ matmul

interface MatmulOperand {
  buf: Buf;
  trans: boolean;
  batchStride: number;
}

/** Decide how a [..., R, C] operand feeds the GEMM: fused transpose, broadcast batch, or materialise. */
function prepareOperand(t: Tensor, batchShape: number[]): MatmulOperand {
  const nd = t.ndim;
  const R = t.shape[nd - 2] as number;
  const C = t.shape[nd - 1] as number;
  const sR = t.strides[nd - 2] as number;
  const sC = t.strides[nd - 1] as number;
  const rowMajor = (C === 1 || sC === 1) && (R === 1 || sR === C);
  const colMajor = !rowMajor && (R === 1 || sR === 1) && (C === 1 || sC === R);

  const bRank = batchShape.length;
  const tBatch = t.shape.slice(0, nd - 2);
  let allOnes = true;
  for (const d of tBatch) if (d !== 1) allOnes = false;

  let batchOk = true;
  let batchStride = 0;
  if (!allOnes) {
    const expected = contiguousStrides(batchShape);
    const pad = bRank - tBatch.length;
    for (let i = 0; i < bRank; i++) {
      const size = i < pad ? 1 : (tBatch[i - pad] as number);
      // A missing (padded) or size-1 dim can only be fused when the broadcast size is 1 as well;
      // otherwise the flat batch index would not map onto this operand's matrices.
      if (size !== batchShape[i]) {
        batchOk = false;
        break;
      }
      if (size === 1) continue;
      const stride = t.strides[i - pad] as number;
      if (stride !== (expected[i] as number) * R * C) {
        batchOk = false;
        break;
      }
    }
    batchStride = R * C;
  }

  if (t.offset === 0 && (rowMajor || colMajor) && batchOk) {
    return { buf: t.buf, trans: colMajor, batchStride };
  }
  if (allOnes) {
    return { buf: dense(t).buf, trans: false, batchStride: 0 };
  }
  const full = [...batchShape, R, C];
  const e = expandStrides(t.shape, t.strides, full);
  const m = contiguousRaw(viewRaw(t, e.shape, e.strides, t.offset));
  return { buf: m.buf, trans: false, batchStride: R * C };
}

/** Matrix product with NumPy semantics: 1-D vectors, 2-D matrices, batched with broadcast batch dims. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  assertDevice(a, b, "matmul");
  assertFloat(a, "matmul");
  if (a.ndim === 0 || b.ndim === 0) throw new Error("matmul: both arguments must have at least one dimension");
  if (a.ndim === 1 && b.ndim === 1) return sum(mul(a, b));
  if (a.ndim === 1) return squeeze(matmul(unsqueeze(a, 0), b), -2);
  if (b.ndim === 1) return squeeze(matmul(a, unsqueeze(b, 1)), -1);

  const M = a.shape[a.ndim - 2] as number;
  const K = a.shape[a.ndim - 1] as number;
  const K2 = b.shape[b.ndim - 2] as number;
  const N = b.shape[b.ndim - 1] as number;
  if (K !== K2) throw new Error(`matmul: inner dimensions mismatch ([${a.shape.join(",")}] @ [${b.shape.join(",")}])`);
  const batchShape = broadcastShapes(a.shape.slice(0, -2), b.shape.slice(0, -2));
  const batch = sizeOf(batchShape);
  const opA = prepareOperand(a, batchShape);
  const opB = prepareOperand(b, batchShape);
  const out = emptyLike(a, [...batchShape, M, N], "f32");
  if (out.numel > 0 && K > 0) {
    a.backend.matmul(opA.buf, opB.buf, out.buf, batch, M, K, N, opA.trans, opB.trans, opA.batchStride, opB.batchStride);
  } else if (out.numel > 0) {
    a.backend.fill(out.buf, out.numel, 0);
  }
  return record(out, "matmul", [a, b], (g) => [
    unbroadcast(matmul(g, transpose(b, -1, -2)), a.shape),
    unbroadcast(matmul(transpose(a, -1, -2), g), b.shape),
  ]);
}

// ============================================================================ views

/** Swap two dimensions (zero-copy). */
export function transpose(x: Tensor, dim0 = -2, dim1 = -1): Tensor {
  const d0 = normalizeAxis(dim0, x.ndim);
  const d1 = normalizeAxis(dim1, x.ndim);
  const perm: number[] = [];
  for (let i = 0; i < x.ndim; i++) perm.push(i);
  perm[d0] = d1;
  perm[d1] = d0;
  return permute(x, perm);
}

/** Reorder dimensions (zero-copy). */
export function permute(x: Tensor, perm: readonly number[]): Tensor {
  const p = checkPermutation(perm, x.ndim);
  const out = viewRaw(x, permuteShape(x.shape, p), permuteShape(x.strides, p), x.offset);
  const inv = invertPermutation(p);
  return record(out, "permute", [x], (g) => [permute(g, inv)]);
}

/** Reshape (zero-copy when the input is contiguous, otherwise copies). */
export function reshape(x: Tensor, shape: Shape): Tensor {
  const newShape = inferReshape(shape, x.numel);
  if (shapesEqual(newShape, x.shape)) return x;
  if (!x.isContiguous) return reshape(contiguous(x), newShape);
  const out = viewRaw(x, newShape, contiguousStrides(newShape), x.offset);
  const inShape = x.shape;
  return record(out, "reshape", [x], (g) => [reshape(g, inShape)]);
}

/** Reshape that requires a contiguous input (throws otherwise). */
export function view(x: Tensor, shape: Shape): Tensor {
  if (!x.isContiguous) throw new Error("view: tensor is not contiguous; use reshape() or contiguous()");
  return reshape(x, shape);
}

/** Flatten dims [start, end] into one. */
export function flatten(x: Tensor, start = 0, end = -1): Tensor {
  if (x.ndim === 0) return reshape(x, [1]);
  const s = normalizeAxis(start, x.ndim);
  const e = normalizeAxis(end, x.ndim);
  if (e < s) throw new Error("flatten: end must be >= start");
  const shape = [...x.shape.slice(0, s), sizeOf(x.shape.slice(s, e + 1)), ...x.shape.slice(e + 1)];
  return reshape(x, shape);
}

/** Remove size-1 dimensions (all, or just `dim`). */
export function squeeze(x: Tensor, dim?: number): Tensor {
  const shape: number[] = [];
  const strides: number[] = [];
  const d = dim === undefined ? -1 : normalizeAxis(dim, x.ndim);
  for (let i = 0; i < x.ndim; i++) {
    const keep = dim === undefined ? x.shape[i] !== 1 : !(i === d && x.shape[i] === 1);
    if (keep) {
      shape.push(x.shape[i] as number);
      strides.push(x.strides[i] as number);
    }
  }
  if (shape.length === x.ndim) return x;
  const out = viewRaw(x, shape, strides, x.offset);
  const inShape = x.shape;
  return record(out, "squeeze", [x], (g) => [reshape(g, inShape)]);
}

/** Insert a size-1 dimension at `dim` (may equal ndim). */
export function unsqueeze(x: Tensor, dim: number): Tensor {
  const d = dim < 0 ? dim + x.ndim + 1 : dim;
  if (d < 0 || d > x.ndim) throw new Error(`unsqueeze: dim ${dim} out of range`);
  const shape = insertDim(x.shape, d);
  const strides = x.strides.slice();
  const newStride = d < x.ndim ? (x.shape[d] as number) * (x.strides[d] as number) : 1;
  strides.splice(d, 0, newStride);
  const out = viewRaw(x, shape, strides, x.offset);
  const inShape = x.shape;
  return record(out, "unsqueeze", [x], (g) => [reshape(g, inShape)]);
}

/** Broadcast view to `shape` (-1 keeps a size). */
export function expand(x: Tensor, shape: Shape): Tensor {
  const e = expandStrides(x.shape, x.strides, shape);
  if (shapesEqual(e.shape, x.shape)) return x;
  const out = viewRaw(x, e.shape, e.strides, x.offset);
  return record(out, "expand", [x], (g) => [unbroadcast(g, x.shape)]);
}

/** Alias of `expand`. */
export function broadcastTo(x: Tensor, shape: Shape): Tensor {
  return expand(x, shape);
}

/** Write `g` into a zero tensor of `inShape` at the slice positions (inverse of slice). No autograd. */
function sliceBackward(g: Tensor, inShape: Shape, dim: number, start: number, step: number): Tensor {
  const out = Tensor.zeros(inShape, { device: g.device, dtype: g.dtype });
  const len = g.shape[dim] as number;
  if (len === 0 || out.numel === 0) return out;
  const gd = dense(g);
  const es = elemBytes(g);
  const outer = sizeOf(inShape.slice(0, dim));
  const inner = sizeOf(inShape.slice(dim + 1));
  const sizeD = inShape[dim] as number;
  const be = g.backend;
  if (step === 1) {
    const chunk = len * inner * es;
    for (let o = 0; o < outer; o++) be.copy(gd.buf, out.buf, chunk, o * chunk, (o * sizeD + start) * inner * es);
  } else {
    const run = inner * es;
    for (let o = 0; o < outer; o++) {
      for (let j = 0; j < len; j++) {
        be.copy(gd.buf, out.buf, run, (o * len + j) * run, (o * sizeD + start + j * step) * run);
      }
    }
  }
  return out;
}

/** Python-style slice `x[..., start:end:step, ...]` along `dim` (zero-copy). */
export function slice(x: Tensor, dim: number, start?: number, end?: number, step = 1): Tensor {
  const d = normalizeAxis(dim, x.ndim);
  const r = sliceStridesOffset(x.shape, x.strides, x.offset, d, start, end, step);
  if (shapesEqual(r.shape, x.shape) && step === 1) return x;
  const out = viewRaw(x, r.shape, r.strides, r.offset);
  const size = x.shape[d] as number;
  const s0 = start === undefined ? 0 : Math.min(Math.max(start < 0 ? start + size : start, 0), size);
  const inShape = x.shape;
  return record(out, "slice", [x], (g) => [sliceBackward(g, inShape, d, s0, step)]);
}

/** `length` elements along `dim` starting at `start`. */
export function narrow(x: Tensor, dim: number, start: number, length: number): Tensor {
  const d = normalizeAxis(dim, x.ndim);
  const size = x.shape[d] as number;
  const s = start < 0 ? start + size : start;
  if (s < 0 || s + length > size) throw new Error(`narrow: range [${s}, ${s + length}) exceeds size ${size}`);
  return slice(x, d, s, s + length, 1);
}

/** Concatenate along `dim`. */
export function cat(tensors: readonly Tensor[], dim = 0): Tensor {
  if (tensors.length === 0) throw new Error("cat: need at least one tensor");
  const ref = tensors[0] as Tensor;
  if (tensors.length === 1) return ref;
  const d = normalizeAxis(dim, ref.ndim);
  let total = 0;
  for (const t of tensors) {
    assertDevice(ref, t, "cat");
    if (t.ndim !== ref.ndim || t.dtype !== ref.dtype) throw new Error("cat: tensors must have the same rank and dtype");
    for (let i = 0; i < ref.ndim; i++) {
      if (i !== d && t.shape[i] !== ref.shape[i]) {
        throw new Error(`cat: shape mismatch [${t.shape.join(",")}] vs [${ref.shape.join(",")}] along dim ${i}`);
      }
    }
    total += t.shape[d] as number;
  }
  const outShape = ref.shape.slice();
  outShape[d] = total;
  const out = emptyLike(ref, outShape);
  const outer = sizeOf(outShape.slice(0, d));
  const inner = sizeOf(outShape.slice(d + 1));
  const es = elemBytes(ref);
  const be = ref.backend;
  let off = 0;
  for (const t of tensors) {
    const sz = t.shape[d] as number;
    const chunk = sz * inner;
    if (chunk > 0) {
      const td = dense(t);
      for (let o = 0; o < outer; o++) be.copy(td.buf, out.buf, chunk * es, o * chunk * es, (o * total + off) * inner * es);
    }
    off += sz;
  }
  return record(out, "cat", tensors, (g) => {
    let o = 0;
    return tensors.map((t) => {
      const sz = t.shape[d] as number;
      const r = sliceStridesOffset(g.shape, g.strides, g.offset, d, o, o + sz, 1);
      o += sz;
      return contiguousRaw(viewRaw(g, r.shape, r.strides, r.offset));
    });
  });
}

/** Stack along a new dimension. */
export function stack(tensors: readonly Tensor[], dim = 0): Tensor {
  if (tensors.length === 0) throw new Error("stack: need at least one tensor");
  const nd = (tensors[0] as Tensor).ndim + 1;
  const d = dim < 0 ? dim + nd : dim;
  return cat(tensors.map((t) => unsqueeze(t, d)), d);
}

/** Dense copy (returns `x` itself when already contiguous). */
export function contiguous(x: Tensor): Tensor {
  if (x.isContiguous) return x;
  const out = contiguousRaw(x);
  return record(out, "contiguous", [x], (g) => [g]);
}

/** Always-fresh copy with identity gradient. */
export function clone(x: Tensor): Tensor {
  const out = cloneRaw(x);
  return record(out, "clone", [x], (g) => [g]);
}

// ============================================================================ indexing

function assertIds(ids: Tensor, op: string): void {
  if (ids.dtype !== "i32") throw new Error(`${op}: indices must be an i32 tensor (got ${ids.dtype})`);
}

/** out[..., :] = table[ids[...], :] - embedding lookup. table is [V, C]; ids any shape of i32. */
export function embedding(table: Tensor, ids: Tensor): Tensor {
  assertIds(ids, "embedding");
  assertDevice(table, ids, "embedding");
  if (table.ndim !== 2) throw new Error("embedding: table must be 2-D [vocab, dim]");
  const C = table.shape[1] as number;
  const V = table.shape[0] as number;
  const n = ids.numel;
  const td = dense(table);
  const idd = dense(ids);
  const out = emptyLike(table, [...ids.shape, C], "f32");
  if (n > 0) table.backend.gather(td.buf, idd.buf, n, C, out.buf);
  return record(out, "embedding", [table], (g) => {
    const gt = Tensor.zeros([V, C], { device: table.device });
    if (n > 0) table.backend.scatterAdd(dense(g).buf, idd.buf, n, C, gt.buf);
    return [gt];
  }, fresh([idd], [ids]));
}

/** Select rows along `dim` by an i32 index tensor (any shape; result replaces `dim` with ids.shape). */
export function indexSelect(x: Tensor, dim: number, ids: Tensor): Tensor {
  assertIds(ids, "indexSelect");
  const d = normalizeAxis(dim, x.ndim);
  if (d !== 0) {
    const perm: number[] = [d];
    for (let i = 0; i < x.ndim; i++) if (i !== d) perm.push(i);
    const picked = indexSelect(permute(x, perm), 0, ids);
    // picked: [...ids.shape, rest...] -> move the ids dims back to position d.
    const k = ids.ndim;
    const inv: number[] = [];
    for (let i = 0; i < x.ndim - 1; i++) {
      if (i === d) for (let j = 0; j < k; j++) inv.push(j);
      inv.push(k + i);
    }
    if (d === x.ndim - 1) for (let j = 0; j < k; j++) inv.push(j);
    return permute(picked, inv);
  }
  const rest = x.shape.slice(1);
  const C = sizeOf(rest);
  const n = ids.numel;
  const table = reshape(x, [x.shape[0] as number, C]);
  const picked = embedding(table, ids);
  return reshape(picked, [...ids.shape, ...rest]);
}

/** Alias of `indexSelect`. */
export function gather(x: Tensor, dim: number, ids: Tensor): Tensor {
  return indexSelect(x, dim, ids);
}

/**
 * Out-of-place scatter-add along dim 0: result = target.clone(); result[ids[i], ...] += src[i, ...].
 * `src` is shaped [...ids.shape, ...target.shape[1:]]. Differentiable w.r.t. target and src.
 */
export function scatterAdd(target: Tensor, ids: Tensor, src: Tensor): Tensor {
  assertIds(ids, "scatterAdd");
  assertDevice(target, src, "scatterAdd");
  const rest = target.shape.slice(1);
  const C = sizeOf(rest);
  const n = ids.numel;
  if (src.numel !== n * C) throw new Error(`scatterAdd: src [${src.shape.join(",")}] must hold ids.numel × ${C} elements`);
  const out = cloneRaw(target);
  const idd = dense(ids);
  if (n > 0) target.backend.scatterAdd(dense(src).buf, idd.buf, n, C, out.buf);
  return record(out, "scatterAdd", [target, src], (g) => [
    g,
    reshape(indexSelect(reshape(g, [target.shape[0] as number, C]), 0, idd), src.shape),
  ], fresh([idd], [ids]));
}

// ============================================================================ selection / masking

function expandDense(t: Tensor, shape: Shape): Tensor {
  if (shapesEqual(t.shape, shape)) return dense(t);
  const e = expandStrides(t.shape, t.strides, shape);
  return contiguousRaw(viewRaw(t, e.shape, e.strides, t.offset));
}

/** cond ? a : b with broadcasting. `cond` is any dtype (non-zero = true). */
export function where(cond: Tensor, a: Scalar, b: Scalar): Tensor {
  const at = asTensor(a, cond);
  const bt = asTensor(b, cond);
  assertDevice(cond, at, "where");
  assertDevice(cond, bt, "where");
  const shape = broadcastShapesN([cond.shape, at.shape, bt.shape]);
  const n = sizeOf(shape);
  const cd = expandDense(cond, shape);
  const ad = expandDense(at, shape);
  const bd = expandDense(bt, shape);
  const out = emptyLike(at, shape, "f32");
  if (n > 0) cond.backend.where(cd.buf, ad.buf, bd.buf, n, out.buf);
  return record(out, "where", [at, bt], (g) => {
    const gd = dense(g);
    const zero = Tensor.zeros(shape, { device: g.device });
    const ga = emptyLike(g, shape);
    const gb = emptyLike(g, shape);
    cond.backend.where(cd.buf, gd.buf, zero.buf, n, ga.buf);
    cond.backend.where(cd.buf, zero.buf, gd.buf, n, gb.buf);
    return [unbroadcast(ga, at.shape), unbroadcast(gb, bt.shape)];
  }, fresh([cd], [cond]));
}

/** Fill positions where `mask` is non-zero with `value` (mask broadcasts to x). */
export function maskedFill(x: Tensor, mask: Tensor, value: number): Tensor {
  assertDevice(x, mask, "maskedFill");
  const md = expandDense(mask, x.shape);
  const xd = dense(x);
  const out = emptyLike(x, x.shape, "f32");
  const n = x.numel;
  if (n > 0) x.backend.maskedFill(xd.buf, md.buf, value, n, out.buf);
  return record(out, "maskedFill", [x], (g) => {
    const gi = emptyLike(g, x.shape);
    x.backend.maskedFill(dense(g).buf, md.buf, 0, n, gi.buf);
    return [gi];
  }, fresh([md], [mask]));
}

const triMaskCache = new Map<string, Tensor>();

/** f32 mask [rows, cols] with 1 where the element must be zeroed for tril/triu with diagonal `k`. */
function triMask(rows: number, cols: number, k: number, upper: boolean, device: Device): Tensor {
  const key = `${rows},${cols},${k},${upper ? "u" : "l"},${device}`;
  const hit = triMaskCache.get(key);
  if (hit) return hit;
  const data = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      // tril keeps j - i <= k (zero where j - i > k); triu keeps j - i >= k (zero where j - i < k).
      const zero = upper ? j - i < k : j - i > k;
      data[i * cols + j] = zero ? 1 : 0;
    }
  }
  const m = escape(Tensor.fromArray(data, [rows, cols], { device: "cpu" }).to(device));
  triMaskCache.set(key, m);
  return m;
}

/** Lower triangle of the last two dims (elements above diagonal `k` set to 0). */
export function tril(x: Tensor, k = 0): Tensor {
  if (x.ndim < 2) throw new Error("tril requires at least 2 dimensions");
  return maskedFill(x, triMask(x.shape[x.ndim - 2] as number, x.shape[x.ndim - 1] as number, k, false, x.device), 0);
}

/** Upper triangle of the last two dims (elements below diagonal `k` set to 0). */
export function triu(x: Tensor, k = 0): Tensor {
  if (x.ndim < 2) throw new Error("triu requires at least 2 dimensions");
  return maskedFill(x, triMask(x.shape[x.ndim - 2] as number, x.shape[x.ndim - 1] as number, k, true, x.device), 0);
}

// ============================================================================ losses / normalisation / dropout

/** Options for `crossEntropy`. */
export interface CrossEntropyOptions {
  /** Target value to ignore (default −100, like PyTorch). */
  ignoreIndex?: number;
  reduction?: "mean" | "sum" | "none";
  /**
   * Number of non-ignored targets. Counted automatically when targets live on the CPU; on the GPU it
   * defaults to the number of rows unless supplied here.
   */
  validCount?: number;
}

/**
 * Cross-entropy between logits [..., C] and i32 targets [...]. Fused log-softmax + NLL with the gradient
 * computed in the forward pass.
 */
export function crossEntropy(logits: Tensor, targets: Tensor, opts: CrossEntropyOptions = {}): Tensor {
  assertIds(targets, "crossEntropy");
  assertDevice(logits, targets, "crossEntropy");
  if (logits.ndim < 1) throw new Error("crossEntropy: logits must have a class dimension");
  const C = logits.shape[logits.ndim - 1] as number;
  const rows = C === 0 ? 0 : logits.numel / C;
  if (targets.numel !== rows) {
    throw new Error(`crossEntropy: targets [${targets.shape.join(",")}] do not match logits [${logits.shape.join(",")}]`);
  }
  const ignoreIndex = opts.ignoreIndex ?? -100;
  const reduction = opts.reduction ?? "mean";
  const ld = dense(logits);
  const td = dense(targets);
  let validCount = opts.validCount ?? rows;
  if (opts.validCount === undefined && targets.device === "cpu") {
    const ids = td.dataSync();
    let c = 0;
    for (let i = 0; i < rows; i++) if (ids[i] !== ignoreIndex) c++;
    validCount = c;
  }
  const denom = Math.max(1, validCount);
  const lossRows = emptyLike(logits, [rows], "f32");
  const wantGrad = isGradEnabled() && logits.requiresGrad;
  const gradBuf = wantGrad ? emptyLike(logits, [rows, C], "f32") : undefined;
  // For sum/none we ask the kernel for the unscaled gradient (denominator 1) and scale ourselves.
  logits.backend.crossEntropy(ld.buf, td.buf, rows, C, lossRows.buf, gradBuf?.buf, ignoreIndex, reduction === "mean" ? denom : 1);

  const saved = gradBuf ? [gradBuf] : [];
  if (reduction === "none") {
    const out = viewRaw(lossRows, targets.shape, contiguousStrides(targets.shape), 0);
    return record(out, "crossEntropy", [logits], (g) => {
      const gr = viewRaw(dense(g), [rows, 1], [1, 1], 0);
      return [viewRaw(binaryRaw("mul", gradBuf as Tensor, gr), logits.shape, contiguousStrides(logits.shape), 0)];
    }, saved);
  }
  const total = reduceRaw("sum", lossRows, [0]);
  const out = reduction === "mean" ? unaryRaw("scale", total, 1 / denom) : total;
  return record(out, "crossEntropy", [logits], (g) => [
    viewRaw(binaryRaw("mul", gradBuf as Tensor, g), logits.shape, contiguousStrides(logits.shape), 0),
  ], saved);
}

/** Layer normalisation over the last dimension with affine gamma/beta of shape [C]. */
export function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor {
  assertDevice(x, gamma, "layerNorm");
  assertDevice(x, beta, "layerNorm");
  if (x.ndim < 1) throw new Error("layerNorm requires at least one dimension");
  const C = x.shape[x.ndim - 1] as number;
  if (gamma.numel !== C || beta.numel !== C) throw new Error(`layerNorm: gamma/beta must have ${C} elements`);
  const rows = C === 0 ? 0 : x.numel / C;
  const xd = dense(x);
  const gd = dense(gamma);
  const bd = dense(beta);
  const out = emptyLike(x, x.shape, "f32");
  const meanRstd = emptyLike(x, [2 * rows], "f32");
  if (rows > 0) x.backend.layerNorm(xd.buf, gd.buf, bd.buf, rows, C, eps, out.buf, meanRstd.buf);
  return record(out, "layerNorm", [x, gamma, beta], (g) => {
    const gradX = emptyLike(x, x.shape);
    const gradGamma = emptyLike(x, gamma.shape);
    const gradBeta = emptyLike(x, beta.shape);
    if (rows > 0) {
      x.backend.layerNormBackward(xd.buf, gd.buf, meanRstd.buf, dense(g).buf, rows, C, gradX.buf, gradGamma.buf, gradBeta.buf);
    } else {
      x.backend.fill(gradGamma.buf, C, 0);
      x.backend.fill(gradBeta.buf, C, 0);
    }
    return [gradX, gradGamma, gradBeta];
  }, [...fresh([xd, gd], [x, gamma]), meanRstd]);
}

/** Inverted dropout: zero with probability p and scale survivors by 1/(1−p). Identity when not training. */
export function dropout(x: Tensor, p: number, training = true, seed?: number): Tensor {
  if (!training || p === 0) return x;
  if (p < 0 || p >= 1) throw new Error(`dropout: p must be in [0, 1), got ${p}`);
  const n = x.numel;
  const mask = emptyLike(x, x.shape, "f32");
  x.backend.dropoutMask(mask.buf, n, p, (seed ?? nextSeed()) >>> 0);
  const scaled = unaryRaw("scale", mask, 1 / (1 - p));
  const out = binaryRaw("mul", dense(x), scaled);
  return record(out, "dropout", [x], (g) => [binaryRaw("mul", dense(g), scaled)], [scaled]);
}
