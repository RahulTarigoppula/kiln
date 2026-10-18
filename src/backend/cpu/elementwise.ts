/**
 * CPU elementwise kernels: unary ops, broadcasting binary ops (via strides), strided copies.
 * Hot loops are plain indexed loops with the op switch hoisted outside.
 */
import type { BinaryOp, UnaryOp } from "../types";
import { contiguousStrides } from "../../tensor/shape";

const SQRT_2_OVER_PI = 0.7978845608028654;
const GELU_C = 0.044715;

/** GELU (tanh approximation), matching PyTorch's `approximate="tanh"`. */
export function geluScalar(x: number): number {
  return 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + GELU_C * x * x * x)));
}

/** d/dx of the tanh-approximate GELU. */
export function geluGradScalar(x: number): number {
  const u = SQRT_2_OVER_PI * (x + GELU_C * x * x * x);
  const t = Math.tanh(u);
  const du = SQRT_2_OVER_PI * (1 + 3 * GELU_C * x * x);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
}

/** out[i] = op(a[i]) for i < n. */
export function unaryKernel(op: UnaryOp, a: Float32Array, n: number, out: Float32Array, alpha: number, beta: number): void {
  switch (op) {
    case "neg": for (let i = 0; i < n; i++) out[i] = -(a[i] as number); break;
    case "exp": for (let i = 0; i < n; i++) out[i] = Math.exp(a[i] as number); break;
    case "log": for (let i = 0; i < n; i++) out[i] = Math.log(a[i] as number); break;
    case "sqrt": for (let i = 0; i < n; i++) out[i] = Math.sqrt(a[i] as number); break;
    case "rsqrt": for (let i = 0; i < n; i++) out[i] = 1 / Math.sqrt(a[i] as number); break;
    case "tanh": for (let i = 0; i < n; i++) out[i] = Math.tanh(a[i] as number); break;
    case "sigmoid": for (let i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-(a[i] as number))); break;
    case "relu": for (let i = 0; i < n; i++) { const v = a[i] as number; out[i] = v > 0 ? v : 0; } break;
    case "gelu": for (let i = 0; i < n; i++) out[i] = geluScalar(a[i] as number); break;
    case "square": for (let i = 0; i < n; i++) { const v = a[i] as number; out[i] = v * v; } break;
    case "abs": for (let i = 0; i < n; i++) out[i] = Math.abs(a[i] as number); break;
    case "recip": for (let i = 0; i < n; i++) out[i] = 1 / (a[i] as number); break;
    case "sin": for (let i = 0; i < n; i++) out[i] = Math.sin(a[i] as number); break;
    case "cos": for (let i = 0; i < n; i++) out[i] = Math.cos(a[i] as number); break;
    case "scale": for (let i = 0; i < n; i++) out[i] = (a[i] as number) * alpha; break;
    case "addScalar": for (let i = 0; i < n; i++) out[i] = (a[i] as number) + alpha; break;
    case "powScalar":
      if (alpha === 2) for (let i = 0; i < n; i++) { const v = a[i] as number; out[i] = v * v; }
      else if (alpha === 0.5) for (let i = 0; i < n; i++) out[i] = Math.sqrt(a[i] as number);
      else if (alpha === -1) for (let i = 0; i < n; i++) out[i] = 1 / (a[i] as number);
      else for (let i = 0; i < n; i++) out[i] = Math.pow(a[i] as number, alpha);
      break;
    case "geluGrad": for (let i = 0; i < n; i++) out[i] = geluGradScalar(a[i] as number); break;
    case "tanhGrad": for (let i = 0; i < n; i++) { const t = Math.tanh(a[i] as number); out[i] = 1 - t * t; } break;
    case "sigmoidGrad": for (let i = 0; i < n; i++) { const s = 1 / (1 + Math.exp(-(a[i] as number))); out[i] = s * (1 - s); } break;
    case "reluGrad": for (let i = 0; i < n; i++) out[i] = (a[i] as number) > 0 ? 1 : 0; break;
    case "clamp": {
      const lo = alpha;
      const hi = beta;
      for (let i = 0; i < n; i++) { const v = a[i] as number; out[i] = v < lo ? lo : v > hi ? hi : v; }
      break;
    }
    default: {
      const never: never = op;
      throw new Error(`Unknown unary op ${String(never)}`);
    }
  }
}

/** Inner row of a binary op: out[oOff + i] = op(a[aOff + i*as], b[bOff + i*bs]). */
function binaryRow(
  op: BinaryOp, A: Float32Array, aOff: number, as: number, B: Float32Array, bOff: number, bs: number,
  O: Float32Array, oOff: number, n: number,
): void {
  switch (op) {
    case "add": for (let i = 0; i < n; i++) O[oOff + i] = (A[aOff + i * as] as number) + (B[bOff + i * bs] as number); break;
    case "sub": for (let i = 0; i < n; i++) O[oOff + i] = (A[aOff + i * as] as number) - (B[bOff + i * bs] as number); break;
    case "mul": for (let i = 0; i < n; i++) O[oOff + i] = (A[aOff + i * as] as number) * (B[bOff + i * bs] as number); break;
    case "div": for (let i = 0; i < n; i++) O[oOff + i] = (A[aOff + i * as] as number) / (B[bOff + i * bs] as number); break;
    case "pow": for (let i = 0; i < n; i++) O[oOff + i] = Math.pow(A[aOff + i * as] as number, B[bOff + i * bs] as number); break;
    case "max": for (let i = 0; i < n; i++) { const x = A[aOff + i * as] as number; const y = B[bOff + i * bs] as number; O[oOff + i] = x >= y ? x : y; } break;
    case "min": for (let i = 0; i < n; i++) { const x = A[aOff + i * as] as number; const y = B[bOff + i * bs] as number; O[oOff + i] = x <= y ? x : y; } break;
    case "eq": for (let i = 0; i < n; i++) O[oOff + i] = (A[aOff + i * as] as number) === (B[bOff + i * bs] as number) ? 1 : 0; break;
    case "lt": for (let i = 0; i < n; i++) O[oOff + i] = (A[aOff + i * as] as number) < (B[bOff + i * bs] as number) ? 1 : 0; break;
    case "gt": for (let i = 0; i < n; i++) O[oOff + i] = (A[aOff + i * as] as number) > (B[bOff + i * bs] as number) ? 1 : 0; break;
    case "le": for (let i = 0; i < n; i++) O[oOff + i] = (A[aOff + i * as] as number) <= (B[bOff + i * bs] as number) ? 1 : 0; break;
    case "ge": for (let i = 0; i < n; i++) O[oOff + i] = (A[aOff + i * as] as number) >= (B[bOff + i * bs] as number) ? 1 : 0; break;
    default: {
      const never: never = op;
      throw new Error(`Unknown binary op ${String(never)}`);
    }
  }
}

/** Contiguous fast path (no strides). */
function binaryFlat(op: BinaryOp, A: Float32Array, B: Float32Array, O: Float32Array, n: number): void {
  switch (op) {
    case "add": for (let i = 0; i < n; i++) O[i] = (A[i] as number) + (B[i] as number); break;
    case "sub": for (let i = 0; i < n; i++) O[i] = (A[i] as number) - (B[i] as number); break;
    case "mul": for (let i = 0; i < n; i++) O[i] = (A[i] as number) * (B[i] as number); break;
    case "div": for (let i = 0; i < n; i++) O[i] = (A[i] as number) / (B[i] as number); break;
    default: binaryRow(op, A, 0, 1, B, 0, 1, O, 0, n);
  }
}

/** True when `strides` equal the contiguous strides of `shape` on every dim of size > 1. */
export function stridesAreContiguous(shape: number[], strides: number[], contig: number[]): boolean {
  for (let i = 0; i < shape.length; i++) {
    if ((shape[i] as number) > 1 && strides[i] !== contig[i]) return false;
  }
  return true;
}

/** Broadcasting binary op driven by explicit per-dim strides (0 for broadcast dims). */
export function binaryKernel(
  op: BinaryOp, A: Float32Array, B: Float32Array, O: Float32Array,
  outShape: number[], aStrides: number[], bStrides: number[],
): void {
  const nd = outShape.length;
  let n = 1;
  for (let i = 0; i < nd; i++) n *= outShape[i] as number;
  if (n === 0) return;
  if (nd === 0) {
    binaryRow(op, A, 0, 0, B, 0, 0, O, 0, 1);
    return;
  }
  const contig = contiguousStrides(outShape);
  if (stridesAreContiguous(outShape, aStrides, contig) && stridesAreContiguous(outShape, bStrides, contig)) {
    binaryFlat(op, A, B, O, n);
    return;
  }
  const last = nd - 1;
  const inner = outShape[last] as number;
  const as = aStrides[last] as number;
  const bs = bStrides[last] as number;
  const rows = n / inner;
  const idx = new Int32Array(nd);
  let aOff = 0;
  let bOff = 0;
  let oOff = 0;
  for (let r = 0; r < rows; r++) {
    binaryRow(op, A, aOff, as, B, bOff, bs, O, oOff, inner);
    oOff += inner;
    for (let d = last - 1; d >= 0; d--) {
      const sa = aStrides[d] as number;
      const sb = bStrides[d] as number;
      idx[d] = (idx[d] as number) + 1;
      aOff += sa;
      bOff += sb;
      if ((idx[d] as number) < (outShape[d] as number)) break;
      aOff -= sa * (outShape[d] as number);
      bOff -= sb * (outShape[d] as number);
      idx[d] = 0;
    }
  }
}

/**
 * Materialise a strided view into a contiguous array. Operates on 32-bit words (Int32Array views) so it
 * is dtype-agnostic.
 */
export function copyStridedKernel(src: Int32Array, shape: number[], strides: number[], offset: number, dst: Int32Array): void {
  const nd = shape.length;
  let n = 1;
  for (let i = 0; i < nd; i++) n *= shape[i] as number;
  if (n === 0) return;
  if (nd === 0) {
    dst[0] = src[offset] as number;
    return;
  }
  const last = nd - 1;
  const inner = shape[last] as number;
  const is = strides[last] as number;
  const rows = n / inner;
  const idx = new Int32Array(nd);
  let sOff = offset;
  let dOff = 0;
  for (let r = 0; r < rows; r++) {
    if (is === 1) {
      dst.set(src.subarray(sOff, sOff + inner), dOff);
    } else {
      for (let i = 0; i < inner; i++) dst[dOff + i] = src[sOff + i * is] as number;
    }
    dOff += inner;
    for (let d = last - 1; d >= 0; d--) {
      const s = strides[d] as number;
      idx[d] = (idx[d] as number) + 1;
      sOff += s;
      if ((idx[d] as number) < (shape[d] as number)) break;
      sOff -= s * (shape[d] as number);
      idx[d] = 0;
    }
  }
}
