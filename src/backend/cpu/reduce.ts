/**
 * CPU reductions over arbitrary axes, row-wise argmax and sum of squares.
 */
import type { ReduceOp } from "../types";
import { contiguousStrides, reduceShape } from "../../tensor/shape";

/**
 * Reduce contiguous `a` of `shape` over `axes` (sorted ascending) into `out` (reduced shape).
 * Implemented as a strided accumulate: every input element maps to an output offset computed with the
 * output strides where reduced axes contribute 0.
 */
export function reduceKernel(op: ReduceOp, a: Float32Array, shape: number[], axes: number[], out: Float32Array): void {
  const nd = shape.length;
  let n = 1;
  for (let i = 0; i < nd; i++) n *= shape[i] as number;
  const outShape = reduceShape(shape, axes, false);
  let outN = 1;
  for (let i = 0; i < outShape.length; i++) outN *= outShape[i] as number;
  const init = op === "sum" ? 0 : op === "max" ? -Infinity : Infinity;
  out.fill(init, 0, outN);
  if (n === 0) return;
  if (nd === 0) {
    out[0] = a[0] as number;
    return;
  }
  // Output strides indexed by *input* dim (0 for reduced dims).
  const outContig = contiguousStrides(outShape);
  const oStrides = new Array<number>(nd);
  let oi = 0;
  for (let d = 0; d < nd; d++) {
    if (axes.includes(d)) oStrides[d] = 0;
    else oStrides[d] = outContig[oi++] as number;
  }
  const last = nd - 1;
  const inner = shape[last] as number;
  const lastReduced = (oStrides[last] as number) === 0;
  const rows = n / inner;
  const idx = new Int32Array(nd);
  let aOff = 0;
  let oOff = 0;
  for (let r = 0; r < rows; r++) {
    if (lastReduced) {
      let acc = out[oOff] as number;
      if (op === "sum") for (let i = 0; i < inner; i++) acc += a[aOff + i] as number;
      else if (op === "max") for (let i = 0; i < inner; i++) { const v = a[aOff + i] as number; if (v > acc) acc = v; }
      else for (let i = 0; i < inner; i++) { const v = a[aOff + i] as number; if (v < acc) acc = v; }
      out[oOff] = acc;
    } else {
      if (op === "sum") for (let i = 0; i < inner; i++) out[oOff + i] = (out[oOff + i] as number) + (a[aOff + i] as number);
      else if (op === "max") for (let i = 0; i < inner; i++) { const v = a[aOff + i] as number; if (v > (out[oOff + i] as number)) out[oOff + i] = v; }
      else for (let i = 0; i < inner; i++) { const v = a[aOff + i] as number; if (v < (out[oOff + i] as number)) out[oOff + i] = v; }
    }
    aOff += inner;
    for (let d = last - 1; d >= 0; d--) {
      const s = oStrides[d] as number;
      idx[d] = (idx[d] as number) + 1;
      oOff += s;
      if ((idx[d] as number) < (shape[d] as number)) break;
      oOff -= s * (shape[d] as number);
      idx[d] = 0;
    }
  }
}

/** Row-wise argmax (first maximal index) into an Int32Array of `rows`. */
export function argmaxKernel(a: Float32Array, rows: number, cols: number, out: Int32Array): void {
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let best = -Infinity;
    let bi = 0;
    for (let c = 0; c < cols; c++) {
      const v = a[base + c] as number;
      if (v > best) {
        best = v;
        bi = c;
      }
    }
    out[r] = bi;
  }
}

/** Sum of squares of `n` elements (double accumulation). */
export function sumSquaresKernel(a: Float32Array, n: number): number {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  const n4 = n & ~3;
  let i = 0;
  for (; i < n4; i += 4) {
    const v0 = a[i] as number, v1 = a[i + 1] as number, v2 = a[i + 2] as number, v3 = a[i + 3] as number;
    s0 += v0 * v0; s1 += v1 * v1; s2 += v2 * v2; s3 += v3 * v3;
  }
  for (; i < n; i++) { const v = a[i] as number; s0 += v * v; }
  return (s0 + s1) + (s2 + s3);
}
