/**
 * Shape and stride arithmetic shared by the Tensor class, autograd ops and the CPU backend.
 * Everything here is pure and allocation-light; nothing knows about buffers or devices.
 */

/** A tensor shape (element counts per dimension, outermost first). */
export type Shape = readonly number[];

/** Number of elements described by `shape` (1 for a 0-d shape). */
export function sizeOf(shape: Shape): number {
  let n = 1;
  for (let i = 0; i < shape.length; i++) n *= shape[i] as number;
  return n;
}

/** Row-major (C-order) strides for a contiguous tensor of `shape`. */
export function contiguousStrides(shape: Shape): number[] {
  const nd = shape.length;
  const strides = new Array<number>(nd);
  let s = 1;
  for (let i = nd - 1; i >= 0; i--) {
    strides[i] = s;
    s *= shape[i] as number;
  }
  return strides;
}

/** True when both shapes have identical ranks and sizes. */
export function shapesEqual(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** NumPy broadcasting of two shapes; throws when they are incompatible. */
export function broadcastShapes(a: Shape, b: Shape): number[] {
  const nd = Math.max(a.length, b.length);
  const out = new Array<number>(nd);
  for (let i = 0; i < nd; i++) {
    const da = i < nd - a.length ? 1 : (a[i - (nd - a.length)] as number);
    const db = i < nd - b.length ? 1 : (b[i - (nd - b.length)] as number);
    if (da === db || db === 1) out[i] = da;
    else if (da === 1) out[i] = db;
    else throw new Error(`Shapes [${a.join(",")}] and [${b.join(",")}] are not broadcastable`);
  }
  return out;
}

/** Broadcast any number of shapes together. */
export function broadcastShapesN(shapes: Shape[]): number[] {
  let out: number[] = [];
  for (const s of shapes) out = broadcastShapes(out, s);
  return out;
}

/**
 * Strides to read a tensor of `shape`/`strides` as if it had `outShape` (which it must broadcast to).
 * Broadcast dimensions (size 1 → size > 1, and newly prepended dimensions) get stride 0.
 */
export function broadcastStrides(shape: Shape, strides: Shape, outShape: Shape): number[] {
  const nd = outShape.length;
  const pad = nd - shape.length;
  if (pad < 0) throw new Error(`Cannot broadcast rank ${shape.length} to rank ${nd}`);
  const out = new Array<number>(nd);
  for (let i = 0; i < nd; i++) {
    if (i < pad) {
      out[i] = 0;
    } else {
      const d = shape[i - pad] as number;
      const o = outShape[i] as number;
      if (d === o) out[i] = strides[i - pad] as number;
      else if (d === 1) out[i] = 0;
      else throw new Error(`Cannot broadcast [${shape.join(",")}] to [${outShape.join(",")}]`);
    }
  }
  return out;
}

/** True when `strides` describe a dense row-major layout for `shape` (size-1 dims are ignored). */
export function isContiguous(shape: Shape, strides: Shape): boolean {
  let expected = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    const d = shape[i] as number;
    if (d === 0) return true;
    if (d !== 1 && strides[i] !== expected) return false;
    expected *= d;
  }
  return true;
}

/** Normalise a possibly-negative axis into [0, ndim). */
export function normalizeAxis(axis: number, ndim: number): number {
  const a = axis < 0 ? axis + ndim : axis;
  if (a < 0 || a >= ndim || !Number.isInteger(a)) {
    throw new Error(`Axis ${axis} out of range for rank ${ndim}`);
  }
  return a;
}

/** Normalise `axes` (undefined = all axes) into a sorted, de-duplicated ascending list. */
export function normalizeAxes(axes: number | readonly number[] | undefined, ndim: number): number[] {
  if (axes === undefined) {
    const all = new Array<number>(ndim);
    for (let i = 0; i < ndim; i++) all[i] = i;
    return all;
  }
  const list = typeof axes === "number" ? [axes] : axes;
  const seen = new Set<number>();
  for (const a of list) seen.add(normalizeAxis(a, ndim));
  return Array.from(seen).sort((x, y) => x - y);
}

/** Shape after reducing `axes`; with keepDim the reduced axes stay as size 1. */
export function reduceShape(shape: Shape, axes: readonly number[], keepDim = false): number[] {
  const out: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    if (axes.includes(i)) {
      if (keepDim) out.push(1);
    } else out.push(shape[i] as number);
  }
  return out;
}

/** Validate a permutation of [0, ndim). */
export function checkPermutation(perm: readonly number[], ndim: number): number[] {
  if (perm.length !== ndim) throw new Error(`Permutation [${perm.join(",")}] must have length ${ndim}`);
  const norm = perm.map((p) => normalizeAxis(p, ndim));
  const seen = new Set(norm);
  if (seen.size !== ndim) throw new Error(`Permutation [${perm.join(",")}] has repeated axes`);
  return norm;
}

/** Apply a permutation to a shape (or strides): out[i] = arr[perm[i]]. */
export function permuteShape(arr: Shape, perm: readonly number[]): number[] {
  const out = new Array<number>(perm.length);
  for (let i = 0; i < perm.length; i++) out[i] = arr[perm[i] as number] as number;
  return out;
}

/** Inverse of a permutation. */
export function invertPermutation(perm: readonly number[]): number[] {
  const inv = new Array<number>(perm.length);
  for (let i = 0; i < perm.length; i++) inv[perm[i] as number] = i;
  return inv;
}

/** Result of slicing one dimension of a strided view. */
export interface SliceResult {
  shape: number[];
  strides: number[];
  offset: number;
}

/**
 * Python-style slice `[start:end:step]` along `dim` for a strided view. Negative start/end count from the
 * end and out-of-range bounds are clamped; `step` must be a positive integer.
 */
export function sliceStridesOffset(
  shape: Shape,
  strides: Shape,
  offset: number,
  dim: number,
  start: number | undefined,
  end: number | undefined,
  step = 1,
): SliceResult {
  const d = normalizeAxis(dim, shape.length);
  const size = shape[d] as number;
  if (!Number.isInteger(step) || step <= 0) throw new Error(`slice step must be a positive integer, got ${step}`);
  let s = start === undefined ? 0 : start < 0 ? start + size : start;
  let e = end === undefined ? size : end < 0 ? end + size : end;
  s = Math.min(Math.max(s, 0), size);
  e = Math.min(Math.max(e, s), size);
  const len = Math.max(0, Math.ceil((e - s) / step));
  const outShape = shape.slice();
  const outStrides = strides.slice();
  outShape[d] = len;
  outStrides[d] = (strides[d] as number) * step;
  return { shape: outShape, strides: outStrides, offset: offset + s * (strides[d] as number) };
}

/** Multi-index of the `flat`-th element (row-major) of `shape`. */
export function unravelIndex(flat: number, shape: Shape): number[] {
  const nd = shape.length;
  const idx = new Array<number>(nd);
  let rem = flat;
  for (let i = nd - 1; i >= 0; i--) {
    const d = shape[i] as number;
    idx[i] = d === 0 ? 0 : rem % d;
    rem = d === 0 ? 0 : Math.floor(rem / d);
  }
  return idx;
}

/** Element offset of a multi-index under `strides` (plus a base offset). */
export function ravelIndex(idx: readonly number[], strides: Shape, offset = 0): number {
  let o = offset;
  for (let i = 0; i < idx.length; i++) o += (idx[i] as number) * (strides[i] as number);
  return o;
}

/** Element offset of the `flat`-th row-major element of a `shape`/`strides` view. */
export function offsetOfFlat(flat: number, shape: Shape, strides: Shape, offset = 0): number {
  let rem = flat;
  let o = offset;
  for (let i = shape.length - 1; i >= 0; i--) {
    const d = shape[i] as number;
    const k = rem % d;
    rem = (rem - k) / d;
    o += k * (strides[i] as number);
  }
  return o;
}

/** Resolve a reshape target that may contain a single -1 against `numel`. */
export function inferReshape(newShape: Shape, numel: number): number[] {
  const out = newShape.slice();
  let neg = -1;
  let known = 1;
  for (let i = 0; i < out.length; i++) {
    const d = out[i] as number;
    if (d === -1) {
      if (neg >= 0) throw new Error("Only one dimension may be -1 in reshape");
      neg = i;
    } else {
      if (d < 0 || !Number.isInteger(d)) throw new Error(`Invalid dimension ${d}`);
      known *= d;
    }
  }
  if (neg >= 0) {
    if (known === 0 || numel % known !== 0) throw new Error(`Cannot reshape ${numel} elements into [${newShape.join(",")}]`);
    out[neg] = numel / known;
  } else if (known !== numel) {
    throw new Error(`Cannot reshape ${numel} elements into [${newShape.join(",")}]`);
  }
  return out;
}

/** Strides for expanding a `shape`/`strides` view to `newShape` (-1 keeps a size). */
export function expandStrides(shape: Shape, strides: Shape, newShape: Shape): { shape: number[]; strides: number[] } {
  const nd = newShape.length;
  const pad = nd - shape.length;
  if (pad < 0) throw new Error(`expand cannot reduce rank ${shape.length} to ${nd}`);
  const outShape = new Array<number>(nd);
  const outStrides = new Array<number>(nd);
  for (let i = 0; i < nd; i++) {
    let want = newShape[i] as number;
    if (i < pad) {
      if (want === -1) throw new Error("expand: -1 is not allowed for new leading dimensions");
      outShape[i] = want;
      outStrides[i] = 0;
    } else {
      const have = shape[i - pad] as number;
      if (want === -1) want = have;
      if (have === want) {
        outShape[i] = want;
        outStrides[i] = strides[i - pad] as number;
      } else if (have === 1) {
        outShape[i] = want;
        outStrides[i] = 0;
      } else {
        throw new Error(`Cannot expand [${shape.join(",")}] to [${newShape.join(",")}]`);
      }
    }
  }
  return { shape: outShape, strides: outStrides };
}

/** Nested JS arrays of numbers (arbitrary depth). */
export type Nested = number | boolean | readonly Nested[];

/** Shape of a (rectangular) nested array. */
export function inferNestedShape(data: Nested): number[] {
  const shape: number[] = [];
  let cur: Nested = data;
  while (Array.isArray(cur)) {
    shape.push(cur.length);
    if (cur.length === 0) break;
    cur = cur[0] as Nested;
  }
  return shape;
}

/** Flatten a nested array into `out` (row-major), validating rectangularity against `shape`. */
export function flattenNested(data: Nested, shape: Shape, out: { set(i: number, v: number): void } | number[], pos = 0, depth = 0): number {
  if (depth === shape.length) {
    if (Array.isArray(data)) throw new Error("Nested array is ragged");
    const v = typeof data === "boolean" ? (data ? 1 : 0) : (data as number);
    if (Array.isArray(out)) out[pos] = v;
    else out.set(pos, v);
    return pos + 1;
  }
  if (!Array.isArray(data) || data.length !== shape[depth]) throw new Error("Nested array is ragged");
  let p = pos;
  for (let i = 0; i < data.length; i++) p = flattenNested(data[i] as Nested, shape, out, p, depth + 1);
  return p;
}

/** Shape with `dim` removed. */
export function removeDim(shape: Shape, dim: number): number[] {
  const out = shape.slice();
  out.splice(dim, 1);
  return out;
}

/** Shape with a size-1 dimension inserted at `dim` (dim may equal ndim). */
export function insertDim(shape: Shape, dim: number): number[] {
  const out = shape.slice();
  out.splice(dim, 0, 1);
  return out;
}
