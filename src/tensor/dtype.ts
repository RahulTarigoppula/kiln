/**
 * dtype helpers: element sizes and typed-array constructors.
 * All Kiln dtypes are 4 bytes wide, which keeps buffer reinterpretation trivial on both backends.
 */
import type { DType } from "../backend/types";

export type { DType };

/** Typed arrays a Tensor can expose. */
export type TypedArray = Float32Array | Int32Array | Uint32Array;

/** Constructor type for the three supported typed arrays. */
export type TypedArrayCtor = Float32ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor;

/** Bytes per element for every dtype. */
export const DTYPE_SIZE: Readonly<Record<DType, number>> = { f32: 4, i32: 4, u32: 4 };

/** Bytes per element of `dtype`. */
export function dtypeSize(dtype: DType): number {
  return DTYPE_SIZE[dtype];
}

/** Typed-array constructor for `dtype`. */
export function typedArrayCtor(dtype: DType): TypedArrayCtor {
  switch (dtype) {
    case "f32":
      return Float32Array;
    case "i32":
      return Int32Array;
    case "u32":
      return Uint32Array;
  }
}

/** Wrap an ArrayBuffer region as the typed array matching `dtype`. */
export function makeTypedArray(dtype: DType, buffer: ArrayBuffer, byteOffset = 0, length?: number): TypedArray {
  const Ctor = typedArrayCtor(dtype);
  return length === undefined ? new Ctor(buffer, byteOffset) : new Ctor(buffer, byteOffset, length);
}

/** Allocate a fresh zero-filled typed array of `n` elements for `dtype`. */
export function allocTypedArray(dtype: DType, n: number): TypedArray {
  return new (typedArrayCtor(dtype))(n);
}

/** Infer the dtype of a typed array (plain arrays are f32). */
export function dtypeOfArray(data: ArrayLike<number>): DType {
  if (data instanceof Int32Array) return "i32";
  if (data instanceof Uint32Array) return "u32";
  return "f32";
}

/** True when `dtype` is a floating-point type. */
export function isFloatDType(dtype: DType): boolean {
  return dtype === "f32";
}
