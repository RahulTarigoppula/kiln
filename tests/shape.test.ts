import { describe, expect, it } from "vitest";
import {
  broadcastShapes, broadcastStrides, contiguousStrides, expandStrides, inferNestedShape, inferReshape, invertPermutation,
  isContiguous, normalizeAxes, normalizeAxis, offsetOfFlat, permuteShape, ravelIndex, reduceShape, sizeOf,
  sliceStridesOffset, unravelIndex,
} from "../src/tensor/shape";

describe("shape utils", () => {
  it("sizeOf and contiguousStrides", () => {
    expect(sizeOf([])).toBe(1);
    expect(sizeOf([2, 3, 4])).toBe(24);
    expect(sizeOf([2, 0, 4])).toBe(0);
    expect(contiguousStrides([2, 3, 4])).toEqual([12, 4, 1]);
    expect(contiguousStrides([])).toEqual([]);
  });

  it("broadcastShapes follows NumPy rules", () => {
    expect(broadcastShapes([3, 1], [1, 4])).toEqual([3, 4]);
    expect(broadcastShapes([5, 3, 1], [3, 4])).toEqual([5, 3, 4]);
    expect(broadcastShapes([], [2, 2])).toEqual([2, 2]);
    expect(() => broadcastShapes([3, 2], [2, 2])).toThrow();
  });

  it("broadcastStrides zeroes broadcast dims", () => {
    expect(broadcastStrides([3, 1], [1, 1], [3, 4])).toEqual([1, 0]);
    expect(broadcastStrides([4], [1], [2, 3, 4])).toEqual([0, 0, 1]);
    expect(broadcastStrides([], [], [2, 2])).toEqual([0, 0]);
    expect(() => broadcastStrides([3], [1], [4])).toThrow();
  });

  it("isContiguous ignores size-1 dims", () => {
    expect(isContiguous([2, 3], [3, 1])).toBe(true);
    expect(isContiguous([2, 3], [1, 2])).toBe(false);
    expect(isContiguous([1, 3], [99, 1])).toBe(true);
    expect(isContiguous([2, 1, 3], [3, 7, 1])).toBe(true);
    expect(isContiguous([2, 3], [6, 1])).toBe(false);
  });

  it("normalizeAxis / normalizeAxes", () => {
    expect(normalizeAxis(-1, 3)).toBe(2);
    expect(normalizeAxis(0, 3)).toBe(0);
    expect(() => normalizeAxis(3, 3)).toThrow();
    expect(normalizeAxes(undefined, 3)).toEqual([0, 1, 2]);
    expect(normalizeAxes([-1, 0, 0], 3)).toEqual([0, 2]);
    expect(normalizeAxes(1, 3)).toEqual([1]);
  });

  it("reduceShape", () => {
    expect(reduceShape([2, 3, 4], [1])).toEqual([2, 4]);
    expect(reduceShape([2, 3, 4], [1], true)).toEqual([2, 1, 4]);
    expect(reduceShape([2, 3], [0, 1])).toEqual([]);
  });

  it("permuteShape and invertPermutation", () => {
    expect(permuteShape([2, 3, 4], [2, 0, 1])).toEqual([4, 2, 3]);
    expect(invertPermutation([2, 0, 1])).toEqual([1, 2, 0]);
  });

  it("sliceStridesOffset", () => {
    const r = sliceStridesOffset([4, 6], [6, 1], 0, 1, 1, 5, 2);
    expect(r.shape).toEqual([4, 2]);
    expect(r.strides).toEqual([6, 2]);
    expect(r.offset).toBe(1);
    const neg = sliceStridesOffset([5], [1], 3, 0, -2, undefined, 1);
    expect(neg.shape).toEqual([2]);
    expect(neg.offset).toBe(6);
    const clamped = sliceStridesOffset([5], [1], 0, 0, 2, 100, 1);
    expect(clamped.shape).toEqual([3]);
    expect(() => sliceStridesOffset([5], [1], 0, 0, 0, 5, 0)).toThrow();
  });

  it("index <-> offset helpers", () => {
    expect(unravelIndex(7, [2, 3, 4])).toEqual([0, 1, 3]);
    expect(ravelIndex([0, 1, 3], [12, 4, 1])).toBe(7);
    expect(ravelIndex([1, 1], [1, 2], 5)).toBe(8);
    // transposed 3x2 view of a 2x3 buffer: strides [1, 3]
    expect(offsetOfFlat(1, [3, 2], [1, 3])).toBe(3);
    expect(offsetOfFlat(2, [3, 2], [1, 3])).toBe(1);
  });

  it("inferReshape resolves -1", () => {
    expect(inferReshape([-1, 4], 12)).toEqual([3, 4]);
    expect(inferReshape([2, -1], 12)).toEqual([2, 6]);
    expect(() => inferReshape([-1, -1], 12)).toThrow();
    expect(() => inferReshape([5, -1], 12)).toThrow();
    expect(() => inferReshape([3, 3], 12)).toThrow();
  });

  it("expandStrides", () => {
    const e = expandStrides([3, 1], [1, 1], [2, 3, 4]);
    expect(e.shape).toEqual([2, 3, 4]);
    expect(e.strides).toEqual([0, 1, 0]);
    expect(expandStrides([3], [1], [-1]).shape).toEqual([3]);
    expect(() => expandStrides([3], [1], [4])).toThrow();
  });

  it("inferNestedShape", () => {
    expect(inferNestedShape([[1, 2], [3, 4], [5, 6]])).toEqual([3, 2]);
    expect(inferNestedShape(5)).toEqual([]);
    expect(inferNestedShape([])).toEqual([0]);
  });
});
