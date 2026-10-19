import { describe, expect, it } from "vitest";
import { collapseDims, pickChunks } from "../src/backend/webgpu/index";

describe("WebGPU backend host helpers", () => {
  it("collapseDims merges contiguous dims and drops size-1 dims", () => {
    const c = collapseDims([2, 1, 3, 4], [[12, 12, 4, 1]]);
    expect(c.shape).toEqual([24]);
    expect(c.strides).toEqual([[1]]);
    const b = collapseDims([2, 3, 4], [[12, 4, 1], [0, 4, 1]]);
    expect(b.shape).toEqual([2, 12]);
    expect(b.strides).toEqual([[12, 1], [0, 1]]);
    expect(collapseDims([1, 1], [[1, 1]]).shape).toEqual([1]);
  });

  it("pickChunks returns a divisor in [2, 128] leaving >= 8 rows per chunk, or 1", () => {
    expect(pickChunks(4096)).toBe(128);
    expect(pickChunks(1000)).toBe(125);
    expect(pickChunks(600)).toBe(75);
    expect(pickChunks(1021)).toBe(1); // prime
    expect(pickChunks(15)).toBe(1); // too small for 8 rows per chunk
    for (const n of [512, 640, 777, 1024, 3000, 65536]) {
      const c = pickChunks(n);
      if (c > 1) {
        expect(n % c).toBe(0);
        expect(n / c).toBeGreaterThanOrEqual(8);
        expect(c).toBeLessThanOrEqual(128);
      }
    }
  });
});
