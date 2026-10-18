import { describe, expect, it } from "vitest";
import { TokenDataset } from "../src/data/index";
import { Random } from "../src/tensor/random";

const ids = Int32Array.from({ length: 1000 }, (_, i) => i);

describe("TokenDataset", () => {
  it("reports sizes", () => {
    const ds = new TokenDataset(ids, 8);
    expect(ds.length).toBe(1000);
    expect(ds.numBlocks).toBe(992);
    expect(() => new TokenDataset(ids, 0)).toThrow();
  });

  it("produces flat [B*T] batches where y is x shifted by one", () => {
    const ds = new TokenDataset(ids, 16);
    const { x, y } = ds.getBatch(4, new Random(1));
    expect(x).toBeInstanceOf(Int32Array);
    expect(x.length).toBe(64);
    expect(y.length).toBe(64);
    for (let b = 0; b < 4; b++) {
      for (let t = 0; t < 16; t++) {
        expect(y[b * 16 + t]).toBe((x[b * 16 + t] as number) + 1);
        if (t > 0) expect(x[b * 16 + t]).toBe((x[b * 16 + t - 1] as number) + 1);
      }
      expect(x[b * 16] as number).toBeLessThanOrEqual(ds.numBlocks - 1);
    }
  });

  it("is deterministic for a given seed and accepts a plain uniform function", () => {
    const ds = new TokenDataset(ids, 8);
    const a = ds.getBatch(8, new Random(42));
    const b = ds.getBatch(8, new Random(42));
    expect(Array.from(a.x)).toEqual(Array.from(b.x));
    const c = ds.getBatch(8, new Random(43));
    expect(Array.from(c.x)).not.toEqual(Array.from(a.x));
    const r = new Random(5);
    const d = ds.getBatch(3, () => r.uniform());
    expect(d.x.length).toBe(24);
    const edge = ds.getBatch(2, () => 0.999999999);
    expect(edge.x[0]).toBe(ds.numBlocks - 1);
  });

  it("splits into train/val without overlap", () => {
    const ds = new TokenDataset(ids, 8);
    const { train, val } = ds.trainValSplit(0.9);
    expect(train.length).toBe(900);
    expect(val.length).toBe(100);
    expect(train.ids[899]).toBe(899);
    expect(val.ids[0]).toBe(900);
    expect(() => ds.trainValSplit(1)).toThrow();
  });

  it("getBlock and sequential iteration cover the stream", () => {
    const ds = new TokenDataset(ids.slice(0, 50), 8);
    const blk = ds.getBlock(3);
    expect(Array.from(blk.x)).toEqual([3, 4, 5, 6, 7, 8, 9, 10]);
    expect(Array.from(blk.y)).toEqual([4, 5, 6, 7, 8, 9, 10, 11]);
    expect(() => ds.getBlock(42)).toThrow();
    let total = 0;
    for (const batch of ds.iterSequential(2)) {
      expect(batch.x.length).toBe(batch.count * 8);
      total += batch.count;
    }
    expect(total).toBe(Math.ceil(ds.numBlocks / 8));
  });

  it("throws when the stream is shorter than the block", () => {
    const ds = new TokenDataset(ids.slice(0, 8), 8);
    expect(() => ds.getBatch(1, new Random(0))).toThrow();
  });
});
