/**
 * Token-stream dataset producing random (x, y) language-modelling batches.
 */

import { Random } from "../tensor/random";

/** Source of uniform randomness accepted by {@link TokenDataset.getBatch}: a `Random` or any `() => [0,1)` function. */
export type BatchRng = Random | { int(n: number): number } | (() => number);

/** A flat token stream with fixed context length. */
export class TokenDataset {
  /** Context length T. */
  readonly blockSize: number;
  /** The token ids. */
  readonly ids: Int32Array;

  constructor(ids: Int32Array, blockSize: number) {
    if (!Number.isInteger(blockSize) || blockSize <= 0) throw new Error("TokenDataset: blockSize must be a positive integer");
    this.ids = ids;
    this.blockSize = blockSize;
  }

  /** Number of tokens. */
  get length(): number {
    return this.ids.length;
  }

  /** Number of valid block start positions (each block needs blockSize + 1 tokens). */
  get numBlocks(): number {
    return Math.max(0, this.ids.length - this.blockSize);
  }

  /** Split the stream into a leading train part and trailing validation part. */
  trainValSplit(trainFrac = 0.9): { train: TokenDataset; val: TokenDataset } {
    if (!(trainFrac > 0 && trainFrac < 1)) throw new Error("trainValSplit: fraction must be in (0, 1)");
    const cut = Math.floor(this.ids.length * trainFrac);
    return {
      train: new TokenDataset(this.ids.subarray(0, cut), this.blockSize),
      val: new TokenDataset(this.ids.subarray(cut), this.blockSize),
    };
  }

  /** (x, y) for the block starting at `start`: x = ids[start, start+T), y = ids[start+1, start+T+1). */
  getBlock(start: number): { x: Int32Array; y: Int32Array } {
    if (start < 0 || start >= this.numBlocks) throw new Error(`getBlock: start ${start} out of range [0, ${this.numBlocks})`);
    return {
      x: this.ids.slice(start, start + this.blockSize),
      y: this.ids.slice(start + 1, start + this.blockSize + 1),
    };
  }

  /**
   * Random batch of `batchSize` blocks as flat `[B*T]` arrays (row-major, row b = block b).
   */
  getBatch(batchSize: number, rng: BatchRng): { x: Int32Array; y: Int32Array } {
    const T = this.blockSize;
    const nb = this.numBlocks;
    if (nb <= 0) throw new Error(`getBatch: dataset has ${this.ids.length} tokens, need more than blockSize (${T})`);
    const x = new Int32Array(batchSize * T);
    const y = new Int32Array(batchSize * T);
    const pick = typeof rng === "function" ? (n: number): number => Math.min(n - 1, Math.floor(rng() * n)) : (n: number): number => rng.int(n);
    for (let b = 0; b < batchSize; b++) {
      const s = pick(nb);
      x.set(this.ids.subarray(s, s + T), b * T);
      y.set(this.ids.subarray(s + 1, s + T + 1), b * T);
    }
    return { x, y };
  }

  /** Sequential batches covering the stream without overlap (for evaluation). */
  *iterSequential(batchSize: number): Generator<{ x: Int32Array; y: Int32Array; count: number }> {
    const T = this.blockSize;
    let start = 0;
    while (start < this.numBlocks) {
      const count = Math.min(batchSize, Math.ceil((this.numBlocks - start) / T));
      const x = new Int32Array(count * T);
      const y = new Int32Array(count * T);
      for (let b = 0; b < count; b++) {
        const s = Math.min(start + b * T, this.numBlocks - 1);
        x.set(this.ids.subarray(s, s + T), b * T);
        y.set(this.ids.subarray(s + 1, s + T + 1), b * T);
      }
      yield { x, y, count };
      start += count * T;
    }
  }
}
