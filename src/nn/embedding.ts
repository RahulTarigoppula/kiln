/**
 * Lookup table `[numEmbeddings, dim]` indexed by i32 ids.
 */
import type { Device } from "../backend/types";
import type { Tensor } from "../tensor/tensor";
import { Module } from "./module";
import { normal } from "./init";

/** Options for {@link Embedding}. */
export interface EmbeddingOptions {
  /** Std of the normal init (default 0.02). */
  std?: number;
  device?: Device;
}

/** Embedding layer. */
export class Embedding extends Module {
  readonly numEmbeddings: number;
  readonly dim: number;
  weight: Tensor;

  constructor(numEmbeddings: number, dim: number, opts: EmbeddingOptions = {}) {
    super();
    this.numEmbeddings = numEmbeddings;
    this.dim = dim;
    this.weight = this.param("weight", normal([numEmbeddings, dim], 0, opts.std ?? 0.02, { device: opts.device }));
  }

  /** ids: i32 [...] → [..., dim]. */
  forward(ids: Tensor): Tensor {
    return this.weight.embedding(ids);
  }

  override extraRepr(): string {
    return `${this.numEmbeddings}, ${this.dim}`;
  }
}
