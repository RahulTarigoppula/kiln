/**
 * Inverted dropout; identity in eval mode or when p = 0.
 */
import type { Tensor } from "../tensor/tensor";
import { Module } from "./module";

/** Dropout layer. */
export class Dropout extends Module {
  readonly p: number;

  constructor(p: number) {
    super();
    if (p < 0 || p >= 1) throw new Error(`Dropout: p must be in [0, 1), got ${p}`);
    this.p = p;
  }

  forward(x: Tensor): Tensor {
    return x.dropout(this.p, this.training);
  }

  override extraRepr(): string {
    return `p=${this.p}`;
  }
}
