/**
 * Chain of single-input, single-output modules.
 */
import type { Tensor } from "../tensor/tensor";
import { Module } from "./module";

/** A module with a `forward(x: Tensor): Tensor`. */
export interface UnaryModule extends Module {
  forward(x: Tensor): Tensor;
}

/** Applies its layers in order; children are named "0", "1", ... */
export class Sequential extends Module {
  readonly layers: UnaryModule[];

  constructor(...layers: UnaryModule[]) {
    super();
    this.layers = layers;
    layers.forEach((m, i) => this.child(String(i), m));
  }

  forward(x: Tensor): Tensor {
    let h = x;
    for (const layer of this.layers) h = layer.forward(h);
    return h;
  }

  get length(): number {
    return this.layers.length;
  }
}
