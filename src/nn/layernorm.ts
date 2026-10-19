/**
 * Layer normalisation over the last dimension (fused forward/backward kernel).
 */
import type { Device } from "../backend/types";
import type { Tensor } from "../tensor/tensor";
import { Module } from "./module";
import { ones, zeros } from "./init";

/** Options for {@link LayerNorm}. */
export interface LayerNormOptions {
  /** Learnable bias (default true). Without it a constant zero beta feeds the fused kernel. */
  bias?: boolean;
  eps?: number;
  device?: Device;
}

/** LayerNorm with learnable gain (and optional bias). */
export class LayerNorm extends Module {
  readonly dim: number;
  readonly eps: number;
  weight: Tensor;
  bias: Tensor | null;
  /** Zero beta used when `bias: false` (non-persistent buffer). */
  private zeroBias: Tensor | null;

  constructor(dim: number, opts: LayerNormOptions = {}) {
    super();
    this.dim = dim;
    this.eps = opts.eps ?? 1e-5;
    this.weight = this.param("weight", ones([dim], { device: opts.device }));
    if (opts.bias ?? true) {
      this.bias = this.param("bias", zeros([dim], { device: opts.device }));
      this.zeroBias = null;
    } else {
      this.bias = null;
      this.zeroBias = this.buffer("zeroBias", zeros([dim], { device: opts.device }), { persistent: false });
    }
  }

  /** x: [..., dim] -> same shape. */
  forward(x: Tensor): Tensor {
    return x.layerNorm(this.weight, this.bias ?? (this.zeroBias as Tensor), this.eps);
  }

  override extraRepr(): string {
    return `${this.dim}, eps=${this.eps}, bias=${this.bias !== null}`;
  }
}
