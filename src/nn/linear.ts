/**
 * Fully connected layer `y = x·Wᵀ + b` with PyTorch's weight layout `[outFeatures, inFeatures]`.
 */
import type { Device } from "../backend/types";
import type { Tensor } from "../tensor/tensor";
import { Module } from "./module";
import { normal, zeros } from "./init";

/** Options for {@link Linear}. */
export interface LinearOptions {
  /** Add a bias vector (default true). */
  bias?: boolean;
  /** Std of the normal init (default 0.02, GPT-2). */
  std?: number;
  /**
   * Residual-projection scaling: when set, the weight std becomes `std / sqrt(2 · nLayer)` as in GPT-2
   * (`scaleInit` is the number of layers).
   */
  scaleInit?: number;
  device?: Device;
}

/** Linear layer. */
export class Linear extends Module {
  readonly inFeatures: number;
  readonly outFeatures: number;
  weight: Tensor;
  bias: Tensor | null;

  constructor(inFeatures: number, outFeatures: number, opts: LinearOptions = {}) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    const std = (opts.std ?? 0.02) / (opts.scaleInit ? Math.sqrt(2 * opts.scaleInit) : 1);
    this.weight = this.param("weight", normal([outFeatures, inFeatures], 0, std, { device: opts.device }));
    this.bias = (opts.bias ?? true) ? this.param("bias", zeros([outFeatures], { device: opts.device })) : null;
  }

  /** x: [..., inFeatures] -> [..., outFeatures]. */
  forward(x: Tensor): Tensor {
    const y = x.matmul(this.weight.transpose(0, 1));
    return this.bias ? y.add(this.bias) : y;
  }

  override extraRepr(): string {
    return `in=${this.inFeatures}, out=${this.outFeatures}, bias=${this.bias !== null}`;
  }
}
