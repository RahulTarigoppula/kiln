/**
 * Pre-LayerNorm transformer block: `x + attn(ln_1(x))`, then `x + mlp(ln_2(x))`.
 */
import type { Device } from "../backend/types";
import type { Tensor } from "../tensor/tensor";
import { Module } from "./module";
import { Linear } from "./linear";
import { LayerNorm } from "./layernorm";
import { Dropout } from "./dropout";
import { GELU } from "./activations";
import { CausalSelfAttention } from "./attention";

/** Options for {@link MLP} and {@link Block}. */
export interface BlockOptions {
  nEmbd: number;
  nHead: number;
  blockSize: number;
  dropout?: number;
  bias?: boolean;
  /** Number of layers (residual-projection init scaling). */
  nLayer?: number;
  device?: Device;
}

/** Feed-forward network: Linear(4×) → GELU → Linear → Dropout. */
export class MLP extends Module {
  c_fc: Linear;
  gelu: GELU;
  c_proj: Linear;
  dropout: Dropout;

  constructor(opts: BlockOptions) {
    super();
    const bias = opts.bias ?? true;
    this.c_fc = this.child("c_fc", new Linear(opts.nEmbd, 4 * opts.nEmbd, { bias, device: opts.device }));
    this.gelu = this.child("gelu", new GELU());
    this.c_proj = this.child("c_proj", new Linear(4 * opts.nEmbd, opts.nEmbd, { bias, scaleInit: opts.nLayer, device: opts.device }));
    this.dropout = this.child("dropout", new Dropout(opts.dropout ?? 0));
  }

  forward(x: Tensor): Tensor {
    return this.dropout.forward(this.c_proj.forward(this.gelu.forward(this.c_fc.forward(x))));
  }
}

/** One transformer layer. */
export class Block extends Module {
  ln_1: LayerNorm;
  attn: CausalSelfAttention;
  ln_2: LayerNorm;
  mlp: MLP;

  constructor(opts: BlockOptions) {
    super();
    const bias = opts.bias ?? true;
    this.ln_1 = this.child("ln_1", new LayerNorm(opts.nEmbd, { bias, device: opts.device }));
    this.attn = this.child("attn", new CausalSelfAttention({
      nEmbd: opts.nEmbd, nHead: opts.nHead, blockSize: opts.blockSize, dropout: opts.dropout, bias, nLayer: opts.nLayer, device: opts.device,
    }));
    this.ln_2 = this.child("ln_2", new LayerNorm(opts.nEmbd, { bias, device: opts.device }));
    this.mlp = this.child("mlp", new MLP(opts));
  }

  /** x: [B, T, nEmbd] → same shape. */
  forward(x: Tensor): Tensor {
    const h = x.add(this.attn.forward(this.ln_1.forward(x)));
    return h.add(this.mlp.forward(this.ln_2.forward(h)));
  }
}
