/**
 * Multi-head causal self-attention (nanoGPT layout): one fused qkv projection, per-head scaled dot-product
 * attention with a cached causal mask, attention/residual dropout and an output projection.
 */
import type { Device } from "../backend/types";
import { Tensor } from "../tensor/tensor";
import { escape } from "../tensor/scope";
import { Module } from "./module";
import { Linear } from "./linear";
import { Dropout } from "./dropout";

/**
 * Value written into masked (future) attention logits before the softmax. `-Infinity` gives exact zeros
 * after softmax on both backends (`exp(-inf - max) = 0`); the diagonal is never masked so no row is all -inf.
 * Switch to a large finite negative (e.g. -1e9) if a backend ever produces NaN from `-inf` arithmetic.
 */
export const ATTENTION_MASK_VALUE = -Infinity;

/** Options for {@link CausalSelfAttention}. */
export interface CausalSelfAttentionOptions {
  nEmbd: number;
  nHead: number;
  /** Maximum sequence length (size of the cached causal mask). */
  blockSize: number;
  dropout?: number;
  /** Biases on the projections (default true). */
  bias?: boolean;
  /** Number of layers, for GPT-2's residual-projection init scaling. */
  nLayer?: number;
  device?: Device;
}

/** Causal self-attention block. */
export class CausalSelfAttention extends Module {
  readonly nEmbd: number;
  readonly nHead: number;
  readonly headDim: number;
  readonly blockSize: number;
  readonly dropout: number;
  /** Fused query/key/value projection `[nEmbd -> 3·nEmbd]`. */
  c_attn: Linear;
  /** Output projection. */
  c_proj: Linear;
  attn_dropout: Dropout;
  resid_dropout: Dropout;
  /** Non-persistent buffer: 1 above the diagonal (positions to mask), shape [1, 1, blockSize, blockSize]. */
  private mask: Tensor;
  /** When true, `forward` stores the last softmax output (detached) in {@link lastAttention}. */
  captureAttention = false;
  /** Last attention weights [B, nHead, T, T] when `captureAttention` is on. */
  lastAttention: Tensor | null = null;

  constructor(opts: CausalSelfAttentionOptions) {
    super();
    if (opts.nEmbd % opts.nHead !== 0) throw new Error(`CausalSelfAttention: nEmbd ${opts.nEmbd} not divisible by nHead ${opts.nHead}`);
    this.nEmbd = opts.nEmbd;
    this.nHead = opts.nHead;
    this.headDim = opts.nEmbd / opts.nHead;
    this.blockSize = opts.blockSize;
    this.dropout = opts.dropout ?? 0;
    const bias = opts.bias ?? true;
    this.c_attn = this.child("c_attn", new Linear(opts.nEmbd, 3 * opts.nEmbd, { bias, device: opts.device }));
    this.c_proj = this.child("c_proj", new Linear(opts.nEmbd, opts.nEmbd, { bias, scaleInit: opts.nLayer, device: opts.device }));
    this.attn_dropout = this.child("attn_dropout", new Dropout(this.dropout));
    this.resid_dropout = this.child("resid_dropout", new Dropout(this.dropout));
    this.mask = this.buffer("mask", causalMask(opts.blockSize, opts.device ?? "cpu"), { persistent: false });
  }

  protected override onTensorMoved(from: Tensor, to: Tensor): void {
    if (this.mask === from) this.mask = to;
  }

  /** x: [B, T, nEmbd] -> [B, T, nEmbd]. */
  forward(x: Tensor): Tensor {
    const [B, T, C] = x.shape as [number, number, number];
    if (T > this.blockSize) throw new Error(`CausalSelfAttention: sequence length ${T} exceeds blockSize ${this.blockSize}`);
    const nh = this.nHead;
    const hd = this.headDim;
    const qkv = this.c_attn.forward(x); // [B, T, 3C]
    const q = qkv.slice(2, 0, C).reshape(B, T, nh, hd).transpose(1, 2); // [B, nh, T, hd]
    const k = qkv.slice(2, C, 2 * C).reshape(B, T, nh, hd).transpose(1, 2);
    const v = qkv.slice(2, 2 * C, 3 * C).reshape(B, T, nh, hd).transpose(1, 2);

    let att = q.matmul(k.transpose(-2, -1)).mul(1 / Math.sqrt(hd)); // [B, nh, T, T]
    const maskT = this.mask.slice(2, 0, T).slice(3, 0, T);
    att = att.maskedFill(maskT, ATTENTION_MASK_VALUE).softmax(-1);
    if (this.captureAttention) {
      if (this.lastAttention) this.lastAttention.free();
      this.lastAttention = escape(att.detach());
    }
    att = this.attn_dropout.forward(att);
    const y = att.matmul(v).transpose(1, 2).reshape(B, T, C); // merge heads
    return this.resid_dropout.forward(this.c_proj.forward(y));
  }

  override extraRepr(): string {
    return `nEmbd=${this.nEmbd}, nHead=${this.nHead}, blockSize=${this.blockSize}, dropout=${this.dropout}`;
  }
}

/** f32 [1, 1, n, n] with 1 strictly above the diagonal. */
export function causalMask(n: number, device: Device): Tensor {
  const data = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) data[i * n + j] = 1;
  return Tensor.fromArray(data, [1, 1, n, n], { device: "cpu" }).to(device);
}
