/**
 * AdamW with decoupled weight decay, bias-corrected (PyTorch semantics), using the backend's fused kernel
 * — one dispatch per parameter per step.
 */
import { Tensor } from "../tensor/tensor";
import type { Module } from "../nn/module";
import { contiguousRaw } from "../autograd/ops";
import { Optimizer, type ParamsInput } from "./optimizer";

/** AdamW hyper-parameters. */
export type AdamWHyper = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
};

/** AdamW optimizer. */
export class AdamW extends Optimizer<AdamWHyper> {
  constructor(params: ParamsInput<AdamWHyper>, opts: Partial<AdamWHyper> & { betas?: [number, number] } = {}) {
    super(params, {
      lr: opts.lr ?? 1e-3,
      beta1: opts.beta1 ?? opts.betas?.[0] ?? 0.9,
      beta2: opts.beta2 ?? opts.betas?.[1] ?? 0.999,
      eps: opts.eps ?? 1e-8,
      weightDecay: opts.weightDecay ?? 0.01,
    });
  }

  get type(): string {
    return "adamw";
  }

  protected update(p: Tensor, grad: Tensor, h: AdamWHyper): void {
    if (!p.isDense) throw new Error("AdamW: parameters must be dense tensors");
    const m = this.stateTensor(p, "m", () => Tensor.zeros(p.shape, { device: p.device }));
    const v = this.stateTensor(p, "v", () => Tensor.zeros(p.shape, { device: p.device }));
    const g = grad.isDense ? grad : contiguousRaw(grad);
    p.backend.adamw(p.buf, g.buf, m.buf, v.buf, p.numel, h.lr, h.beta1, h.beta2, h.eps, h.weightDecay, this.stepCount);
    if (g !== grad) g.free();
  }
}

/** Options for {@link configureAdamW}. */
export interface ConfigureAdamWOptions {
  lr?: number;
  weightDecay?: number;
  betas?: [number, number];
  eps?: number;
}

/**
 * nanoGPT-style parameter grouping: weight decay on every ≥2-D parameter (matmul weights, embeddings), none on
 * biases and LayerNorm gains. Returns the two groups (decay first).
 */
export function adamwParamGroups(model: Module, weightDecay: number): { params: Tensor[]; hyper: Partial<AdamWHyper> }[] {
  const decay: Tensor[] = [];
  const noDecay: Tensor[] = [];
  for (const p of model.parameters()) (p.ndim >= 2 ? decay : noDecay).push(p);
  const groups: { params: Tensor[]; hyper: Partial<AdamWHyper> }[] = [];
  if (decay.length) groups.push({ params: decay, hyper: { weightDecay } });
  if (noDecay.length) groups.push({ params: noDecay, hyper: { weightDecay: 0 } });
  return groups;
}

/** Build an AdamW over `model` with nanoGPT weight-decay grouping. */
export function configureAdamW(model: Module, opts: ConfigureAdamWOptions = {}): AdamW {
  const wd = opts.weightDecay ?? 0.1;
  return new AdamW(adamwParamGroups(model, wd), { lr: opts.lr ?? 6e-4, betas: opts.betas ?? [0.9, 0.95], eps: opts.eps ?? 1e-8, weightDecay: wd });
}
