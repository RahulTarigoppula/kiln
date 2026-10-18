/**
 * Stochastic gradient descent with momentum, Nesterov and (coupled) weight decay — PyTorch semantics.
 */
import type { Tensor } from "../tensor/tensor";
import { tidy } from "../tensor/scope";
import { Optimizer, type ParamsInput } from "./optimizer";

/** SGD hyper-parameters. */
export type SGDHyper = {
  lr: number;
  momentum: number;
  /** L2 penalty added to the gradient (PyTorch style). */
  weightDecay: number;
  /** 1 = Nesterov momentum, 0 = classic (numbers so groups stay serialisable). */
  nesterov: number;
};

/** SGD optimizer. */
export class SGD extends Optimizer<SGDHyper> {
  constructor(params: ParamsInput<SGDHyper>, opts: Partial<SGDHyper> = {}) {
    super(params, {
      lr: opts.lr ?? 1e-2,
      momentum: opts.momentum ?? 0,
      weightDecay: opts.weightDecay ?? 0,
      nesterov: opts.nesterov ?? 0,
    });
  }

  get type(): string {
    return "sgd";
  }

  protected update(p: Tensor, grad: Tensor, h: SGDHyper): void {
    tidy(() => {
      let g = grad;
      if (h.weightDecay !== 0) g = g.add(p.mul(h.weightDecay));
      if (h.momentum !== 0) {
        const existing = this.state.get(p)?.["momentum"] !== undefined;
        const buf = this.stateTensor(p, "momentum", () => p.detach().clone().zero_());
        const newBuf = existing ? buf.mul(h.momentum).add(g) : g; // PyTorch: first step buf = g
        buf.copy_(newBuf);
        g = h.nesterov ? g.add(newBuf.mul(h.momentum)) : newBuf;
      }
      p.copy_(p.sub(g.mul(h.lr)));
    });
  }
}
