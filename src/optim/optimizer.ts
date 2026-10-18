/**
 * Optimizer base: parameter groups with per-group hyper-parameters, step counting and per-parameter state
 * tensors (exposed for checkpointing as `optim.<param name>.<key>` records).
 */
import type { Tensor } from "../tensor/tensor";
import { escape } from "../tensor/scope";
import { noGrad } from "../autograd/graph";

/** Hyper-parameters are plain numbers. */
export type Hyper = Record<string, number>;

/** A group of parameters sharing hyper-parameter overrides. */
export interface ParamGroup<H extends Hyper> {
  params: Tensor[];
  /** Overrides for this group (merged over the optimizer defaults). */
  hyper: Partial<H>;
}

/** Input accepted by optimizer constructors. */
export type ParamsInput<H extends Hyper> = Tensor[] | { params: Tensor[]; hyper?: Partial<H> }[];

/** Serialisable optimizer description (state tensors travel separately, see `stateTensors()`). */
export interface OptimizerStateDict<H extends Hyper> {
  type: string;
  step: number;
  defaults: H;
  groups: { size: number; hyper: Partial<H> }[];
}

/** Base class for optimizers. */
export abstract class Optimizer<H extends Hyper = Hyper> {
  readonly groups: ParamGroup<H>[];
  readonly defaults: H;
  /** Number of `step()` calls so far (1-based inside the first step). */
  stepCount = 0;
  /** Per-parameter state tensors (e.g. AdamW's `m`, `v`). */
  protected readonly state = new Map<Tensor, Record<string, Tensor>>();

  constructor(params: ParamsInput<H>, defaults: H) {
    this.defaults = { ...defaults };
    const first = params[0];
    if (first === undefined) {
      this.groups = [];
    } else if ("params" in first) {
      this.groups = (params as { params: Tensor[]; hyper?: Partial<H> }[]).map((g) => ({ params: [...g.params], hyper: { ...(g.hyper ?? {}) } as Partial<H> }));
    } else {
      this.groups = [{ params: [...(params as Tensor[])], hyper: {} }];
    }
    const seen = new Set<Tensor>();
    for (const g of this.groups) {
      for (const p of g.params) {
        if (seen.has(p)) throw new Error("Optimizer: a parameter appears in more than one group");
        seen.add(p);
      }
    }
  }

  /** Short type tag used in checkpoints. */
  abstract get type(): string;

  /** All parameters across groups. */
  get params(): Tensor[] {
    const out: Tensor[] = [];
    for (const g of this.groups) out.push(...g.params);
    return out;
  }

  /** Effective hyper-parameters of a group. */
  hyperOf(group: ParamGroup<H>): H {
    return { ...this.defaults, ...group.hyper } as H;
  }

  /** Learning rate of the first group. */
  get lr(): number {
    const g = this.groups[0];
    return g ? (this.hyperOf(g) as { lr?: number }).lr ?? (this.defaults as { lr?: number }).lr ?? 0 : (this.defaults as { lr?: number }).lr ?? 0;
  }

  /** Set the learning rate on every group (schedulers call this each step). */
  set lr(value: number) {
    (this.defaults as Hyper).lr = value;
    for (const g of this.groups) (g.hyper as Hyper).lr = value;
  }

  /** Perform one update for every parameter that has a gradient. */
  step(): void {
    this.stepCount++;
    noGrad(() => {
      for (const g of this.groups) {
        const hyper = this.hyperOf(g);
        for (const p of g.params) {
          if (!p.grad) continue;
          this.update(p, p.grad, hyper);
        }
      }
    });
  }

  /** Update one parameter in place. */
  protected abstract update(param: Tensor, grad: Tensor, hyper: H): void;

  /** Free every parameter gradient. */
  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }

  /** Lazily created state tensor for `param` (zeros shaped like it, escaped from scopes). */
  protected stateTensor(param: Tensor, key: string, make: () => Tensor): Tensor {
    let s = this.state.get(param);
    if (!s) {
      s = {};
      this.state.set(param, s);
    }
    let t = s[key];
    if (!t) {
      t = escape(make());
      s[key] = t;
    }
    return t;
  }

  /** Per-parameter state tensors (for checkpointing). */
  stateTensors(): Map<Tensor, Record<string, Tensor>> {
    return new Map([...this.state].map(([p, s]) => [p, { ...s }]));
  }

  /** Install state tensors for `param` (copied in place into fresh state on the parameter's device). */
  loadParamState(param: Tensor, tensors: Record<string, Tensor>): void {
    for (const [key, src] of Object.entries(tensors)) {
      const dst = this.stateTensor(param, key, () => param.detach().clone().zero_());
      if (dst.numel !== src.numel) throw new Error(`Optimizer.loadParamState: "${key}" has ${src.numel} elements, expected ${dst.numel}`);
      dst.copy_(src.device === dst.device ? src : src.to(dst.device));
    }
  }

  /** Hyper-parameters and step count (state tensors are separate, see `stateTensors()`). */
  stateDict(): OptimizerStateDict<H> {
    return {
      type: this.type,
      step: this.stepCount,
      defaults: { ...this.defaults },
      groups: this.groups.map((g) => ({ size: g.params.length, hyper: { ...g.hyper } })),
    };
  }

  /** Restore hyper-parameters and step count from `stateDict()` output. */
  loadStateDict(sd: OptimizerStateDict<H>): void {
    if (sd.type !== this.type) throw new Error(`Optimizer.loadStateDict: type mismatch (${sd.type} vs ${this.type})`);
    if (sd.groups.length !== this.groups.length) throw new Error("Optimizer.loadStateDict: group count mismatch");
    this.stepCount = sd.step;
    Object.assign(this.defaults, sd.defaults);
    sd.groups.forEach((g, i) => {
      const mine = this.groups[i] as ParamGroup<H>;
      if (g.size !== mine.params.length) throw new Error(`Optimizer.loadStateDict: group ${i} has ${mine.params.length} params, checkpoint has ${g.size}`);
      mine.hyper = { ...g.hyper };
    });
  }
}
