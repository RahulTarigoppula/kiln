/**
 * Autograd graph: grad-mode flag, `Node` (a recorded backward function) and the reverse-mode engine.
 */
import { Tensor, type GradFn } from "../tensor/tensor";
import { shapesEqual } from "../tensor/shape";
import { add, cloneRaw, contiguousRaw } from "./ops";

let gradEnabled = true;

/** True when ops record autograd history. */
export function isGradEnabled(): boolean {
  return gradEnabled;
}

/** Set grad mode; returns the previous value. */
export function setGradEnabled(enabled: boolean): boolean {
  const prev = gradEnabled;
  gradEnabled = enabled;
  return prev;
}

/** Run `fn` with autograd recording disabled. */
export function noGrad<T>(fn: () => T): T {
  const prev = setGradEnabled(false);
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

/** Run `fn` with autograd recording enabled (inside a noGrad scope). */
export function enableGrad<T>(fn: () => T): T {
  const prev = setGradEnabled(true);
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

/** Backward closure: maps the output gradient to one gradient (or null) per input. */
export type BackwardFn = (gradOut: Tensor) => (Tensor | null)[];

/** A node in the autograd graph: the inputs of an op and how to propagate a gradient through it. */
export class Node implements GradFn {
  readonly name: string;
  readonly inputs: readonly Tensor[];
  /** Tensors kept alive for backward; cleared by `release()`. */
  saved: Tensor[];
  private fn: BackwardFn | null;

  constructor(name: string, inputs: readonly Tensor[], fn: BackwardFn, saved: Tensor[] = []) {
    this.name = name;
    this.inputs = inputs;
    this.fn = fn;
    this.saved = saved;
  }

  backward(gradOut: Tensor): (Tensor | null)[] {
    if (!this.fn) {
      throw new Error(`Trying to backward through the graph a second time (node ${this.name} was already released)`);
    }
    return this.fn(gradOut);
  }

  /** Drop saved tensors and the backward closure so their buffers can be reclaimed. */
  release(): void {
    this.fn = null;
    this.saved = [];
  }
}

/** True when grad mode is on and any input requires grad. */
export function needsGrad(inputs: readonly Tensor[]): boolean {
  if (!gradEnabled) return false;
  for (let i = 0; i < inputs.length; i++) if ((inputs[i] as Tensor).requiresGrad) return true;
  return false;
}

/** Attach a backward node to `out` if recording is needed; returns `out`. */
export function record(out: Tensor, name: string, inputs: readonly Tensor[], fn: BackwardFn, saved: Tensor[] = []): Tensor {
  if (!needsGrad(inputs)) return out;
  out.gradFn = new Node(name, inputs, fn, saved);
  out.requiresGrad = true;
  return out;
}

/** Topological order (inputs before consumers) of the requires-grad subgraph rooted at `root`. */
export function topologicalOrder(root: Tensor): Tensor[] {
  const order: Tensor[] = [];
  const visited = new Set<Tensor>([root]);
  const stackT: Tensor[] = [root];
  const stackI: number[] = [0];
  while (stackT.length > 0) {
    const top = stackT.length - 1;
    const t = stackT[top] as Tensor;
    const inputs = t.gradFn ? t.gradFn.inputs : [];
    const i = stackI[top] as number;
    if (i < inputs.length) {
      stackI[top] = i + 1;
      const next = inputs[i] as Tensor;
      if (next.requiresGrad && !visited.has(next)) {
        visited.add(next);
        stackT.push(next);
        stackI.push(0);
      }
    } else {
      order.push(t);
      stackT.pop();
      stackI.pop();
    }
  }
  return order;
}

/**
 * Reverse-mode backward from `root`. Gradients are accumulated into leaf tensors' `.grad`
 * (and into non-leaves that called `retainGrad()`); saved tensors are released as nodes are consumed.
 */
export function backward(root: Tensor, grad?: Tensor): void {
  if (!root.requiresGrad) throw new Error("backward() called on a tensor that does not require grad");
  let g0: Tensor;
  if (grad) {
    if (!shapesEqual(grad.shape, root.shape)) {
      throw new Error(`backward: grad shape [${grad.shape.join(",")}] does not match [${root.shape.join(",")}]`);
    }
    g0 = grad;
  } else {
    if (root.numel !== 1) throw new Error("grad can be implicitly created only for scalar outputs");
    g0 = Tensor.ones(root.shape, { device: root.device });
  }
  noGrad(() => {
    const order = topologicalOrder(root);
    const grads = new Map<Tensor, Tensor>();
    const claimed = new Set<Tensor>();
    if (grad) claimed.add(grad);
    grads.set(root, g0);

    const store = (t: Tensor, g: Tensor): void => {
      let own = g;
      if (claimed.has(g) || !g.isDense) own = g.isDense ? cloneRaw(g) : contiguousRaw(g);
      claimed.add(own);
      t.grad = t.grad ? add(t.grad, own) : own;
    };

    for (let i = order.length - 1; i >= 0; i--) {
      const t = order[i] as Tensor;
      const g = grads.get(t);
      if (!g) continue;
      const fn = t.gradFn;
      if (fn) {
        const inGrads = fn.backward(g);
        const inputs = fn.inputs;
        if (inGrads.length !== inputs.length) {
          throw new Error(`Node ${fn.name} returned ${inGrads.length} gradients for ${inputs.length} inputs`);
        }
        for (let k = 0; k < inputs.length; k++) {
          const inp = inputs[k] as Tensor;
          const ig = inGrads[k];
          if (!ig || !inp.requiresGrad) continue;
          if (!shapesEqual(ig.shape, inp.shape)) {
            throw new Error(`Node ${fn.name}: gradient shape [${ig.shape.join(",")}] != input shape [${inp.shape.join(",")}]`);
          }
          const prev = grads.get(inp);
          grads.set(inp, prev ? add(prev, ig) : ig);
        }
        fn.release();
        if (t.retainsGrad) store(t, g);
      } else {
        store(t, g);
      }
      grads.delete(t);
    }
  });
}
