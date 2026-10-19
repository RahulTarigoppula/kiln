/**
 * Autograd graph: grad-mode flag, `Node` (a recorded backward function) and the reverse-mode engine.
 *
 * Memory: `backward()` frees what it can as it goes - each node's owned `saved` tensors (via `release()`),
 * temporaries created inside backward closures (an internal scope per node), and the gradients of non-leaf
 * tensors once they have been propagated. Gradients stored on leaves are `escape`d from any enclosing
 * `tidy` scope: the leaf owns them (`zeroGrad()` frees them, accumulation frees the replaced grad).
 */
import { Tensor, type GradFn } from "../tensor/tensor";
import { shapesEqual } from "../tensor/shape";
import { beginScope, endScope, escape } from "../tensor/scope";
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
  /** Fresh tensors the op created for backward; owned by the node and freed by `release()`. */
  saved: Tensor[];
  released = false;
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

  /** Free the owned saved tensors and drop the backward closure so buffers can be reclaimed. */
  release(): void {
    this.fn = null;
    this.released = true;
    for (const s of this.saved) s.free();
    this.saved = [];
  }
}

/** True when grad mode is on and any input requires grad. */
export function needsGrad(inputs: readonly Tensor[]): boolean {
  if (!gradEnabled) return false;
  for (let i = 0; i < inputs.length; i++) if ((inputs[i] as Tensor).requiresGrad) return true;
  return false;
}

/**
 * Attach a backward node to `out` if recording is needed; returns `out`. `saved` lists *fresh* tensors the op
 * made for backward (never its inputs or output): they are freed when the node is released.
 */
export function record(out: Tensor, name: string, inputs: readonly Tensor[], fn: BackwardFn, saved: Tensor[] = []): Tensor {
  if (!needsGrad(inputs)) {
    for (const s of saved) s.free();
    return out;
  }
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

/** Store `g` as `t.grad` (accumulating), taking ownership: the stored tensor is escaped from scopes. */
function storeGrad(t: Tensor, g: Tensor, claimed: Set<Tensor>): void {
  let own = g;
  if (claimed.has(g) || !g.isDense) own = g.isDense ? cloneRaw(g) : contiguousRaw(g);
  claimed.add(own);
  if (t.grad) {
    const sum = add(t.grad, own);
    t.grad.free();
    t.grad = escape(sum);
  } else {
    t.grad = escape(own);
  }
}

/**
 * Reverse-mode backward from `root`. Gradients are accumulated into leaf tensors' `.grad`
 * (and into non-leaves that called `retainGrad()`); saved tensors and intermediate grads are freed as nodes
 * are consumed.
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
    /** Tensors that must not be freed by the engine: the caller's grad and everything stored as `.grad`. */
    const claimed = new Set<Tensor>();
    /** How many `grads` entries reference each gradient tensor object (backward fns may alias outputs). */
    const refs = new Map<Tensor, number>();
    const incr = (t: Tensor): void => { refs.set(t, (refs.get(t) ?? 0) + 1); };
    const decr = (t: Tensor): void => {
      const n = (refs.get(t) ?? 0) - 1;
      if (n > 0) {
        refs.set(t, n);
        return;
      }
      refs.delete(t);
      if (!claimed.has(t)) t.free();
    };
    if (grad) claimed.add(grad);
    grads.set(root, g0);
    incr(g0);

    for (let i = order.length - 1; i >= 0; i--) {
      const t = order[i] as Tensor;
      const g = grads.get(t);
      if (!g) continue;
      const fn = t.gradFn;
      if (fn) {
        // Temporaries created by the closure (dense copies, products, ...) die with this scope; returned
        // gradients survive because they are the scope's result.
        beginScope();
        let inGrads: (Tensor | null)[];
        try {
          inGrads = fn.backward(g);
        } catch (e) {
          endScope(undefined);
          throw e;
        }
        endScope(inGrads);
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
          incr(ig);
        }
        for (let k = 0; k < inputs.length; k++) {
          const inp = inputs[k] as Tensor;
          const ig = inGrads[k];
          if (!ig || !inp.requiresGrad) continue;
          const prev = grads.get(inp);
          if (prev) {
            const sum = add(prev, ig);
            grads.set(inp, sum);
            incr(sum);
            decr(prev);
            decr(ig);
          } else {
            grads.set(inp, ig);
          }
        }
        fn.release();
        if (t.retainsGrad) storeGrad(t, g, claimed);
      } else {
        storeGrad(t, g, claimed);
      }
      grads.delete(t);
      decr(g);
    }
  });
}
