/**
 * Memory scopes ("tidy"): automatic freeing of intermediate tensors.
 *
 * Kiln has no garbage collector for device buffers — `Storage` is refcounted and a buffer is returned to the
 * backend only when every `Tensor` handle referencing it has called `free()`. Ops create many intermediates
 * (dense copies, mask tensors, saved activations, gradient temporaries), so a training step would leak GPU
 * memory without help. A scope fixes that:
 *
 *   const loss = tidy(() => { const { loss } = model.forward(x, y); loss.backward(); return loss; });
 *
 * Every Tensor constructed while `fn` runs is *tracked* by the innermost scope. When `fn` returns, the scope
 * frees all tracked tensors except
 *   - tensors returned from `fn` (a Tensor, or any array/object nesting of Tensors),
 *   - tensors explicitly marked with `keep(t)`,
 *   - tensors reachable from a kept tensor's (unreleased) autograd graph — so `tidy(() => model.forward(x))`
 *     followed by `backward()` outside the scope is still safe,
 *   - tensors removed from scope tracking with `escape(t)`: parameters, buffers, optimizer state and
 *     gradients stored on leaves are escaped automatically by `Module`, the optimisers and `backward()`.
 * Kept tensors are handed to the enclosing scope (if any), TF.js style, so nested scopes compose.
 *
 * Scopes are synchronous: `fn` must not return a Promise (an `await` inside would interleave unrelated
 * allocations into the scope). Use several short `tidy` calls around the awaits instead.
 */
import type { Tensor } from "./tensor";

interface Scope {
  tracked: Set<Tensor>;
  kept: Set<Tensor>;
}

const stack: Scope[] = [];

/** Record a freshly constructed tensor in the innermost scope (called by the Tensor constructor). */
export function trackTensor(t: Tensor): void {
  const top = stack[stack.length - 1];
  if (top) top.tracked.add(t);
}

/** True when a `tidy` scope is active. */
export function inScope(): boolean {
  return stack.length > 0;
}

/** Depth of the scope stack (for tests). */
export function scopeDepth(): number {
  return stack.length;
}

/** Keep `t` alive past the end of the current scope (it is handed to the enclosing scope). */
export function keep<T extends Tensor>(t: T): T {
  const top = stack[stack.length - 1];
  if (top) top.kept.add(t);
  return t;
}

/** Remove `t` from every scope so it is never freed automatically (parameters, grads, caches). */
export function escape<T extends Tensor>(t: T): T {
  for (let i = stack.length - 1; i >= 0; i--) {
    const s = stack[i] as Scope;
    s.tracked.delete(t);
    s.kept.delete(t);
  }
  return t;
}

/** Collect every Tensor inside a return value (Tensor, arrays and plain objects, nested). */
function collect(value: unknown, out: Set<Tensor>, depth: number): void {
  if (value === null || value === undefined || depth > 4) return;
  if (isTensor(value)) {
    out.add(value);
    return;
  }
  if (Array.isArray(value)) {
    for (const v of value) collect(v, out, depth + 1);
    return;
  }
  if (typeof value === "object") {
    if (typeof (value as { then?: unknown }).then === "function") {
      throw new Error("tidy(): the callback returned a Promise; scopes are synchronous — wrap the sync parts in tidy() separately");
    }
    const proto = Object.getPrototypeOf(value) as unknown;
    if (proto === Object.prototype || proto === null) {
      for (const v of Object.values(value as Record<string, unknown>)) collect(v, out, depth + 1);
    } else if (value instanceof Map) {
      for (const v of value.values()) collect(v, out, depth + 1);
    } else if (value instanceof Set) {
      for (const v of value) collect(v, out, depth + 1);
    }
  }
}

function isTensor(x: unknown): x is Tensor {
  return typeof x === "object" && x !== null && "storage" in x && "gradFn" in x && typeof (x as Tensor).free === "function";
}

/** Mark everything reachable from `roots` through unreleased autograd nodes as kept. */
function keepGraph(roots: Iterable<Tensor>, tracked: Set<Tensor>, kept: Set<Tensor>): void {
  const todo: Tensor[] = [];
  const seen = new Set<Tensor>();
  for (const r of roots) {
    todo.push(r);
    seen.add(r);
  }
  while (todo.length > 0) {
    const t = todo.pop() as Tensor;
    if (tracked.has(t)) kept.add(t);
    const fn = t.gradFn;
    if (!fn || fn.released) continue;
    for (const inp of fn.inputs) {
      if (!seen.has(inp)) {
        seen.add(inp);
        todo.push(inp);
      }
    }
    for (const s of fn.saved) {
      if (!seen.has(s)) {
        seen.add(s);
        todo.push(s);
      }
    }
  }
}

/** Start a scope manually (prefer `tidy`). Pair with `endScope`. */
export function beginScope(): void {
  stack.push({ tracked: new Set(), kept: new Set() });
}

/**
 * End the innermost scope: tensors in `result` and kept tensors (plus their live autograd graph) survive and
 * move to the parent scope; everything else tracked is freed. Returns the number of tensors freed.
 */
export function endScope(result?: unknown): number {
  const scope = stack.pop();
  if (!scope) throw new Error("endScope() without a matching beginScope()");
  const roots = new Set<Tensor>(scope.kept);
  collect(result, roots, 0);
  const kept = new Set<Tensor>();
  keepGraph(roots, scope.tracked, kept);
  for (const r of roots) kept.add(r);
  let freed = 0;
  for (const t of scope.tracked) {
    if (kept.has(t)) continue;
    t.free();
    freed++;
  }
  const parent = stack[stack.length - 1];
  if (parent) {
    for (const t of kept) if (scope.tracked.has(t)) parent.tracked.add(t);
  }
  return freed;
}

/** Run `fn` in a memory scope: every tensor it creates and does not return (or `keep`) is freed afterwards. */
export function tidy<T>(fn: () => T): T {
  beginScope();
  let result: T;
  try {
    result = fn();
  } catch (e) {
    endScope(undefined);
    throw e;
  }
  endScope(result);
  return result;
}

/** Number of tensors currently tracked by the innermost scope (for tests/debugging). */
export function trackedCount(): number {
  const top = stack[stack.length - 1];
  return top ? top.tracked.size : 0;
}
