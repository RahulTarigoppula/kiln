/**
 * `Module`: the base class for neural-network layers (PyTorch-style).
 *
 * Parameters, buffers and child modules are registered explicitly (`this.param()`, `this.buffer()`,
 * `this.child()`) *or* discovered automatically from own properties: a property holding a `Module`, an array
 * of Modules, or a Tensor with `requiresGrad` is picked up by `namedParameters()`/`namedChildren()` in
 * property order. Names are dotted like nanoGPT: `transformer.h.0.attn.c_attn.weight`.
 *
 * Registered parameters and buffers are `escape`d from memory scopes so a `tidy` never frees them.
 */
import type { Device } from "../backend/types";
import { Tensor } from "../tensor/tensor";
import { escape } from "../tensor/scope";

/** A state dict: dotted name -> tensor (parameters first, then persistent buffers). */
export type StateDict = Map<string, Tensor>;

/** Result of {@link Module.loadStateDict}. */
export interface LoadResult {
  missing: string[];
  unexpected: string[];
}

interface BufferEntry {
  tensor: Tensor;
  persistent: boolean;
}

/** Base class for all layers. */
export class Module {
  /** Training-mode flag (dropout etc. read it). */
  training = true;
  private readonly _params = new Map<string, Tensor>();
  private readonly _buffers = new Map<string, BufferEntry>();
  private readonly _children = new Map<string, Module>();

  // ------------------------------------------------------------------ registration

  /** Register `t` as a learnable parameter named `name`; returns it (with `requiresGrad = true`). */
  protected param(name: string, t: Tensor): Tensor {
    t.requiresGrad = true;
    this._params.set(name, escape(t));
    return t;
  }

  /** Register a non-learnable tensor. Non-persistent buffers are moved by `to()` but left out of `stateDict()`. */
  protected buffer(name: string, t: Tensor, opts: { persistent?: boolean } = {}): Tensor {
    t.requiresGrad = false;
    this._buffers.set(name, { tensor: escape(t), persistent: opts.persistent ?? true });
    return t;
  }

  /** Register a child module under `name`; returns it. */
  protected child<M extends Module>(name: string, m: M): M {
    this._children.set(name, m);
    return m;
  }

  /** Public form of `param()` for wiring done outside a constructor (weight tying, tests). */
  registerParameter(name: string, t: Tensor): Tensor {
    return this.param(name, t);
  }

  /** Public form of `buffer()`. */
  registerBuffer(name: string, t: Tensor, opts: { persistent?: boolean } = {}): Tensor {
    return this.buffer(name, t, opts);
  }

  /** Public form of `child()`. */
  registerModule<M extends Module>(name: string, m: M): M {
    return this.child(name, m);
  }

  // ------------------------------------------------------------------ enumeration

  /** Own parameters (name -> tensor): registered ones, then auto-discovered requires-grad tensor properties. */
  ownParameters(): Map<string, Tensor> {
    const out = new Map<string, Tensor>(this._params);
    const seen = new Set(out.values());
    for (const [key, value] of Object.entries(this)) {
      if (value instanceof Tensor && value.requiresGrad && !seen.has(value) && !out.has(key)) {
        out.set(key, value);
        seen.add(value);
      }
    }
    return out;
  }

  /** Own buffers (name -> tensor). */
  ownBuffers(persistentOnly = false): Map<string, Tensor> {
    const out = new Map<string, Tensor>();
    for (const [k, e] of this._buffers) if (!persistentOnly || e.persistent) out.set(k, e.tensor);
    return out;
  }

  /** Direct children (name -> module): registered ones, then Module / Module[] properties. */
  namedChildren(): Map<string, Module> {
    const out = new Map<string, Module>(this._children);
    const seen = new Set(out.values());
    for (const [key, value] of Object.entries(this)) {
      if (value instanceof Module) {
        if (!seen.has(value) && !out.has(key)) {
          out.set(key, value);
          seen.add(value);
        }
      } else if (Array.isArray(value) && value.length > 0 && value.every((v) => v instanceof Module)) {
        (value as Module[]).forEach((m, i) => {
          const name = `${key}.${i}`;
          if (!seen.has(m) && !out.has(name)) {
            out.set(name, m);
            seen.add(m);
          }
        });
      }
    }
    return out;
  }

  /** Children as a list. */
  children(): Module[] {
    return [...this.namedChildren().values()];
  }

  /** This module and every descendant with dotted names (`""` for this module), pre-order. */
  namedModules(prefix = ""): [string, Module][] {
    const out: [string, Module][] = [[prefix, this]];
    for (const [name, m] of this.namedChildren()) {
      out.push(...m.namedModules(prefix ? `${prefix}.${name}` : name));
    }
    return out;
  }

  /** All modules in the tree, pre-order. */
  modules(): Module[] {
    return this.namedModules().map(([, m]) => m);
  }

  /** Dotted parameter names -> tensors over the whole tree, each tensor listed once (first name wins). */
  namedParameters(): [string, Tensor][] {
    const out: [string, Tensor][] = [];
    const seen = new Set<Tensor>();
    for (const [prefix, m] of this.namedModules()) {
      for (const [name, t] of m.ownParameters()) {
        if (seen.has(t)) continue;
        seen.add(t);
        out.push([prefix ? `${prefix}.${name}` : name, t]);
      }
    }
    return out;
  }

  /** Unique parameter tensors over the whole tree. */
  parameters(): Tensor[] {
    return this.namedParameters().map(([, t]) => t);
  }

  /** Dotted buffer names -> tensors over the whole tree. */
  namedBuffers(persistentOnly = false): [string, Tensor][] {
    const out: [string, Tensor][] = [];
    for (const [prefix, m] of this.namedModules()) {
      for (const [name, t] of m.ownBuffers(persistentOnly)) out.push([prefix ? `${prefix}.${name}` : name, t]);
    }
    return out;
  }

  /** All buffer tensors. */
  buffers(): Tensor[] {
    return this.namedBuffers().map(([, t]) => t);
  }

  /** Total number of parameter elements (shared tensors counted once). */
  numParameters(): number {
    let n = 0;
    for (const p of this.parameters()) n += p.numel;
    return n;
  }

  // ------------------------------------------------------------------ state

  /**
   * Parameters and persistent buffers by dotted name. Tied parameters appear under every name (PyTorch
   * semantics) - `io.ts` deduplicates them when writing safetensors.
   */
  stateDict(): StateDict {
    const sd: StateDict = new Map();
    for (const [prefix, m] of this.namedModules()) {
      for (const [name, t] of m.ownParameters()) sd.set(prefix ? `${prefix}.${name}` : name, t);
    }
    for (const [name, t] of this.namedBuffers(true)) sd.set(name, t);
    return sd;
  }

  /**
   * Copy tensors from `source` into this module's parameters/buffers (in place, shapes must match).
   * In strict mode a missing or unexpected key throws - except a tied parameter whose tensor was already
   * loaded under another name.
   */
  loadStateDict(source: StateDict | Record<string, Tensor>, strict = true): LoadResult {
    const src: StateDict = source instanceof Map ? source : new Map(Object.entries(source));
    const own = this.stateDict();
    const missing: string[] = [];
    const loaded = new Set<Tensor>();
    for (const [name, target] of own) {
      const value = src.get(name);
      if (!value) {
        if (!loaded.has(target)) missing.push(name);
        continue;
      }
      if (value.numel !== target.numel || value.shape.length !== target.shape.length || value.shape.some((d, i) => d !== target.shape[i])) {
        throw new Error(`loadStateDict: shape mismatch for "${name}": expected [${target.shape.join(",")}], got [${value.shape.join(",")}]`);
      }
      target.copy_(value.device === target.device ? value : value.to(target.device));
      loaded.add(target);
    }
    const unexpected = [...src.keys()].filter((k) => !own.has(k));
    if (strict && (missing.length || unexpected.length)) {
      const parts: string[] = [];
      if (missing.length) parts.push(`missing keys: ${missing.join(", ")}`);
      if (unexpected.length) parts.push(`unexpected keys: ${unexpected.join(", ")}`);
      throw new Error(`loadStateDict: ${parts.join("; ")}`);
    }
    return { missing, unexpected };
  }

  // ------------------------------------------------------------------ mode / device / grads

  /** Set training mode recursively. */
  train(mode = true): this {
    for (const m of this.modules()) m.training = mode;
    return this;
  }

  /** Set evaluation mode recursively. */
  eval(): this {
    return this.train(false);
  }

  /**
   * Move every parameter and buffer to `device` (CPU -> device only; returns `this`). Tied tensors stay tied.
   * Properties holding a moved tensor are rebound so `this.weight` keeps working.
   */
  to(device: Device): this {
    const memo = new Map<Tensor, Tensor>();
    const move = (t: Tensor): Tensor => {
      if (t.device === device) return t;
      let moved = memo.get(t);
      if (!moved) {
        moved = escape(t.to(device));
        moved.requiresGrad = t.requiresGrad;
        memo.set(t, moved);
      }
      return moved;
    };
    for (const m of this.modules()) m.moveOwn(move);
    return this;
  }

  private moveOwn(move: (t: Tensor) => Tensor): void {
    const rebind = (from: Tensor, to: Tensor): void => {
      if (from === to) return;
      for (const [key, value] of Object.entries(this)) {
        if (value === from) (this as unknown as Record<string, unknown>)[key] = to;
      }
      this.onTensorMoved(from, to);
    };
    for (const [name, t] of this.ownParameters()) {
      const moved = move(t);
      if (this._params.has(name)) this._params.set(name, moved);
      rebind(t, moved);
    }
    for (const [name, e] of this._buffers) {
      const moved = move(e.tensor);
      this._buffers.set(name, { tensor: moved, persistent: e.persistent });
      rebind(e.tensor, moved);
    }
  }

  /** Hook for subclasses holding unregistered tensor references (e.g. caches) that must follow `to()`. */
  protected onTensorMoved(_from: Tensor, _to: Tensor): void {
    // default: nothing
  }

  /** Device of the first parameter (or "cpu" when there are none). */
  get device(): Device {
    const p = this.parameters()[0];
    return p ? p.device : "cpu";
  }

  /** Free every parameter gradient. */
  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }

  /** Apply `fn` to every module in the tree (children first, like PyTorch). */
  apply(fn: (m: Module) => void): this {
    for (const c of this.children()) c.apply(fn);
    fn(this);
    return this;
  }

  /** One-line description used by `toString()`. */
  extraRepr(): string {
    return "";
  }

  toString(): string {
    const lines: string[] = [];
    const walk = (m: Module, name: string, indent: string): void => {
      const extra = m.extraRepr();
      const kids = m.namedChildren();
      const head = `${indent}${name ? `(${name}): ` : ""}${m.constructor.name}(${extra}`;
      if (kids.size === 0) {
        lines.push(`${head})`);
        return;
      }
      lines.push(head);
      for (const [k, c] of kids) walk(c, k, indent + "  ");
      lines.push(`${indent})`);
    };
    walk(this, "", "");
    return lines.join("\n");
  }
}

/** An indexable list of modules whose children are named "0", "1", … */
export class ModuleList<M extends Module = Module> extends Module {
  readonly items: M[];

  constructor(items: readonly M[] = []) {
    super();
    this.items = [...items];
    this.items.forEach((m, i) => this.child(String(i), m));
  }

  get length(): number {
    return this.items.length;
  }

  /** Module at `i` (throws when out of range). */
  at(i: number): M {
    const m = this.items[i];
    if (!m) throw new Error(`ModuleList: index ${i} out of range`);
    return m;
  }

  /** Append a module. */
  push(m: M): this {
    this.child(String(this.items.length), m);
    this.items.push(m);
    return this;
  }

  [Symbol.iterator](): Iterator<M> {
    return this.items[Symbol.iterator]();
  }
}
