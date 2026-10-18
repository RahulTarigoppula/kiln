/**
 * A tiny signal / effect store (no framework).
 *
 * ```ts
 * const count = signal(0);
 * effect(() => { el.textContent = String(count()); });
 * count.set(1); // effect re-runs
 * ```
 * Effects track the signals they read during their last run and re-run synchronously on change
 * (deduplicated inside `batch()`). An effect may return a cleanup function.
 */

type Subscriber = () => void;

interface Tracker {
  run: Subscriber;
  deps: Set<Set<Tracker>>;
}

let activeTracker: Tracker | null = null;
let batchDepth = 0;
const pendingTrackers = new Set<Tracker>();

export interface ReadonlySignal<T> {
  (): T;
  /** Read without subscribing the current effect. */
  peek(): T;
  /** Subscribe explicitly; returns an unsubscribe function. */
  subscribe(fn: (value: T) => void): () => void;
}

export interface Signal<T> extends ReadonlySignal<T> {
  set(value: T): void;
  update(fn: (prev: T) => T): void;
}

function flush(): void {
  if (batchDepth > 0) return;
  const items = [...pendingTrackers];
  pendingTrackers.clear();
  for (const t of items) t.run();
}

export function signal<T>(initial: T, equals: (a: T, b: T) => boolean = Object.is): Signal<T> {
  let value = initial;
  const trackers = new Set<Tracker>();

  const read = (() => {
    if (activeTracker) {
      trackers.add(activeTracker);
      activeTracker.deps.add(trackers);
    }
    return value;
  }) as Signal<T>;

  read.peek = () => value;
  read.set = (next: T) => {
    if (equals(value, next)) return;
    value = next;
    for (const t of trackers) pendingTrackers.add(t);
    flush();
  };
  read.update = (fn) => read.set(fn(value));
  read.subscribe = (fn) => effect(() => fn(read()));
  return read;
}

/** Run `fn` now and again whenever any signal it read changes. Returns a disposer. */
export function effect(fn: () => void | (() => void)): () => void {
  let cleanup: void | (() => void);
  let disposed = false;
  const tracker: Tracker = {
    deps: new Set(),
    run: () => {
      if (disposed) return;
      if (typeof cleanup === "function") cleanup();
      for (const d of tracker.deps) d.delete(tracker);
      tracker.deps.clear();
      const prev = activeTracker;
      activeTracker = tracker;
      try {
        cleanup = fn();
      } finally {
        activeTracker = prev;
      }
    },
  };
  tracker.run();
  return () => {
    disposed = true;
    if (typeof cleanup === "function") cleanup();
    for (const d of tracker.deps) d.delete(tracker);
    tracker.deps.clear();
    pendingTrackers.delete(tracker);
  };
}

/** Derived read-only signal. */
export function computed<T>(fn: () => T, equals?: (a: T, b: T) => boolean): ReadonlySignal<T> & { dispose(): void } {
  const inner = signal<T>(undefined as unknown as T, equals);
  const dispose = effect(() => inner.set(fn()));
  const out = (() => inner()) as ReadonlySignal<T> & { dispose(): void };
  out.peek = () => inner.peek();
  out.subscribe = (cb) => inner.subscribe(cb);
  out.dispose = dispose;
  return out;
}

/** Coalesce several writes into one round of effect runs. */
export function batch(fn: () => void): void {
  batchDepth++;
  try {
    fn();
  } finally {
    batchDepth--;
    flush();
  }
}

/** Read signals inside `fn` without subscribing. */
export function untracked<T>(fn: () => T): T {
  const prev = activeTracker;
  activeTracker = null;
  try {
    return fn();
  } finally {
    activeTracker = prev;
  }
}

/** Collects disposers so a page can tear everything down in one call. */
export class Scope {
  private readonly disposers: (() => void)[] = [];
  add(d: () => void): void {
    this.disposers.push(d);
  }
  effect(fn: () => void | (() => void)): void {
    this.add(effect(fn));
  }
  listen<K extends keyof HTMLElementEventMap>(el: HTMLElement | Window | Document, type: K | string, fn: (ev: Event) => void, opts?: AddEventListenerOptions): void {
    el.addEventListener(type, fn, opts);
    this.add(() => el.removeEventListener(type, fn, opts));
  }
  interval(fn: () => void, ms: number): void {
    const id = setInterval(fn, ms);
    this.add(() => clearInterval(id));
  }
  dispose(): void {
    while (this.disposers.length) this.disposers.pop()?.();
  }
}
