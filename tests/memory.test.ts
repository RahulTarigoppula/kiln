import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Backend, Buf } from "../src/backend/types";
import { cpuBackend } from "../src/backend/cpu/index";
import { registerBackend } from "../src/tensor/device";
import { Tensor, fromArray } from "../src/tensor/tensor";
import { beginScope, endScope, escape, keep, scopeDepth, tidy } from "../src/tensor/scope";
import { GPT, gptConfig } from "../src/models/gpt";
import { configureAdamW } from "../src/optim/adamw";
import { clipGradNorm } from "../src/optim/clip";

/** CPU backend wrapper that counts live buffers (the real CPU `free` is a GC no-op). */
function countingBackend(): Backend & { live: Set<Buf>; allocs: number } {
  const live = new Set<Buf>();
  const wrapper = { live, allocs: 0 } as unknown as Backend & { live: Set<Buf>; allocs: number };
  return new Proxy(wrapper, {
    get(target, prop) {
      if (prop === "live" || prop === "allocs") return target[prop as "live" | "allocs"];
      if (prop === "alloc") {
        return (bytes: number): Buf => {
          const b = cpuBackend.alloc(bytes);
          live.add(b);
          target.allocs++;
          return b;
        };
      }
      if (prop === "free") {
        return (b: Buf): void => {
          if (!live.has(b)) throw new Error("double free");
          live.delete(b);
          cpuBackend.free(b);
        };
      }
      const v = (cpuBackend as unknown as Record<string | symbol, unknown>)[prop];
      return typeof v === "function" ? (v as (...a: unknown[]) => unknown).bind(cpuBackend) : v;
    },
  });
}

const counting = countingBackend();

beforeAll(() => registerBackend("cpu", counting));
afterAll(() => registerBackend("cpu", cpuBackend));

describe("tidy", () => {
  it("frees intermediates but not results, kept or escaped tensors", () => {
    const a = fromArray(Float32Array.from([1, 2, 3]), [3]);
    const base = counting.live.size;
    let inner: Tensor | null = null;
    let kept: Tensor | null = null;
    let escaped: Tensor | null = null;
    const out = tidy(() => {
      inner = a.add(1);
      kept = keep(inner.mul(3));
      escaped = escape(a.mul(0));
      return inner.mul(2);
    });
    expect(inner!.isFreed).toBe(true);
    expect(out.isFreed).toBe(false);
    expect(kept!.isFreed).toBe(false);
    expect(escaped!.isFreed).toBe(false);
    expect(counting.live.size).toBe(base + 3);
    expect(Array.from(out.dataSync())).toEqual([4, 6, 8]);
    out.free();
    kept!.free();
    escaped!.free();
    expect(counting.live.size).toBe(base);
    expect(scopeDepth()).toBe(0);
  });

  it("collects tensors from arrays/objects, hands kept tensors to the parent scope, survives throws", () => {
    const a = fromArray(Float32Array.from([1, 2]), [2]);
    const base = counting.live.size;
    const outer = tidy(() => {
      const r = tidy(() => ({ x: a.add(1), ys: [a.add(2), a.add(3)], n: 42 }));
      expect(r.x.isFreed).toBe(false);
      expect(counting.live.size).toBe(base + 3);
      return r.ys[1] as Tensor; // x and ys[0] belong to the outer scope now → freed here
    });
    expect(counting.live.size).toBe(base + 1);
    outer.free();
    expect(() => tidy(() => { a.add(5); throw new Error("boom"); })).toThrow("boom");
    expect(counting.live.size).toBe(base);
    expect(scopeDepth()).toBe(0);
    expect(() => tidy(() => Promise.resolve(a))).toThrow(/Promise/);
    expect(scopeDepth()).toBe(0);
  });

  it("keeps the live autograd graph of a returned tensor and frees it after backward", () => {
    const w = fromArray(Float32Array.from([1, 2, 3]), [3], { requiresGrad: true });
    const base = counting.live.size;
    const y = tidy(() => w.mul(w).add(1).sum());
    expect(counting.live.size).toBeGreaterThan(base); // graph intermediates alive
    y.backward();
    expect(Array.from(w.grad!.dataSync())).toEqual([2, 4, 6]);
    y.free();
    // After backward only w.grad should remain allocated beyond the baseline.
    const alive = counting.live.size - base;
    expect(alive).toBeLessThanOrEqual(4); // grad + unreleasable op outputs still referenced by y's released graph
    w.zeroGrad();
    expect(w.grad).toBeNull();
    // manual scopes
    beginScope();
    const t = w.add(1);
    const freed = endScope(undefined);
    expect(freed).toBe(1);
    expect(t.isFreed).toBe(true);
  });
});

describe("allocation flatness", () => {
  it("live buffer count is flat across 20 training steps (forward+backward in tidy, AdamW, clipping)", () => {
    const gpt = new GPT(gptConfig("nano", { vocabSize: 16, blockSize: 8, dropout: 0.1 }));
    const opt = configureAdamW(gpt, { lr: 1e-3 });
    const B = 4;
    const T = 8;
    const x = fromArray(Int32Array.from({ length: B * T }, (_, i) => i % 16), [B, T], { dtype: "i32" });
    const y = fromArray(Int32Array.from({ length: B * T }, (_, i) => (i + 1) % 16), [B, T], { dtype: "i32" });
    const step = (): number => {
      const l = tidy(() => {
        const { loss } = gpt.forward(x, y);
        (loss as Tensor).backward();
        return (loss as Tensor).detach();
      });
      clipGradNorm(gpt.parameters(), 1).free();
      opt.step();
      opt.zeroGrad();
      const v = l.itemSync();
      l.free();
      return v;
    };
    step(); // warm-up: creates AdamW m/v state and lazily cached masks/positions
    const live = counting.live.size;
    const allocsBefore = counting.allocs;
    const losses: number[] = [];
    for (let i = 0; i < 20; i++) {
      losses.push(step());
      expect(counting.live.size).toBe(live);
    }
    expect(counting.allocs).toBeGreaterThan(allocsBefore); // work actually happened
    expect(losses.every((v) => Number.isFinite(v))).toBe(true);
    // Parameters + optimizer state are the bulk of what stays alive.
    const params = gpt.parameters().length;
    expect(live).toBeGreaterThanOrEqual(params * 3);
  });

  it("generate does not leak", async () => {
    const gpt = new GPT(gptConfig("nano", { vocabSize: 16, blockSize: 8 }));
    gpt.eval();
    const prompt = fromArray(Int32Array.from([1, 2]), [1, 2], { dtype: "i32" });
    const warm = await gpt.generate(prompt, 12, { temperature: 0 });
    warm.free();
    const live = counting.live.size;
    const out = await gpt.generate(prompt, 12, { temperature: 0.7, seed: 1 });
    out.free();
    expect(counting.live.size).toBe(live);
  });
});
