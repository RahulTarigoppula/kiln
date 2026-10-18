import { describe, expect, it } from "vitest";
import { Tensor, fromArray } from "../src/tensor/index";
import { AdamW, adamwParamGroups, configureAdamW } from "../src/optim/adamw";
import { SGD } from "../src/optim/sgd";
import { clipGradNorm, gradNorm } from "../src/optim/clip";
import { cosineWithWarmup, makeCosineWithWarmup, stepSchedule, constantSchedule } from "../src/optim/schedulers";
import { Linear } from "../src/nn/linear";
import { GPT, gptConfig } from "../src/models/gpt";

function close(a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-6): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(Math.abs((a[i] as number) - (b[i] as number))).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i] as number)));
}

describe("AdamW", () => {
  it("matches a hand-rolled PyTorch-semantics reference for 5 steps", () => {
    const p = fromArray(Float32Array.from([1, -2, 0.5]), [3], { requiresGrad: true });
    const ref = [1, -2, 0.5];
    const m = [0, 0, 0];
    const v = [0, 0, 0];
    const lr = 0.1;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const wd = 0.1;
    const opt = new AdamW([p], { lr, betas: [b1, b2], eps, weightDecay: wd });
    for (let step = 1; step <= 5; step++) {
      // grad = 2·p + 1 (evaluated at the current params)
      const gvals = Array.from(p.dataSync()).map((x) => 2 * x + 1);
      p.grad = fromArray(Float32Array.from(gvals), [3]);
      opt.step();
      for (let i = 0; i < 3; i++) {
        const g = 2 * (ref[i] as number) + 1;
        ref[i] = (ref[i] as number) * (1 - lr * wd);
        m[i] = b1 * (m[i] as number) + (1 - b1) * g;
        v[i] = b2 * (v[i] as number) + (1 - b2) * g * g;
        const mhat = (m[i] as number) / (1 - b1 ** step);
        const vhat = (v[i] as number) / (1 - b2 ** step);
        ref[i] = (ref[i] as number) - lr * mhat / (Math.sqrt(vhat) + eps);
      }
      close(p.dataSync(), ref, 1e-5);
    }
    expect(opt.stepCount).toBe(5);
    const st = opt.stateTensors().get(p) as Record<string, Tensor>;
    close(st.m!.dataSync(), m, 1e-5);
    close(st.v!.dataSync(), v, 1e-5);
    expect(opt.stateDict()).toMatchObject({ type: "adamw", step: 5, defaults: { lr, weightDecay: wd } });
  });

  it("groups weight decay like nanoGPT (≥2-D decays, biases/LayerNorm do not)", () => {
    const gpt = new GPT(gptConfig("nano", { vocabSize: 10, blockSize: 4 }));
    const groups = adamwParamGroups(gpt, 0.1);
    expect(groups.length).toBe(2);
    expect(groups[0]!.hyper.weightDecay).toBe(0.1);
    expect(groups[1]!.hyper.weightDecay).toBe(0);
    expect(groups[0]!.params.every((p) => p.ndim >= 2)).toBe(true);
    expect(groups[1]!.params.every((p) => p.ndim < 2)).toBe(true);
    expect(groups[0]!.params.length + groups[1]!.params.length).toBe(gpt.parameters().length);
    const opt = configureAdamW(gpt, { lr: 1e-3, weightDecay: 0.1 });
    expect(opt.hyperOf(opt.groups[1]!).weightDecay).toBe(0);
    expect(opt.hyperOf(opt.groups[0]!).weightDecay).toBe(0.1);
    opt.lr = 5e-4;
    expect(opt.hyperOf(opt.groups[0]!).lr).toBe(5e-4);
    expect(opt.hyperOf(opt.groups[1]!).lr).toBe(5e-4);
    expect(() => new AdamW([{ params: [gpt.lm_head.weight] }, { params: [gpt.transformer.wte.weight] }])).toThrow(/more than one group/);
  });

  it("loads state tensors and hyper-parameters", () => {
    const p = fromArray(Float32Array.from([1, 2]), [2], { requiresGrad: true });
    const opt = new AdamW([p], { lr: 0.01 });
    opt.loadParamState(p, { m: fromArray(Float32Array.from([0.5, 0.5]), [2]), v: fromArray(Float32Array.from([0.25, 0.25]), [2]) });
    opt.loadStateDict({ type: "adamw", step: 7, defaults: { lr: 0.02, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0 }, groups: [{ size: 1, hyper: {} }] });
    expect(opt.stepCount).toBe(7);
    expect(opt.lr).toBe(0.02);
    close(opt.stateTensors().get(p)!.m!.dataSync(), [0.5, 0.5]);
  });
});

describe("SGD", () => {
  it("matches a hand-rolled reference with momentum, nesterov and weight decay", () => {
    const p = fromArray(Float32Array.from([1, -1]), [2], { requiresGrad: true });
    const opt = new SGD([p], { lr: 0.1, momentum: 0.9, nesterov: 1, weightDecay: 0.01 });
    const ref = [1, -1];
    let buf: number[] | null = null;
    for (let s = 0; s < 4; s++) {
      p.grad = fromArray(Float32Array.from([0.5, -0.25]), [2]);
      opt.step();
      const g = [0.5, -0.25].map((gi, i) => gi + 0.01 * (ref[i] as number));
      buf = buf ? buf.map((b, i) => 0.9 * b + (g[i] as number)) : g.slice();
      const upd = g.map((gi, i) => gi + 0.9 * (buf as number[])[i]!);
      for (let i = 0; i < 2; i++) ref[i] = (ref[i] as number) - 0.1 * (upd[i] as number);
      close(p.dataSync(), ref, 1e-5);
    }
  });
});

describe("clipGradNorm", () => {
  it("scales gradients to the max norm on device and returns the pre-clip norm as a tensor", () => {
    const a = fromArray(Float32Array.from([3, 0]), [2], { requiresGrad: true });
    const b = fromArray(Float32Array.from([0, 4]), [2], { requiresGrad: true });
    a.grad = fromArray(Float32Array.from([3, 0]), [2]);
    b.grad = fromArray(Float32Array.from([0, 4]), [2]);
    const norm = clipGradNorm([a, b], 1); // total norm = 5
    expect(norm).toBeInstanceOf(Tensor);
    expect(norm.itemSync()).toBeCloseTo(5, 5);
    // grads were already scaled synchronously (no await needed)
    const scale = 1 / (5 + 1e-6);
    close(a.grad!.dataSync(), [3 * scale, 0], 1e-5);
    close(b.grad!.dataSync(), [0, 4 * scale], 1e-5);
    expect(gradNorm([a, b]).itemSync()).toBeCloseTo(1, 4);
    // below the threshold nothing changes
    const before = Array.from(a.grad!.dataSync());
    const n2 = clipGradNorm([a, b], 10);
    expect(n2.itemSync()).toBeCloseTo(1, 4);
    close(a.grad!.dataSync(), before);
    // params without grads are skipped
    const c = fromArray(Float32Array.from([1]), [1], { requiresGrad: true });
    expect(clipGradNorm([c], 1).itemSync()).toBe(0);
  });

  it("works on model gradients", () => {
    const lin = new Linear(4, 3);
    lin.forward(Tensor.randn([5, 4])).mul(100).sum().backward();
    const n = clipGradNorm(lin.parameters(), 0.5).itemSync();
    expect(n).toBeGreaterThan(0.5);
    expect(gradNorm(lin.parameters()).itemSync()).toBeCloseTo(0.5, 3);
  });
});

describe("schedulers", () => {
  it("cosine with warmup follows nanoGPT", () => {
    const o = { warmup: 10, total: 110, lrMax: 1e-3, lrMin: 1e-4 };
    expect(cosineWithWarmup(0, o)).toBeCloseTo(1e-3 / 11, 10);
    expect(cosineWithWarmup(9, o)).toBeCloseTo(1e-3 * 10 / 11, 10);
    expect(cosineWithWarmup(10, o)).toBeCloseTo(1e-3, 10);
    expect(cosineWithWarmup(60, o)).toBeCloseTo(1e-4 + 0.5 * (1e-3 - 1e-4), 10);
    expect(cosineWithWarmup(110, o)).toBeCloseTo(1e-4, 10);
    expect(cosineWithWarmup(500, o)).toBeCloseTo(1e-4, 10);
    const f = makeCosineWithWarmup(o);
    let prev = f(10);
    for (let s = 11; s <= 110; s++) {
      const cur = f(s);
      expect(cur).toBeLessThanOrEqual(prev + 1e-12);
      prev = cur;
    }
    expect(cosineWithWarmup(200, { warmup: 0, total: 100, lrMax: 1 })).toBe(0.1); // default lrMin = lrMax / 10
  });

  it("step and constant schedules", () => {
    expect(stepSchedule(0, { lr: 1, stepSize: 10, gamma: 0.5 })).toBe(1);
    expect(stepSchedule(10, { lr: 1, stepSize: 10, gamma: 0.5 })).toBe(0.5);
    expect(stepSchedule(25, { lr: 1, stepSize: 10, gamma: 0.5 })).toBe(0.25);
    expect(constantSchedule(3e-4)(12345)).toBe(3e-4);
  });
});
