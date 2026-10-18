import { describe, expect, it } from "vitest";
import { Random, hashNormal, hashU32, hashUniform, manualSeed, nextSeed, splitmix32 } from "../src/tensor/random";
import { Tensor, rand, randn } from "../src/tensor/tensor";
import { dropout } from "../src/autograd/ops";

function meanVar(arr: ArrayLike<number>): { mean: number; variance: number } {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i] as number;
  const mean = s / arr.length;
  let v = 0;
  for (let i = 0; i < arr.length; i++) { const d = (arr[i] as number) - mean; v += d * d; }
  return { mean, variance: v / arr.length };
}

describe("xoshiro128** Random", () => {
  it("is reproducible for the same seed", () => {
    const a = new Random(123);
    const b = new Random(123);
    for (let i = 0; i < 100; i++) expect(a.nextU32()).toBe(b.nextU32());
    expect(new Random(1).nextU32()).not.toBe(new Random(2).nextU32());
  });

  it("uniform() has the right range and moments", () => {
    const r = new Random(7);
    const xs = new Float64Array(100_000);
    let inRange = true;
    for (let i = 0; i < xs.length; i++) {
      const u = r.uniform();
      xs[i] = u;
      if (!(u >= 0 && u < 1)) inRange = false;
    }
    expect(inRange).toBe(true);
    const { mean, variance } = meanVar(xs);
    expect(mean).toBeCloseTo(0.5, 2);
    expect(variance).toBeCloseTo(1 / 12, 2);
  });

  it("normal() has mean 0 and variance 1", () => {
    const r = new Random(99);
    const xs = new Float64Array(100_000);
    for (let i = 0; i < xs.length; i++) xs[i] = r.normal();
    const { mean, variance } = meanVar(xs);
    expect(Math.abs(mean)).toBeLessThan(0.02);
    expect(Math.abs(variance - 1)).toBeLessThan(0.03);
  });

  it("shuffle is a permutation and int() is in range", () => {
    const r = new Random(5);
    const arr = Array.from({ length: 50 }, (_, i) => i);
    const shuffled = r.shuffle(arr.slice());
    expect(shuffled.slice().sort((a, b) => a - b)).toEqual(arr);
    expect(shuffled).not.toEqual(arr);
    let ok = true;
    for (let i = 0; i < 1000; i++) {
      const k = r.int(10);
      if (!(k >= 0 && k < 10 && Number.isInteger(k))) ok = false;
    }
    expect(ok).toBe(true);
  });

  it("manualSeed makes nextSeed reproducible", () => {
    manualSeed(42);
    const a = [nextSeed(), nextSeed(), nextSeed()];
    manualSeed(42);
    const b = [nextSeed(), nextSeed(), nextSeed()];
    expect(a).toEqual(b);
  });
});

describe("counter-based hash RNG (CPU/GPU shared spec)", () => {
  it("splitmix32 matches the documented reference values", () => {
    // Reference values computed from the spec: mix(x + 0x9E3779B9) with the 0x21f0aaad / 0x735a2d97 constants.
    const ref = (x: number): number => {
      let h = (x + 0x9e3779b9) >>> 0;
      h = (h ^ (h >>> 16)) >>> 0;
      h = Math.imul(h, 0x21f0aaad) >>> 0;
      h = (h ^ (h >>> 15)) >>> 0;
      h = Math.imul(h, 0x735a2d97) >>> 0;
      h = (h ^ (h >>> 15)) >>> 0;
      return h;
    };
    for (const x of [0, 1, 2, 12345, 0xffffffff, 0x80000000]) expect(splitmix32(x)).toBe(ref(x));
    expect(hashU32(0, 0)).toBe(splitmix32(0));
    expect(hashU32(7, 3)).toBe(splitmix32((7 ^ Math.imul(3, 0x9e3779b9)) >>> 0));
  });

  it("hashUniform is in [0,1) with 24-bit resolution and hashNormal pairs", () => {
    let ok = true;
    for (let i = 0; i < 1000; i++) {
      const u = hashUniform(11, i);
      if (!(u >= 0 && u < 1) || u * 16777216 !== Math.floor(u * 16777216)) ok = false;
    }
    expect(ok).toBe(true);
    const u1 = hashUniform(11, 4);
    const u2 = hashUniform(11, 5);
    const r = Math.sqrt(-2 * Math.log(1 - u1));
    expect(hashNormal(11, 4)).toBeCloseTo(r * Math.cos(2 * Math.PI * u2), 12);
    expect(hashNormal(11, 5)).toBeCloseTo(r * Math.sin(2 * Math.PI * u2), 12);
  });

  it("Tensor.rand / randn are reproducible per seed and have sane moments", () => {
    const a = rand([100_000], { seed: 3 }).dataSync();
    const b = rand([100_000], { seed: 3 }).dataSync();
    expect(a).toEqual(b);
    const mu = meanVar(a);
    expect(mu.mean).toBeCloseTo(0.5, 2);
    expect(mu.variance).toBeCloseTo(1 / 12, 2);

    const n = randn([100_000], { seed: 3 }).dataSync();
    const mn = meanVar(n);
    expect(Math.abs(mn.mean)).toBeLessThan(0.02);
    expect(Math.abs(mn.variance - 1)).toBeLessThan(0.03);
    expect(randn([8], { seed: 3 }).dataSync()).not.toEqual(randn([8], { seed: 4 }).dataSync());
  });

  it("manualSeed makes seedless randn reproducible", () => {
    manualSeed(2024);
    const a = randn([16]).dataSync();
    manualSeed(2024);
    const b = randn([16]).dataSync();
    expect(a).toEqual(b);
  });

  it("dropout keeps ~1-p of elements, scales survivors and is reproducible", () => {
    const x = Tensor.ones([50_000]);
    const y = dropout(x, 0.3, true, 77).dataSync();
    let kept = 0;
    let scaledOk = true;
    for (let i = 0; i < y.length; i++) {
      const v = y[i] as number;
      if (v !== 0) {
        kept++;
        if (Math.abs(v - 1 / 0.7) > 1e-5) scaledOk = false;
      }
    }
    expect(scaledOk).toBe(true);
    expect(kept / y.length).toBeCloseTo(0.7, 1);
    expect(dropout(x, 0.3, true, 77).dataSync()).toEqual(y);
    expect(dropout(x, 0.3, false)).toBe(x);
  });
});
