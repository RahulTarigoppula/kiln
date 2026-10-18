/**
 * Seeded randomness.
 *
 *  - `Random` is a xoshiro128** stream generator (host-side init, shuffling, seed derivation).
 *  - `splitmix32` / `hashUniform` / `hashNormal` are the *counter-based* hash RNG the CPU and WebGPU
 *    backends both implement for `rand`/`randn`/`dropoutMask`, so results match across devices:
 *
 *        mix(x)            : x ^= x >>> 16; x *= 0x21f0aaad; x ^= x >>> 15; x *= 0x735a2d97; x ^= x >>> 15
 *        splitmix32(x)     : mix(x + 0x9E3779B9)            (all u32, wrapping)
 *        hash(seed, i)     : splitmix32(seed ^ (i * 0x9E3779B9))
 *        uniform(seed, i)  : (hash(seed, i) >>> 8) / 16777216        in [0, 1)
 *        normal(seed, i)   : j = i & ~1; u1 = uniform(seed, j); u2 = uniform(seed, j | 1)
 *                            r = sqrt(-2 ln(1 - u1)); t = 2π u2;  even i → r cos t, odd i → r sin t
 *        dropoutMask(i)    : uniform(seed, i) < p ? 0 : 1
 */

/** Fold any finite number into a u32. */
export function toU32(x: number): number {
  return (Math.floor(x) >>> 0) || (x !== 0 ? 1 : 0);
}

/** splitmix32 finaliser: a well-mixed u32 hash of a u32 input. */
export function splitmix32(x: number): number {
  let h = (x + 0x9e3779b9) | 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x21f0aaad);
  h ^= h >>> 15;
  h = Math.imul(h, 0x735a2d97);
  h ^= h >>> 15;
  return h >>> 0;
}

/** Counter-based u32 hash of (seed, index). */
export function hashU32(seed: number, i: number): number {
  return splitmix32((seed ^ Math.imul(i, 0x9e3779b9)) >>> 0);
}

/** Counter-based uniform in [0, 1) with 24 bits of precision. */
export function hashUniform(seed: number, i: number): number {
  return (hashU32(seed, i) >>> 8) / 16777216;
}

/** Counter-based standard normal (Box–Muller over the pair (i & ~1, i | 1)). */
export function hashNormal(seed: number, i: number): number {
  const j = i & ~1;
  const u1 = hashUniform(seed, j);
  const u2 = hashUniform(seed, j | 1);
  const r = Math.sqrt(-2 * Math.log(1 - u1));
  const t = 2 * Math.PI * u2;
  return (i & 1) === 0 ? r * Math.cos(t) : r * Math.sin(t);
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/** xoshiro128** pseudo-random generator seeded from a single u32. */
export class Random {
  private s0 = 0;
  private s1 = 0;
  private s2 = 0;
  private s3 = 0;
  private spare: number | null = null;

  constructor(seed: number) {
    this.seed(seed);
  }

  /** Re-seed the generator (state expanded with splitmix32). */
  seed(seed: number): void {
    const s = toU32(seed);
    this.s0 = splitmix32(s);
    this.s1 = splitmix32(this.s0);
    this.s2 = splitmix32(this.s1);
    this.s3 = splitmix32(this.s2);
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s3 = 1;
    this.spare = null;
  }

  /** Next raw u32. */
  nextU32(): number {
    const s0 = this.s0;
    const s1 = this.s1;
    const s2 = this.s2 ^ s0;
    const s3 = this.s3 ^ s1;
    const result = Math.imul(rotl(Math.imul(s1, 5) >>> 0, 7), 9) >>> 0;
    this.s1 = (s1 ^ s2) >>> 0;
    this.s0 = (s0 ^ s3) >>> 0;
    this.s2 = (s2 ^ (s1 << 9)) >>> 0;
    this.s3 = rotl(s3, 11);
    return result;
  }

  /** Uniform float in [0, 1). */
  uniform(): number {
    return this.nextU32() / 4294967296;
  }

  /** Uniform float in [lo, hi). */
  uniformRange(lo: number, hi: number): number {
    return lo + (hi - lo) * this.uniform();
  }

  /** Standard normal via Box–Muller (second sample cached). */
  normal(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    const u1 = 1 - this.uniform();
    const u2 = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    const t = 2 * Math.PI * u2;
    this.spare = r * Math.sin(t);
    return r * Math.cos(t);
  }

  /** Uniform integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.uniform() * n);
  }

  /** In-place Fisher–Yates shuffle. */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const t = arr[i] as T;
      arr[i] = arr[j] as T;
      arr[j] = t;
    }
    return arr;
  }

  /** Derive a fresh u32 seed for a child stream (kernel seeds, data loaders, ...). */
  nextSeed(): number {
    return this.nextU32();
  }
}

let defaultGenerator = new Random(0x5eed);

/** Reset the global generator so init/dropout sequences are reproducible. */
export function manualSeed(seed: number): void {
  defaultGenerator = new Random(seed);
}

/** The process-wide default generator. */
export function getDefaultGenerator(): Random {
  return defaultGenerator;
}

/** Draw a fresh kernel seed from the global generator. */
export function nextSeed(): number {
  return defaultGenerator.nextSeed();
}
