/**
 * Counter-based RNG kernels (`rand`, `randn`, `dropoutMask`), bit-identical to the CPU backend's hash:
 *   h = splitmix32(seed ^ (i * 0x9E3779B9)),  uniform(i) = (h >>> 8) / 2^24.
 * Normal samples use Box–Muller on the pair (i & ~1, i | 1): u1 = uniform(i & ~1), u2 = uniform(i | 1);
 * even i → r·cos(2πu2), odd i → r·sin(2πu2) with r = sqrt(-2 ln(1 - u1)) (1 - u1 ∈ (0, 1], so no clamp),
 * exactly as `src/tensor/random.ts` does on the CPU.
 *
 * Uniform layout (words): [n, seed(u32), p(f32), 0].
 */

import { WGSL_HASH, makeBuilder, type KernelBuilder } from "./common";

/** Generator bodies, in terms of `i`, `P.seed`, `P.p`. */
export const RNG_BODY = {
  rand: `return kiln_uniform(P.seed, i);`,
  randn: `
  let i0 = i & ~1u;
  let i1 = i | 1u;
  let u1 = kiln_uniform(P.seed, i0);
  let u2 = kiln_uniform(P.seed, i1);
  let r = sqrt(-2.0 * log(1.0 - u1));
  let th = 6.283185307179586 * u2;
  return select(r * sin(th), r * cos(th), (i & 1u) == 0u);`,
  dropoutMask: `return select(1.0, 0.0, kiln_uniform(P.seed, i) < P.p);`,
} as const;

export type RngMode = keyof typeof RNG_BODY;

export const RNG_TEMPLATE = /* wgsl */ `
struct Params { n: u32, seed: u32, p: f32, pad: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> O: array<f32>;
${WGSL_HASH}
fn gen(i: u32) -> f32 {
  {{BODY}}
}
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    O[i] = gen(i);
  }
}
`;

export function rngKernel(mode: RngMode, wg: number): KernelBuilder {
  return makeBuilder(`rng_${mode}`, RNG_TEMPLATE, { WG: wg, BODY: RNG_BODY[mode] }, ["read_write"]);
}

/**
 * Reference JS implementation of the same hash (used by tests and available to the CPU backend).
 * All arithmetic is u32.
 */
export function splitmix32(x: number): number {
  let v = (x + 0x9e3779b9) >>> 0;
  v ^= v >>> 16;
  v = Math.imul(v, 0x21f0aaad) >>> 0;
  v ^= v >>> 15;
  v = Math.imul(v, 0x735a2d97) >>> 0;
  v ^= v >>> 15;
  return v >>> 0;
}

/** uniform(seed, i) in [0, 1) — same value the WGSL `kiln_uniform` produces. */
export function hashUniform(seed: number, i: number): number {
  const h = splitmix32(((seed >>> 0) ^ Math.imul(i >>> 0, 0x9e3779b9)) >>> 0);
  return (h >>> 8) / 16777216;
}

/** Reference normal sample for index i (Box–Muller as described above). */
export function hashNormal(seed: number, i: number): number {
  const i0 = (i & ~1) >>> 0;
  const i1 = (i | 1) >>> 0;
  const u1 = hashUniform(seed, i0);
  const u2 = hashUniform(seed, i1);
  const r = Math.sqrt(-2 * Math.log(1 - u1));
  const th = 2 * Math.PI * u2;
  return (i & 1) === 0 ? r * Math.cos(th) : r * Math.sin(th);
}
