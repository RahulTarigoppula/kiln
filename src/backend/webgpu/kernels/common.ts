/**
 * Shared WGSL snippets and the template-substitution helper used by every kernel module.
 *
 * Kernels are WGSL template strings with `{{NAME}}` placeholders. A {@link KernelBuilder} binds the
 * kernel's options and produces a {@link KernelDef} for a given set of storage access modes (the backend
 * promotes `read` bindings to `read_write` when the same buffer is bound as both input and output).
 */

import type { Access, KernelDef } from "../pipeline";

/** Default workgroup size for 1-D kernels. */
export const DEFAULT_WG = 256;

/** Maximum tensor rank supported by the strided kernels (binary broadcast, copyStrided, reduce). */
export const MAX_RANK = 6;

/** Maximum workgroups per dispatch dimension guaranteed by core WebGPU. */
export const MAX_WORKGROUPS = 65535;

/** A kernel with options bound, awaiting the access-mode specialisation. */
export interface KernelBuilder {
  /** Human-readable kernel name (part of the cache key). */
  name: string;
  /** Natural access mode of each storage binding (bindings 1..k). */
  defaultAccess: readonly Access[];
  /** Cheap cache key for the given access modes (no template substitution). */
  key(access: readonly Access[]): string;
  /** Produce the specialised kernel for the given access modes. */
  build(access: readonly Access[]): KernelDef;
}

/** Replace every `{{NAME}}` in `template`; throws if a placeholder has no value. */
export function substitute(template: string, vars: Record<string, string | number>): string {
  const out = template.replace(/\{\{([A-Z0-9_]+)\}\}/g, (_m, name: string) => {
    const v = vars[name];
    if (v === undefined) throw new Error(`WGSL template: missing value for placeholder {{${name}}}`);
    return String(v);
  });
  const leftover = /\{\{[A-Z0-9_]+\}\}/.exec(out);
  if (leftover) throw new Error(`WGSL template: unresolved placeholder ${leftover[0]}`);
  return out;
}

/** Build the `{{A0}}..{{Ak}}` substitution map from access modes. */
export function accessVars(access: readonly Access[]): Record<string, string> {
  const vars: Record<string, string> = {};
  access.forEach((a, i) => {
    vars[`A${i}`] = a;
  });
  return vars;
}

/** Helper to construct a {@link KernelBuilder} from a template and option vars. */
export function makeBuilder(
  name: string,
  template: string,
  vars: Record<string, string | number>,
  defaultAccess: readonly Access[],
): KernelBuilder {
  const base = `${name}|${JSON.stringify(vars)}|`;
  return {
    name,
    defaultAccess,
    key(access) {
      return base + access.join(",");
    },
    build(access) {
      if (access.length !== defaultAccess.length) {
        throw new Error(`${name}: expected ${defaultAccess.length} access modes, got ${access.length}`);
      }
      const code = substitute(template, { ...vars, ...accessVars(access) });
      return { key: base + access.join(","), code, storage: access };
    },
  };
}

/** Numerically robust math helpers shared by several kernels. */
export const WGSL_MATH = /* wgsl */ `
fn kiln_tanh(x: f32) -> f32 {
  let e = exp(-2.0 * abs(x));
  let t = (1.0 - e) / (1.0 + e);
  return select(t, -t, x < 0.0);
}
fn kiln_sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}
fn kiln_tanh_grad(x: f32) -> f32 {
  let t = kiln_tanh(x);
  return 1.0 - t * t;
}
fn kiln_sigmoid_grad(x: f32) -> f32 {
  let s = kiln_sigmoid(x);
  return s * (1.0 - s);
}
// Math.pow semantics for negative bases with integral exponents (WGSL pow is undefined there).
fn kiln_pow(x: f32, p: f32) -> f32 {
  let ip = round(p);
  if (abs(p - ip) < 1e-6) {
    let m = pow(abs(x), p);
    let odd = (i32(ip) & 1) != 0;
    return select(m, -m, odd && (x < 0.0));
  }
  return pow(x, p);
}
fn kiln_gelu(x: f32) -> f32 {
  let u = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + kiln_tanh(u));
}
fn kiln_gelu_grad(x: f32) -> f32 {
  let k = 0.7978845608028654;
  let c = 0.044715;
  let u = k * (x + c * x * x * x);
  let t = kiln_tanh(u);
  let du = k * (1.0 + 3.0 * c * x * x);
  return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du;
}
`;

/**
 * Counter-based hash RNG shared bit-for-bit with the CPU backend:
 * `h = splitmix32(seed ^ (i * 0x9E3779B9))`, uniform = (h >>> 8) / 2^24.
 */
export const WGSL_HASH = /* wgsl */ `
fn kiln_splitmix32(x0: u32) -> u32 {
  var x = x0 + 0x9E3779B9u;
  x = x ^ (x >> 16u);
  x = x * 0x21F0AAADu;
  x = x ^ (x >> 15u);
  x = x * 0x735A2D97u;
  x = x ^ (x >> 15u);
  return x;
}
fn kiln_hash(seed: u32, i: u32) -> u32 {
  return kiln_splitmix32(seed ^ (i * 0x9E3779B9u));
}
fn kiln_uniform(seed: u32, i: u32) -> f32 {
  return f32(kiln_hash(seed, i) >> 8u) / 16777216.0;
}
`;

/**
 * Workgroup tree reductions over `{{WG}}` lanes. Each call must happen in uniform control flow.
 * Shared arrays: `kiln_sh` (f32) and `kiln_sh2` (f32), `kiln_shi` (u32).
 */
export const WGSL_WG_REDUCE = /* wgsl */ `
var<workgroup> kiln_sh: array<f32, {{WG}}>;
var<workgroup> kiln_sh2: array<f32, {{WG}}>;
var<workgroup> kiln_shi: array<u32, {{WG}}>;
fn wg_sum(lid: u32, v: f32) -> f32 {
  kiln_sh[lid] = v;
  workgroupBarrier();
  for (var s = {{WG}}u / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) { kiln_sh[lid] = kiln_sh[lid] + kiln_sh[lid + s]; }
    workgroupBarrier();
  }
  let r = kiln_sh[0];
  workgroupBarrier();
  return r;
}
fn wg_max(lid: u32, v: f32) -> f32 {
  kiln_sh[lid] = v;
  workgroupBarrier();
  for (var s = {{WG}}u / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) { kiln_sh[lid] = max(kiln_sh[lid], kiln_sh[lid + s]); }
    workgroupBarrier();
  }
  let r = kiln_sh[0];
  workgroupBarrier();
  return r;
}
fn wg_min(lid: u32, v: f32) -> f32 {
  kiln_sh[lid] = v;
  workgroupBarrier();
  for (var s = {{WG}}u / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) { kiln_sh[lid] = min(kiln_sh[lid], kiln_sh[lid + s]); }
    workgroupBarrier();
  }
  let r = kiln_sh[0];
  workgroupBarrier();
  return r;
}
// Online (max, sum-of-exp) merge: returns vec2(M, S) for the whole workgroup.
fn wg_online_softmax(lid: u32, m: f32, s: f32) -> vec2<f32> {
  kiln_sh[lid] = m;
  kiln_sh2[lid] = s;
  workgroupBarrier();
  for (var st = {{WG}}u / 2u; st > 0u; st = st >> 1u) {
    if (lid < st) {
      let m1 = kiln_sh[lid];
      let s1 = kiln_sh2[lid];
      let m2 = kiln_sh[lid + st];
      let s2 = kiln_sh2[lid + st];
      let mm = max(m1, m2);
      kiln_sh[lid] = mm;
      kiln_sh2[lid] = s1 * exp(m1 - mm) + s2 * exp(m2 - mm);
    }
    workgroupBarrier();
  }
  let r = vec2<f32>(kiln_sh[0], kiln_sh2[0]);
  workgroupBarrier();
  return r;
}
`;

/** Lowest finite f32, used in place of -inf where a literal is needed. */
export const F32_LOWEST = "-3.4028234663852886e38";
/** Largest finite f32. */
export const F32_MAX = "3.4028234663852886e38";
