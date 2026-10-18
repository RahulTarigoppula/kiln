/**
 * Small elementwise kernels: `copy_strided`, `fill`, `where`, `masked_fill`.
 *
 * Masks/conditions are read as u32 bit patterns so both f32 (0.0/1.0) and u32 (0/1) masks work; the sign
 * bit is masked off so -0.0 counts as false, matching `cond[i] != 0` on the CPU.
 */

import { makeBuilder, type KernelBuilder } from "./common";

/**
 * Materialise a strided view (rank ≤ 6) into a contiguous buffer.
 * Uniform (words): [n, rank, offset(i32), 0, shape(8 u32), strides(8 i32)] — 20 words.
 */
export const COPY_STRIDED_TEMPLATE = /* wgsl */ `
struct Params {
  n: u32, rank: u32, offset: i32, pad: u32,
  shape: array<vec4<u32>, 2>,
  strides: array<vec4<i32>, 2>,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> O: array<f32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    var rem = i;
    var off = P.offset;
    for (var d = P.rank; d > 0u; d = d - 1u) {
      let dd = d - 1u;
      let s = P.shape[dd >> 2u][dd & 3u];
      let idx = rem % s;
      rem = rem / s;
      off = off + i32(idx) * P.strides[dd >> 2u][dd & 3u];
    }
    O[i] = A[u32(off)];
  }
}
`;

/** Fill with a constant. Uniform (words): [n, value(f32)]. */
export const FILL_TEMPLATE = /* wgsl */ `
struct Params { n: u32, value: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> O: array<f32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    O[i] = P.value;
  }
}
`;

/** out = cond != 0 ? a : b. Uniform (words): [n]. */
export const WHERE_TEMPLATE = /* wgsl */ `
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> C: array<u32>;
@group(0) @binding(2) var<storage, {{A1}}> A: array<f32>;
@group(0) @binding(3) var<storage, {{A2}}> B: array<f32>;
@group(0) @binding(4) var<storage, {{A3}}> O: array<f32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    let c = (C[i] & 0x7FFFFFFFu) != 0u;
    O[i] = select(B[i], A[i], c);
  }
}
`;

/** out = mask != 0 ? value : a. Uniform (words): [n, value(f32)]. */
export const MASKED_FILL_TEMPLATE = /* wgsl */ `
struct Params { n: u32, value: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> M: array<u32>;
@group(0) @binding(3) var<storage, {{A2}}> O: array<f32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    let m = (M[i] & 0x7FFFFFFFu) != 0u;
    O[i] = select(A[i], P.value, m);
  }
}
`;

export function copyStridedKernel(wg: number): KernelBuilder {
  return makeBuilder("copy_strided", COPY_STRIDED_TEMPLATE, { WG: wg }, ["read", "read_write"]);
}
export function fillKernel(wg: number): KernelBuilder {
  return makeBuilder("fill", FILL_TEMPLATE, { WG: wg }, ["read_write"]);
}
export function whereKernel(wg: number): KernelBuilder {
  return makeBuilder("where", WHERE_TEMPLATE, { WG: wg }, ["read", "read", "read", "read_write"]);
}
export function maskedFillKernel(wg: number): KernelBuilder {
  return makeBuilder("masked_fill", MASKED_FILL_TEMPLATE, { WG: wg }, ["read", "read", "read_write"]);
}
