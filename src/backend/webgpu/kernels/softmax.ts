/**
 * Row-wise softmax / log-softmax and their backward. One workgroup per row (rows beyond the dispatch
 * limit are handled by a workgroup-stride loop); columns are strided per lane so any `cols` works.
 * Forward uses an online (max, sum-exp) accumulation so the row is read twice in total.
 *
 * Uniform layout (words): [rows, cols, log(u32), 0].
 */

import { F32_LOWEST, WGSL_WG_REDUCE, makeBuilder, type KernelBuilder } from "./common";

export const SOFTMAX_TEMPLATE = /* wgsl */ `
struct Params { rows: u32, cols: u32, logMode: u32, pad: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> O: array<f32>;
${WGSL_WG_REDUCE}
@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  for (var row = wid.x; row < P.rows; row = row + nwg.x) {
    let base = row * P.cols;
    var m = ${F32_LOWEST};
    var s = 0.0;
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) {
      let x = A[base + c];
      if (x > m) {
        s = s * exp(m - x) + 1.0;
        m = x;
      } else {
        s = s + exp(x - m);
      }
    }
    let ms = wg_online_softmax(lid.x, m, s);
    let M = ms.x;
    let S = ms.y;
    if (P.logMode != 0u) {
      let lse = M + log(S);
      for (var c = lid.x; c < P.cols; c = c + {{WG}}u) { O[base + c] = A[base + c] - lse; }
    } else {
      let inv = 1.0 / S;
      for (var c = lid.x; c < P.cols; c = c + {{WG}}u) { O[base + c] = exp(A[base + c] - M) * inv; }
    }
  }
}
`;

/**
 * Backward. softmax: gradIn = y * (g - Σ g·y). log-softmax: gradIn = g - exp(y) * Σ g.
 * Bindings: Y (forward output), G (gradOut), O (gradIn).
 */
export const SOFTMAX_BACKWARD_TEMPLATE = /* wgsl */ `
struct Params { rows: u32, cols: u32, logMode: u32, pad: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> Y: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> G: array<f32>;
@group(0) @binding(3) var<storage, {{A2}}> O: array<f32>;
${WGSL_WG_REDUCE}
@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  for (var row = wid.x; row < P.rows; row = row + nwg.x) {
    let base = row * P.cols;
    var acc = 0.0;
    if (P.logMode != 0u) {
      for (var c = lid.x; c < P.cols; c = c + {{WG}}u) { acc = acc + G[base + c]; }
    } else {
      for (var c = lid.x; c < P.cols; c = c + {{WG}}u) { acc = acc + G[base + c] * Y[base + c]; }
    }
    let total = wg_sum(lid.x, acc);
    if (P.logMode != 0u) {
      for (var c = lid.x; c < P.cols; c = c + {{WG}}u) { O[base + c] = G[base + c] - exp(Y[base + c]) * total; }
    } else {
      for (var c = lid.x; c < P.cols; c = c + {{WG}}u) { O[base + c] = Y[base + c] * (G[base + c] - total); }
    }
  }
}
`;

export function softmaxKernel(wg: number): KernelBuilder {
  return makeBuilder("softmax_rows", SOFTMAX_TEMPLATE, { WG: wg }, ["read", "read_write"]);
}
export function softmaxBackwardKernel(wg: number): KernelBuilder {
  return makeBuilder("softmax_backward", SOFTMAX_BACKWARD_TEMPLATE, { WG: wg }, ["read", "read", "read_write"]);
}
