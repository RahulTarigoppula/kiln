/**
 * LayerNorm forward/backward over rows of `cols` features.
 *
 * Forward: one workgroup per row; two-pass mean/variance; writes `meanRstd[2*row] = mean`,
 * `meanRstd[2*row+1] = rstd`.
 *
 * Backward is split in three kernels:
 *  1. `layernorm_backward_x`: gradX per row (needs two row reductions).
 *  2. `layernorm_gb_partial`: per (column, row-chunk) partial sums of g·x̂ and g.
 *  3. `layernorm_gb_final`: sums the chunk partials -> gradGamma / gradBeta (deterministic order).
 *
 * Uniform layouts (words):
 *  forward:  [rows, cols, eps(f32), 0]
 *  bwd_x:    [rows, cols, 0, 0]
 *  partial:  [rows, cols, chunks, rowsPerChunk]
 *  final:    [cols, chunks, 0, 0]
 */

import { WGSL_WG_REDUCE, makeBuilder, type KernelBuilder } from "./common";

export const LAYERNORM_TEMPLATE = /* wgsl */ `
struct Params { rows: u32, cols: u32, eps: f32, pad: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> X: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> Gamma: array<f32>;
@group(0) @binding(3) var<storage, {{A2}}> Beta: array<f32>;
@group(0) @binding(4) var<storage, {{A3}}> O: array<f32>;
@group(0) @binding(5) var<storage, {{A4}}> MR: array<f32>;
${WGSL_WG_REDUCE}
@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let invN = 1.0 / f32(P.cols);
  for (var row = wid.x; row < P.rows; row = row + nwg.x) {
    let base = row * P.cols;
    var s = 0.0;
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) { s = s + X[base + c]; }
    let mean = wg_sum(lid.x, s) * invN;
    var q = 0.0;
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) {
      let d = X[base + c] - mean;
      q = q + d * d;
    }
    let variance = wg_sum(lid.x, q) * invN;
    let rstd = inverseSqrt(variance + P.eps);
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) {
      O[base + c] = (X[base + c] - mean) * rstd * Gamma[c] + Beta[c];
    }
    if (lid.x == 0u) {
      MR[2u * row] = mean;
      MR[2u * row + 1u] = rstd;
    }
  }
}
`;

export const LAYERNORM_BACKWARD_X_TEMPLATE = /* wgsl */ `
struct Params { rows: u32, cols: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> X: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> Gamma: array<f32>;
@group(0) @binding(3) var<storage, {{A2}}> MR: array<f32>;
@group(0) @binding(4) var<storage, {{A3}}> G: array<f32>;
@group(0) @binding(5) var<storage, {{A4}}> GX: array<f32>;
${WGSL_WG_REDUCE}
@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let invN = 1.0 / f32(P.cols);
  for (var row = wid.x; row < P.rows; row = row + nwg.x) {
    let base = row * P.cols;
    let mean = MR[2u * row];
    let rstd = MR[2u * row + 1u];
    var a = 0.0;
    var b = 0.0;
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) {
      let xhat = (X[base + c] - mean) * rstd;
      let dxhat = G[base + c] * Gamma[c];
      a = a + dxhat;
      b = b + dxhat * xhat;
    }
    let sa = wg_sum(lid.x, a) * invN;
    let sb = wg_sum(lid.x, b) * invN;
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) {
      let xhat = (X[base + c] - mean) * rstd;
      let dxhat = G[base + c] * Gamma[c];
      GX[base + c] = rstd * (dxhat - sa - xhat * sb);
    }
  }
}
`;

/** Partials: thread per column (x), workgroup y = row chunk. Part layout: [chunk][2][cols]. */
export const LAYERNORM_GB_PARTIAL_TEMPLATE = /* wgsl */ `
struct Params { rows: u32, cols: u32, chunks: u32, rowsPerChunk: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> X: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> MR: array<f32>;
@group(0) @binding(3) var<storage, {{A2}}> G: array<f32>;
@group(0) @binding(4) var<storage, {{A3}}> Part: array<f32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let c = gid.x;
  let chunk = wid.y;
  if (c >= P.cols || chunk >= P.chunks) { return; }
  let r0 = chunk * P.rowsPerChunk;
  let r1 = min(P.rows, r0 + P.rowsPerChunk);
  var pg = 0.0;
  var pb = 0.0;
  for (var r = r0; r < r1; r = r + 1u) {
    let g = G[r * P.cols + c];
    let xhat = (X[r * P.cols + c] - MR[2u * r]) * MR[2u * r + 1u];
    pg = pg + g * xhat;
    pb = pb + g;
  }
  Part[(chunk * 2u) * P.cols + c] = pg;
  Part[(chunk * 2u + 1u) * P.cols + c] = pb;
}
`;

export const LAYERNORM_GB_FINAL_TEMPLATE = /* wgsl */ `
struct Params { cols: u32, chunks: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> Part: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> GG: array<f32>;
@group(0) @binding(3) var<storage, {{A2}}> GB: array<f32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= P.cols) { return; }
  var sg = 0.0;
  var sb = 0.0;
  for (var k = 0u; k < P.chunks; k = k + 1u) {
    sg = sg + Part[(k * 2u) * P.cols + c];
    sb = sb + Part[(k * 2u + 1u) * P.cols + c];
  }
  GG[c] = sg;
  GB[c] = sb;
}
`;

export function layerNormKernel(wg: number): KernelBuilder {
  return makeBuilder("layernorm_rows", LAYERNORM_TEMPLATE, { WG: wg }, ["read", "read", "read", "read_write", "read_write"]);
}
export function layerNormBackwardXKernel(wg: number): KernelBuilder {
  return makeBuilder("layernorm_backward_x", LAYERNORM_BACKWARD_X_TEMPLATE, { WG: wg }, ["read", "read", "read", "read", "read_write"]);
}
export function layerNormGbPartialKernel(wg: number): KernelBuilder {
  return makeBuilder("layernorm_gb_partial", LAYERNORM_GB_PARTIAL_TEMPLATE, { WG: wg }, ["read", "read", "read", "read_write"]);
}
export function layerNormGbFinalKernel(wg: number): KernelBuilder {
  return makeBuilder("layernorm_gb_final", LAYERNORM_GB_FINAL_TEMPLATE, { WG: wg }, ["read", "read_write", "read_write"]);
}
