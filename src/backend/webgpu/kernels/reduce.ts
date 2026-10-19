/**
 * Reductions.
 *
 * - `reduce_rows`: fast path for reducing a trailing contiguous block - one workgroup per row, each lane
 *   strides over the columns, then a shared-memory tree reduction (deterministic order).
 * - `reduce_general`: one thread per output element iterating the reduced index space (rank ≤ 6 for
 *   both kept and reduced dims after host-side dimension merging).
 * - `argmax_rows`: row-wise argmax (first index on ties) -> i32.
 * - `sum_squares_partial`: stage 1 of the two-stage sum of squares (stage 2 reuses `reduce_rows`).
 *
 * Uniform layouts (words):
 *  rows:    [rows, cols, 0, 0]
 *  general: [outN, redN, oRank, rRank, oShape(8), oStride(8), rShape(8), rStride(8)] - 36 words
 *  argmax:  [rows, cols, 0, 0]
 *  sumsq:   [n, 0, 0, 0]
 */

import type { ReduceOp } from "../../types";
import { WGSL_WG_REDUCE, makeBuilder, type KernelBuilder } from "./common";

const COMBINE: Record<ReduceOp, string> = {
  sum: "acc + v",
  max: "max(acc, v)",
  min: "min(acc, v)",
};
/** Identity for `sum`; for max/min the first element of the row is used instead (always valid). */
const INIT: Record<ReduceOp, string> = {
  sum: "0.0",
  max: "A[base]",
  min: "A[base]",
};
const WG_FN: Record<ReduceOp, string> = { sum: "wg_sum", max: "wg_max", min: "wg_min" };

/** Trailing-block reduction template. */
export const REDUCE_ROWS_TEMPLATE = /* wgsl */ `
struct Params { rows: u32, cols: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> O: array<f32>;
${WGSL_WG_REDUCE}
@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  for (var row = wid.x; row < P.rows; row = row + nwg.x) {
    let base = row * P.cols;
    var acc = {{INIT}};
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) {
      let v = A[base + c];
      acc = {{COMBINE}};
    }
    let r = {{WGFN}}(lid.x, acc);
    if (lid.x == 0u) { O[row] = r; }
  }
}
`;

/** General strided reduction template. */
export const REDUCE_GENERAL_TEMPLATE = /* wgsl */ `
struct Params {
  outN: u32, redN: u32, oRank: u32, rRank: u32,
  oShape: array<vec4<u32>, 2>,
  oStride: array<vec4<u32>, 2>,
  rShape: array<vec4<u32>, 2>,
  rStride: array<vec4<u32>, 2>,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> O: array<f32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var o = gid.x; o < P.outN; o = o + stride) {
    var rem = o;
    var base = 0u;
    for (var d = P.oRank; d > 0u; d = d - 1u) {
      let dd = d - 1u;
      let s = P.oShape[dd >> 2u][dd & 3u];
      let idx = rem % s;
      rem = rem / s;
      base = base + idx * P.oStride[dd >> 2u][dd & 3u];
    }
    var acc = {{INIT}};
    for (var r = 0u; r < P.redN; r = r + 1u) {
      var rr = r;
      var off = base;
      for (var d = P.rRank; d > 0u; d = d - 1u) {
        let dd = d - 1u;
        let s = P.rShape[dd >> 2u][dd & 3u];
        let idx = rr % s;
        rr = rr / s;
        off = off + idx * P.rStride[dd >> 2u][dd & 3u];
      }
      let v = A[off];
      acc = {{COMBINE}};
    }
    O[o] = acc;
  }
}
`;

/** Row-wise argmax template (first max index on ties). */
export const ARGMAX_TEMPLATE = /* wgsl */ `
struct Params { rows: u32, cols: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> O: array<i32>;
${WGSL_WG_REDUCE}
@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  for (var row = wid.x; row < P.rows; row = row + nwg.x) {
    let base = row * P.cols;
    var bv = A[base];
    var bi = 0u;
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) {
      let v = A[base + c];
      if (v > bv || (v == bv && c < bi)) { bv = v; bi = c; }
    }
    kiln_sh[lid.x] = bv;
    kiln_shi[lid.x] = bi;
    workgroupBarrier();
    for (var s = {{WG}}u / 2u; s > 0u; s = s >> 1u) {
      if (lid.x < s) {
        let v2 = kiln_sh[lid.x + s];
        let i2 = kiln_shi[lid.x + s];
        let v1 = kiln_sh[lid.x];
        let i1 = kiln_shi[lid.x];
        if (v2 > v1 || (v2 == v1 && i2 < i1)) { kiln_sh[lid.x] = v2; kiln_shi[lid.x] = i2; }
      }
      workgroupBarrier();
    }
    if (lid.x == 0u) { O[row] = i32(kiln_shi[0]); }
    workgroupBarrier();
  }
}
`;

/** Stage 1 of sum of squares: each workgroup writes one partial. */
export const SUM_SQUARES_TEMPLATE = /* wgsl */ `
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> Part: array<f32>;
${WGSL_WG_REDUCE}
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  var acc = 0.0;
  for (var i = gid.x; i < P.n; i = i + stride) {
    let v = A[i];
    acc = acc + v * v;
  }
  let r = wg_sum(lid.x, acc);
  if (lid.x == 0u) { Part[wid.x] = r; }
}
`;

export function reduceRowsKernel(op: ReduceOp, wg: number): KernelBuilder {
  return makeBuilder(
    `reduce_rows_${op}`,
    REDUCE_ROWS_TEMPLATE,
    { WG: wg, INIT: INIT[op], COMBINE: COMBINE[op], WGFN: WG_FN[op] },
    ["read", "read_write"],
  );
}
export function reduceGeneralKernel(op: ReduceOp, wg: number): KernelBuilder {
  return makeBuilder(
    `reduce_general_${op}`,
    REDUCE_GENERAL_TEMPLATE,
    { WG: wg, INIT: INIT[op], COMBINE: COMBINE[op] },
    ["read", "read_write"],
  );
}
export function argmaxKernel(wg: number): KernelBuilder {
  return makeBuilder("argmax_rows", ARGMAX_TEMPLATE, { WG: wg }, ["read", "read_write"]);
}
export function sumSquaresKernel(wg: number): KernelBuilder {
  return makeBuilder("sum_squares_partial", SUM_SQUARES_TEMPLATE, { WG: wg }, ["read", "read_write"]);
}
