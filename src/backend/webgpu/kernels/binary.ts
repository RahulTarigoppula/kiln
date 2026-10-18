/**
 * Broadcasting binary kernel (rank ≤ 6). The output index is decomposed over `shape`; input offsets are
 * `Σ idx[d] * strides[d]` with stride 0 for broadcast dimensions. Comparisons write 1.0 / 0.0.
 *
 * Uniform layout (words): [n, rank, 0, 0, shape(8), aStrides(8), bStrides(8)] — 28 words.
 */

import type { BinaryOp } from "../../types";
import { WGSL_MATH, makeBuilder, type KernelBuilder } from "./common";

/** Expression per binary op in terms of `a`, `b`. */
export const BINARY_EXPR: Record<BinaryOp, string> = {
  add: "a + b",
  sub: "a - b",
  mul: "a * b",
  div: "a / b",
  pow: "kiln_pow(a, b)",
  max: "max(a, b)",
  min: "min(a, b)",
  eq: "select(0.0, 1.0, a == b)",
  lt: "select(0.0, 1.0, a < b)",
  gt: "select(0.0, 1.0, a > b)",
  le: "select(0.0, 1.0, a <= b)",
  ge: "select(0.0, 1.0, a >= b)",
};

/** WGSL template for the broadcasting binary kernel. */
export const BINARY_TEMPLATE = /* wgsl */ `
struct Params {
  n: u32, rank: u32, pad0: u32, pad1: u32,
  shape: array<vec4<u32>, 2>,
  sa: array<vec4<u32>, 2>,
  sb: array<vec4<u32>, 2>,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> B: array<f32>;
@group(0) @binding(3) var<storage, {{A2}}> O: array<f32>;
${WGSL_MATH}
fn op(a: f32, b: f32) -> f32 { return {{EXPR}}; }
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    var rem = i;
    var oa = 0u;
    var ob = 0u;
    for (var d = P.rank; d > 0u; d = d - 1u) {
      let dd = d - 1u;
      let s = P.shape[dd >> 2u][dd & 3u];
      let idx = rem % s;
      rem = rem / s;
      oa = oa + idx * P.sa[dd >> 2u][dd & 3u];
      ob = ob + idx * P.sb[dd >> 2u][dd & 3u];
    }
    O[i] = op(A[oa], B[ob]);
  }
}
`;

/** Build the broadcasting binary kernel for `op`. */
export function binaryKernel(op: BinaryOp, wg: number): KernelBuilder {
  const expr = BINARY_EXPR[op];
  if (!expr) throw new Error(`binary: unknown op ${op}`);
  return makeBuilder(`binary_${op}`, BINARY_TEMPLATE, { WG: wg, EXPR: expr }, ["read", "read", "read_write"]);
}
