/**
 * Elementwise unary kernel: `out[i] = op(a[i])` with optional scalars `alpha`/`beta` in the uniform.
 *
 * Backward-helper semantics (match `src/backend/cpu/elementwise.ts`): every *Grad op takes the forward
 * *input* x - `tanhGrad(x) = 1 - tanh(x)²`, `sigmoidGrad(x) = σ(x)(1 - σ(x))`, `geluGrad(x)`, `reluGrad(x)`.
 *
 * Uniform layout (words): [n: u32, alpha: f32, beta: f32, pad].
 */

import type { UnaryOp } from "../../types";
import { WGSL_MATH, makeBuilder, type KernelBuilder } from "./common";

/** Expression for each unary op, in terms of `x`, `P.alpha`, `P.beta`. */
export const UNARY_EXPR: Record<UnaryOp, string> = {
  neg: "-x",
  exp: "exp(x)",
  log: "log(x)",
  sqrt: "sqrt(x)",
  rsqrt: "inverseSqrt(x)",
  tanh: "kiln_tanh(x)",
  sigmoid: "kiln_sigmoid(x)",
  relu: "max(x, 0.0)",
  gelu: "kiln_gelu(x)",
  square: "x * x",
  abs: "abs(x)",
  recip: "1.0 / x",
  sin: "sin(x)",
  cos: "cos(x)",
  scale: "x * P.alpha",
  addScalar: "x + P.alpha",
  powScalar: "kiln_pow(x, P.alpha)",
  geluGrad: "kiln_gelu_grad(x)",
  tanhGrad: "kiln_tanh_grad(x)",
  sigmoidGrad: "kiln_sigmoid_grad(x)",
  reluGrad: "select(0.0, 1.0, x > 0.0)",
  clamp: "min(max(x, P.alpha), P.beta)",
};

/** WGSL template for the unary kernel. */
export const UNARY_TEMPLATE = /* wgsl */ `
struct Params { n: u32, alpha: f32, beta: f32, pad: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> O: array<f32>;
${WGSL_MATH}
fn op(x: f32) -> f32 { return {{EXPR}}; }
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    O[i] = op(A[i]);
  }
}
`;

/** Build the unary kernel for `op`. */
export function unaryKernel(op: UnaryOp, wg: number): KernelBuilder {
  const expr = UNARY_EXPR[op];
  if (!expr) throw new Error(`unary: unknown op ${op}`);
  return makeBuilder(`unary_${op}`, UNARY_TEMPLATE, { WG: wg, EXPR: expr }, ["read", "read_write"]);
}
