/**
 * Fused cross-entropy rows kernel and the fused AdamW step.
 *
 * `cross_entropy_rows`: one workgroup per row; online log-sum-exp; loss = lse - logit[target]; when the
 * grad variant is built, grad = (softmax - onehot) * invValid. Targets equal to `ignoreIndex` (or out of
 * range) give 0 loss and 0 grad.
 *
 * Uniform layouts (words):
 *  ce:    [rows, cols, ignoreIndex(i32), invValid(f32)]
 *  adamw: [n, lr, beta1, beta2, eps, wd, bc1 = 1-β1^t, bc2sqrt = sqrt(1-β2^t)] (all f32 except n)
 */

import { F32_LOWEST, WGSL_WG_REDUCE, makeBuilder, type KernelBuilder } from "./common";

function crossEntropyTemplate(withGrad: boolean): string {
  const gradBinding = withGrad ? `@group(0) @binding(4) var<storage, {{A3}}> Grad: array<f32>;` : "";
  const gradBody = withGrad
    ? `
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) {
      var g = 0.0;
      if (valid) {
        let p = exp(L[base + c] - lse);
        g = (p - select(0.0, 1.0, i32(c) == tgt)) * P.invValid;
      }
      Grad[base + c] = g;
    }`
    : "";
  return /* wgsl */ `
struct Params { rows: u32, cols: u32, ignoreIndex: i32, invValid: f32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> L: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> Tg: array<i32>;
@group(0) @binding(3) var<storage, {{A2}}> Loss: array<f32>;
${gradBinding}
${WGSL_WG_REDUCE}
@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  for (var row = wid.x; row < P.rows; row = row + nwg.x) {
    let base = row * P.cols;
    var m = ${F32_LOWEST};
    var s = 0.0;
    for (var c = lid.x; c < P.cols; c = c + {{WG}}u) {
      let x = L[base + c];
      if (x > m) {
        s = s * exp(m - x) + 1.0;
        m = x;
      } else {
        s = s + exp(x - m);
      }
    }
    let ms = wg_online_softmax(lid.x, m, s);
    let lse = ms.x + log(ms.y);
    let tgt = Tg[row];
    let valid = (tgt != P.ignoreIndex) && (tgt >= 0) && (tgt < i32(P.cols));
    if (lid.x == 0u) {
      var loss = 0.0;
      if (valid) { loss = lse - L[base + u32(tgt)]; }
      Loss[row] = loss;
    }
    ${gradBody}
  }
}
`;
}

/** Cross-entropy template without the gradient output. */
export const CROSS_ENTROPY_TEMPLATE = crossEntropyTemplate(false);
/** Cross-entropy template that also writes d(mean loss)/d(logits). */
export const CROSS_ENTROPY_GRAD_TEMPLATE = crossEntropyTemplate(true);

export const ADAMW_TEMPLATE = /* wgsl */ `
struct Params { n: u32, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, bc1: f32, bc2sqrt: f32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> Param: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> Grad: array<f32>;
@group(0) @binding(3) var<storage, {{A2}}> M: array<f32>;
@group(0) @binding(4) var<storage, {{A3}}> V: array<f32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    let g = Grad[i];
    var p = Param[i] * (1.0 - P.lr * P.wd);
    let m = P.beta1 * M[i] + (1.0 - P.beta1) * g;
    let v = P.beta2 * V[i] + (1.0 - P.beta2) * g * g;
    M[i] = m;
    V[i] = v;
    let denom = sqrt(v) / P.bc2sqrt + P.eps;
    p = p - (P.lr / P.bc1) * m / denom;
    Param[i] = p;
  }
}
`;

export function crossEntropyKernel(wg: number, withGrad: boolean): KernelBuilder {
  return withGrad
    ? makeBuilder("cross_entropy_rows_grad", CROSS_ENTROPY_GRAD_TEMPLATE, { WG: wg }, ["read", "read", "read_write", "read_write"])
    : makeBuilder("cross_entropy_rows", CROSS_ENTROPY_TEMPLATE, { WG: wg }, ["read", "read", "read_write"]);
}
export function adamwKernel(wg: number): KernelBuilder {
  return makeBuilder("adamw", ADAMW_TEMPLATE, { WG: wg }, ["read_write", "read", "read_write", "read_write"]);
}
