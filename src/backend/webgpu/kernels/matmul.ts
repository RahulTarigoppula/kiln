/**
 * Tiled f32 matmul: `out[b] = opA(a[b]) · opB(b[b])`.
 *
 * - 64x64 output tile per workgroup, BK = 16, 16x16 threads each owning a 4x4 register block
 *   (rows ty + 16·i, cols tx + 16·j so shared-memory reads are broadcast/coalesced).
 * - transA/transB handled by index math on the global loads; the shared tile layout is always
 *   `As[k][m]`, `Bs[k][n]`, and the load mapping is chosen so global reads are contiguous either way.
 * - Edges (M/K/N not multiples of the tile) are zero-padded on load and masked on store.
 * - Batch via workgroup z; a batch stride of 0 broadcasts the single matrix.
 *
 * Uniform layout (words): [M, K, N, batch, transA, transB, aBatchStride, bBatchStride].
 */

import { makeBuilder, type KernelBuilder } from "./common";

/** Output tile edge (both M and N). */
export const MATMUL_TILE = 64;
/** K-depth per shared-memory tile. */
export const MATMUL_BK = 16;

/** WGSL template for the tiled matmul. */
export const MATMUL_TEMPLATE = /* wgsl */ `
struct Params { M: u32, K: u32, N: u32, batch: u32, transA: u32, transB: u32, aBatchStride: u32, bBatchStride: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> A: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> B: array<f32>;
@group(0) @binding(3) var<storage, {{A2}}> O: array<f32>;

const BM = {{TILE}}u;
const BN = {{TILE}}u;
const BK = {{BK}}u;
const THREADS = 256u;
const LOADS = (BM * BK) / THREADS;

var<workgroup> As: array<f32, BK * BM>;
var<workgroup> Bs: array<f32, BK * BN>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tx = lid.x;
  let ty = lid.y;
  let t = ty * 16u + tx;
  let m0 = wid.y * BM;
  let n0 = wid.x * BN;
  let bz = wid.z;
  let aBase = bz * P.aBatchStride;
  let bBase = bz * P.bBatchStride;

  var acc: array<vec4<f32>, 4>;
  for (var i = 0u; i < 4u; i = i + 1u) { acc[i] = vec4<f32>(0.0); }

  let numTiles = (P.K + BK - 1u) / BK;
  for (var tile = 0u; tile < numTiles; tile = tile + 1u) {
    let k0 = tile * BK;
    for (var r = 0u; r < LOADS; r = r + 1u) {
      let e = t + r * THREADS;
      var row = 0u;
      var kk = 0u;
      if (P.transA == 0u) { row = e / BK; kk = e % BK; } else { kk = e / BM; row = e % BM; }
      let m = m0 + row;
      let k = k0 + kk;
      var v = 0.0;
      if (m < P.M && k < P.K) {
        if (P.transA == 0u) { v = A[aBase + m * P.K + k]; } else { v = A[aBase + k * P.M + m]; }
      }
      As[kk * BM + row] = v;
    }
    for (var r = 0u; r < LOADS; r = r + 1u) {
      let e = t + r * THREADS;
      var col = 0u;
      var kk = 0u;
      if (P.transB == 0u) { kk = e / BN; col = e % BN; } else { col = e / BK; kk = e % BK; }
      let n = n0 + col;
      let k = k0 + kk;
      var v = 0.0;
      if (n < P.N && k < P.K) {
        if (P.transB == 0u) { v = B[bBase + k * P.N + n]; } else { v = B[bBase + n * P.K + k]; }
      }
      Bs[kk * BN + col] = v;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk = kk + 1u) {
      let bv = vec4<f32>(Bs[kk * BN + tx], Bs[kk * BN + tx + 16u], Bs[kk * BN + tx + 32u], Bs[kk * BN + tx + 48u]);
      for (var i = 0u; i < 4u; i = i + 1u) {
        let av = As[kk * BM + ty + i * 16u];
        acc[i] = acc[i] + av * bv;
      }
    }
    workgroupBarrier();
  }

  let oBase = bz * P.M * P.N;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let m = m0 + ty + i * 16u;
    if (m < P.M) {
      for (var j = 0u; j < 4u; j = j + 1u) {
        let n = n0 + tx + j * 16u;
        if (n < P.N) { O[oBase + m * P.N + n] = acc[i][j]; }
      }
    }
  }
}
`;

/** Build the tiled matmul kernel (fixed 64x64x16 tiles, 256 threads). */
export function matmulKernel(): KernelBuilder {
  return makeBuilder("matmul_tiled", MATMUL_TEMPLATE, { TILE: MATMUL_TILE, BK: MATMUL_BK }, ["read", "read", "read_write"]);
}
