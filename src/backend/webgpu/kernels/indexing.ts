/**
 * Embedding gather and deterministic scatter-add.
 *
 * `gather`: out[i, :] = table[ids[i], :].
 *
 * `scatter_add` (gradTable[ids[i], :] += gradOut[i, :]) cannot use f32 atomics (not in WGSL), so it is a
 * counting sort over the ids followed by a deterministic segmented sum — exact and run-to-run stable:
 *  1. `scatter_hist`:   counts[id] += 1            (atomicAdd on u32)
 *  2. `scan_exclusive`: offsets = exclusive scan(counts), offsets[rows] = n   (single workgroup)
 *  3. `scatter_place`:  perm[offsets[id] + atomicAdd(cursor[id], 1)] = i      (arbitrary order inside a segment)
 *  4. `scatter_rank`:   perm2[offsets[id] + #{p in segment : perm[p] < i}] = i  (sorts each segment ascending)
 *  5. `scatter_segsum`: for each (row r, column c): gradTable[r, c] += Σ_{p in segment(r)} gradOut[perm2[p], c]
 *     iterating p in ascending order, so the f32 summation order is fixed.
 *
 * Uniform layouts (words):
 *  gather:   [n, dim, 0, 0]
 *  hist:     [n, rows, 0, 0]
 *  scan:     [n(=rows), 0, 0, 0]
 *  place:    [n, rows, 0, 0]
 *  rank:     [n, rows, 0, 0]
 *  segsum:   [n, rows, dim, 0]
 */

import { makeBuilder, type KernelBuilder } from "./common";

export const GATHER_TEMPLATE = /* wgsl */ `
struct Params { n: u32, dim: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> T: array<f32>;
@group(0) @binding(2) var<storage, {{A1}}> Ids: array<i32>;
@group(0) @binding(3) var<storage, {{A2}}> O: array<f32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  let total = P.n * P.dim;
  for (var idx = gid.x; idx < total; idx = idx + stride) {
    let i = idx / P.dim;
    let j = idx - i * P.dim;
    let id = u32(max(Ids[i], 0));
    O[idx] = T[id * P.dim + j];
  }
}
`;

export const SCATTER_HIST_TEMPLATE = /* wgsl */ `
struct Params { n: u32, rows: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> Ids: array<i32>;
@group(0) @binding(2) var<storage, {{A1}}> Counts: array<atomic<u32>>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    let id = Ids[i];
    if (id >= 0 && u32(id) < P.rows) { atomicAdd(&Counts[u32(id)], 1u); }
  }
}
`;

/** Single-workgroup exclusive scan of `n` u32 counts; writes n+1 outputs (last = total). */
export const SCAN_EXCLUSIVE_TEMPLATE = /* wgsl */ `
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> In: array<u32>;
@group(0) @binding(2) var<storage, {{A1}}> Out: array<u32>;
var<workgroup> sh: array<u32, {{WG}}>;
@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let chunk = (P.n + {{WG}}u - 1u) / {{WG}}u;
  let s0 = t * chunk;
  let s1 = min(P.n, s0 + chunk);
  var local = 0u;
  for (var i = s0; i < s1; i = i + 1u) { local = local + In[i]; }
  sh[t] = local;
  workgroupBarrier();
  for (var off = 1u; off < {{WG}}u; off = off << 1u) {
    var v = 0u;
    if (t >= off) { v = sh[t - off]; }
    workgroupBarrier();
    sh[t] = sh[t] + v;
    workgroupBarrier();
  }
  var running = sh[t] - local;
  for (var i = s0; i < s1; i = i + 1u) {
    Out[i] = running;
    running = running + In[i];
  }
  if (t == {{WG}}u - 1u) { Out[P.n] = sh[t]; }
}
`;

export const SCATTER_PLACE_TEMPLATE = /* wgsl */ `
struct Params { n: u32, rows: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> Ids: array<i32>;
@group(0) @binding(2) var<storage, {{A1}}> Offs: array<u32>;
@group(0) @binding(3) var<storage, {{A2}}> Cursor: array<atomic<u32>>;
@group(0) @binding(4) var<storage, {{A3}}> Perm: array<u32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    let id = Ids[i];
    if (id >= 0 && u32(id) < P.rows) {
      let k = atomicAdd(&Cursor[u32(id)], 1u);
      Perm[Offs[u32(id)] + k] = i;
    }
  }
}
`;

export const SCATTER_RANK_TEMPLATE = /* wgsl */ `
struct Params { n: u32, rows: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> Ids: array<i32>;
@group(0) @binding(2) var<storage, {{A1}}> Offs: array<u32>;
@group(0) @binding(3) var<storage, {{A2}}> Perm: array<u32>;
@group(0) @binding(4) var<storage, {{A3}}> Perm2: array<u32>;
@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * {{WG}}u;
  for (var i = gid.x; i < P.n; i = i + stride) {
    let id = Ids[i];
    if (id >= 0 && u32(id) < P.rows) {
      let s = Offs[u32(id)];
      let e = Offs[u32(id) + 1u];
      var rank = 0u;
      for (var p = s; p < e; p = p + 1u) {
        if (Perm[p] < i) { rank = rank + 1u; }
      }
      Perm2[s + rank] = i;
    }
  }
}
`;

/** Workgroup (64,1,1): x over columns, y over table rows (strided when rows > 65535). */
export const SCATTER_SEGSUM_TEMPLATE = /* wgsl */ `
struct Params { n: u32, rows: u32, dim: u32, pad: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, {{A0}}> Offs: array<u32>;
@group(0) @binding(2) var<storage, {{A1}}> Perm2: array<u32>;
@group(0) @binding(3) var<storage, {{A2}}> G: array<f32>;
@group(0) @binding(4) var<storage, {{A3}}> T: array<f32>;
@compute @workgroup_size({{WGX}}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c = gid.x;
  if (c >= P.dim) { return; }
  for (var r = wid.y; r < P.rows; r = r + nwg.y) {
    let s = Offs[r];
    let e = Offs[r + 1u];
    if (e > s) {
      var acc = 0.0;
      for (var p = s; p < e; p = p + 1u) {
        acc = acc + G[Perm2[p] * P.dim + c];
      }
      T[r * P.dim + c] = T[r * P.dim + c] + acc;
    }
  }
}
`;

/** Workgroup width (columns) for the segmented-sum kernel. */
export const SEGSUM_WGX = 64;

export function gatherKernel(wg: number): KernelBuilder {
  return makeBuilder("gather", GATHER_TEMPLATE, { WG: wg }, ["read", "read", "read_write"]);
}
export function scatterHistKernel(wg: number): KernelBuilder {
  return makeBuilder("scatter_hist", SCATTER_HIST_TEMPLATE, { WG: wg }, ["read", "read_write"]);
}
export function scanExclusiveKernel(wg: number): KernelBuilder {
  return makeBuilder("scan_exclusive", SCAN_EXCLUSIVE_TEMPLATE, { WG: wg }, ["read", "read_write"]);
}
export function scatterPlaceKernel(wg: number): KernelBuilder {
  return makeBuilder("scatter_place", SCATTER_PLACE_TEMPLATE, { WG: wg }, ["read", "read", "read_write", "read_write"]);
}
export function scatterRankKernel(wg: number): KernelBuilder {
  return makeBuilder("scatter_rank", SCATTER_RANK_TEMPLATE, { WG: wg }, ["read", "read", "read", "read_write"]);
}
export function scatterSegsumKernel(): KernelBuilder {
  return makeBuilder("scatter_segsum", SCATTER_SEGSUM_TEMPLATE, { WGX: SEGSUM_WGX }, ["read", "read", "read", "read_write"]);
}
