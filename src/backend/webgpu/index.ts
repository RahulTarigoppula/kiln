/**
 * WebGPU implementation of the Kiln {@link Backend} contract.
 *
 * Every method enqueues GPU work synchronously into a {@link CommandBatch}; only `read()` and `sync()`
 * are asynchronous. Buffers come from a size-class {@link BufferPool}; per-dispatch parameters travel
 * through a {@link UniformRing}; pipelines are compiled once per (kernel, options, access modes) and
 * cached in a {@link PipelineCache}.
 *
 * Design notes:
 *  - All kernels assume contiguous inputs except `binary` (explicit strides) and `copyStrided`.
 *  - When the same `GPUBuffer` is passed as both an input and an output of one dispatch, the kernel is
 *    specialised with every alias bound as `read_write` so WebGPU usage validation passes.
 *  - `scatterAdd` is a deterministic counting sort + segmented sum (see `kernels/indexing.ts`).
 *  - Tensor rank for strided kernels is limited to 6 after merging contiguous dimensions.
 */

import type { Backend, BinaryOp, Buf, Device, ReduceOp, UnaryOp } from "../types";
import { BufferPool, StagingRing, UniformRing, type PoolStats } from "./bufferpool";
import { CommandBatch } from "./batch";
import { isWebGpuAvailable, requestKilnDevice, type KilnGpuDevice, type WebGpuAdapterInfo, type WebGpuDeviceOptions } from "./device";
import { PipelineCache, type Access, type CompiledKernel } from "./pipeline";
import * as K from "./kernels/index";
import type { KernelBuilder } from "./kernels/common";

export { isWebGpuAvailable } from "./device";
export type { WebGpuAdapterInfo, WebGpuDeviceOptions } from "./device";

/** Buffer handle produced by the WebGPU backend. `byteLength` is the requested size (the GPUBuffer may be larger). */
export class GpuBuf implements Buf {
  readonly device: Device = "webgpu";
  /** Set once `free()` has been called; further use throws. */
  freed = false;
  constructor(readonly handle: GPUBuffer, readonly byteLength: number) {}
}

/** Options for {@link createWebGpuBackend}. */
export interface WebGpuBackendOptions extends WebGpuDeviceOptions {
  /** Dispatches queued before an automatic submit (default 64). */
  batchThreshold?: number;
  /** Called with uncaptured GPU errors (default: console.error). */
  onError?: (message: string) => void;
}

const WORD_BYTES = 4;

function prod(xs: readonly number[]): number {
  let p = 1;
  for (const x of xs) p *= x;
  return p;
}

function contiguousStrides(shape: readonly number[]): number[] {
  const s = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i] ?? 1;
  }
  return s;
}

/**
 * Drop size-1 dims and merge adjacent dims that are contiguous in *every* stride set.
 * Returns rank ≥ 1 (a scalar becomes shape [1]).
 */
export function collapseDims(shape: readonly number[], strideSets: readonly (readonly number[])[]): { shape: number[]; strides: number[][] } {
  const outShape: number[] = [];
  const outStrides: number[][] = strideSets.map(() => []);
  for (let d = 0; d < shape.length; d++) {
    const size = shape[d] ?? 1;
    if (size === 1) continue;
    const last = outShape.length - 1;
    if (last >= 0) {
      const lastSize = outShape[last] ?? 1;
      let mergeable = true;
      for (let k = 0; k < strideSets.length; k++) {
        const prev = outStrides[k]?.[last] ?? 0;
        const cur = strideSets[k]?.[d] ?? 0;
        if (prev !== cur * size) {
          mergeable = false;
          break;
        }
      }
      if (mergeable) {
        outShape[last] = lastSize * size;
        for (let k = 0; k < strideSets.length; k++) {
          const arr = outStrides[k];
          if (arr) arr[last] = strideSets[k]?.[d] ?? 0;
        }
        continue;
      }
    }
    outShape.push(size);
    for (let k = 0; k < strideSets.length; k++) outStrides[k]?.push(strideSets[k]?.[d] ?? 0);
  }
  if (outShape.length === 0) {
    outShape.push(1);
    for (const s of outStrides) s.push(0);
  }
  return { shape: outShape, strides: outStrides };
}

/** Largest divisor of `n` in [2, 128] that leaves at least 8 rows per chunk (1 when none exists, e.g. primes). */
export function pickChunks(n: number): number {
  for (let c = Math.min(128, Math.floor(n / 8)); c >= 2; c--) if (n % c === 0) return c;
  return 1;
}

/** The WebGPU backend. Create with {@link createWebGpuBackend}. */
export class WebGpuBackend implements Backend {
  readonly device: Device = "webgpu";
  /** Adapter description (vendor/architecture/description + timestamp-query support). */
  readonly info: WebGpuAdapterInfo;
  /** The underlying device (exposed for the Bench page's timestamp queries). */
  readonly gpu: GPUDevice;
  readonly adapter: GPUAdapter;
  /** 1-D workgroup size used by elementwise/row kernels. */
  readonly wg: number;
  /** Non-null once the device has been lost. */
  lostInfo: GPUDeviceLostInfo | null = null;
  /** Optional device-lost callback. */
  onDeviceLost: ((info: GPUDeviceLostInfo) => void) | null = null;

  private readonly pool: BufferPool;
  private readonly staging: StagingRing;
  private readonly uniforms: UniformRing;
  private readonly pipelines: PipelineCache;
  private readonly batch: CommandBatch;
  private readonly builders = new Map<string, KernelBuilder>();
  private readonly scratch = new ArrayBuffer(256);
  private readonly u32 = new Uint32Array(this.scratch);
  private readonly i32 = new Int32Array(this.scratch);
  private readonly f32 = new Float32Array(this.scratch);
  private readonly onError: (message: string) => void;

  constructor(dev: KilnGpuDevice, opts: WebGpuBackendOptions = {}) {
    this.gpu = dev.device;
    this.adapter = dev.adapter;
    this.info = dev.info;
    this.onError = opts.onError ?? ((m) => console.error(`[kiln/webgpu] ${m}`));
    const lim = this.gpu.limits;
    this.wg = Math.min(K.DEFAULT_WG, lim.maxComputeInvocationsPerWorkgroup || 256, lim.maxComputeWorkgroupSizeX || 256);
    this.pool = new BufferPool(this.gpu);
    this.staging = new StagingRing(this.gpu);
    this.uniforms = new UniformRing(this.gpu);
    this.pipelines = new PipelineCache(this.gpu);
    this.batch = new CommandBatch(this.gpu, this.uniforms, (b) => this.pool.release(b));
    if (opts.batchThreshold) this.batch.threshold = opts.batchThreshold;
    this.gpu.addEventListener("uncapturederror", (ev: Event) => {
      const e = (ev as GPUUncapturedErrorEvent).error;
      this.onError(e?.message ?? String(e));
    });
    void dev.lost.then((info) => {
      this.lostInfo = info;
      this.onDeviceLost?.(info);
    });
  }

  // ------------------------------------------------------------------ memory

  alloc(byteLength: number): Buf {
    this.assertAlive();
    return new GpuBuf(this.pool.alloc(Math.max(4, byteLength)), byteLength);
  }

  free(buf: Buf): void {
    const g = buf as GpuBuf;
    if (g.device !== "webgpu" || g.freed) return;
    g.freed = true;
    this.batch.deferRelease(g.handle);
  }

  write(buf: Buf, data: ArrayBufferView): void {
    const h = this.h(buf);
    if (data.byteLength === 0) return;
    if (this.batch.isBusy(h)) this.batch.submit(); // keep queue order: pending kernels must see the old contents
    const len = data.byteLength;
    if (len % 4 === 0) {
      this.gpu.queue.writeBuffer(h, 0, data.buffer, data.byteOffset, len);
    } else {
      const padded = new Uint8Array(Math.ceil(len / 4) * 4);
      padded.set(new Uint8Array(data.buffer, data.byteOffset, len));
      this.gpu.queue.writeBuffer(h, 0, padded.buffer, 0, padded.byteLength);
    }
  }

  async read(buf: Buf, byteLength: number): Promise<ArrayBuffer> {
    const h = this.h(buf);
    const n = Math.ceil(byteLength / 4) * 4;
    if (n === 0) return new ArrayBuffer(0);
    const stage = this.staging.acquire(n);
    this.batch.copyBuffer(h, 0, stage, 0, n);
    const done = this.batch.submit();
    await stage.mapAsync(GPUMapMode.READ, 0, n);
    const out = stage.getMappedRange(0, n).slice(0, byteLength);
    stage.unmap();
    this.staging.release(stage);
    await done;
    return out;
  }

  copy(src: Buf, dst: Buf, byteLength: number, srcByteOffset = 0, dstByteOffset = 0): void {
    const s = this.h(src);
    const d = this.h(dst);
    if (byteLength <= 0) return;
    if (s === d) {
      const tmp = this.pool.alloc(byteLength);
      this.batch.copyBuffer(s, srcByteOffset, tmp, 0, byteLength);
      this.batch.copyBuffer(tmp, 0, d, dstByteOffset, byteLength);
      this.batch.deferRelease(tmp);
      return;
    }
    this.batch.copyBuffer(s, srcByteOffset, d, dstByteOffset, byteLength);
  }

  // ------------------------------------------------------------- elementwise

  unary(op: UnaryOp, a: Buf, n: number, out: Buf, alpha = 0, beta = 0): void {
    if (n <= 0) return;
    this.u32[0] = n;
    this.f32[1] = alpha;
    this.f32[2] = beta;
    this.u32[3] = 0;
    this.run(this.builder(`unary_${op}`, () => K.unaryKernel(op, this.wg)), [this.h(a), this.h(out)], 4, this.grid(n));
  }

  binary(op: BinaryOp, a: Buf, b: Buf, out: Buf, outShape: number[], aStrides: number[], bStrides: number[]): void {
    const n = prod(outShape);
    if (n <= 0) return;
    const c = collapseDims(outShape, [aStrides, bStrides]);
    const rank = c.shape.length;
    if (rank > K.MAX_RANK) throw new Error(`binary: rank ${rank} > ${K.MAX_RANK} after collapsing`);
    this.u32.fill(0, 0, 28);
    this.u32[0] = n;
    this.u32[1] = rank;
    for (let d = 0; d < rank; d++) {
      this.u32[4 + d] = c.shape[d] ?? 1;
      this.u32[12 + d] = c.strides[0]?.[d] ?? 0;
      this.u32[20 + d] = c.strides[1]?.[d] ?? 0;
    }
    this.run(this.builder(`binary_${op}`, () => K.binaryKernel(op, this.wg)), [this.h(a), this.h(b), this.h(out)], 28, this.grid(n));
  }

  copyStrided(a: Buf, aShape: number[], aStrides: number[], aOffset: number, out: Buf): void {
    const n = prod(aShape);
    if (n <= 0) return;
    const c = collapseDims(aShape, [aStrides]);
    const rank = c.shape.length;
    if (rank > K.MAX_RANK) throw new Error(`copyStrided: rank ${rank} > ${K.MAX_RANK} after collapsing`);
    this.u32.fill(0, 0, 20);
    this.u32[0] = n;
    this.u32[1] = rank;
    this.i32[2] = aOffset;
    for (let d = 0; d < rank; d++) {
      this.u32[4 + d] = c.shape[d] ?? 1;
      this.i32[12 + d] = c.strides[0]?.[d] ?? 0;
    }
    this.run(this.builder("copy_strided", () => K.copyStridedKernel(this.wg)), [this.h(a), this.h(out)], 20, this.grid(n));
  }

  where(cond: Buf, a: Buf, b: Buf, n: number, out: Buf): void {
    if (n <= 0) return;
    this.u32.fill(0, 0, 4);
    this.u32[0] = n;
    this.run(this.builder("where", () => K.whereKernel(this.wg)), [this.h(cond), this.h(a), this.h(b), this.h(out)], 4, this.grid(n));
  }

  maskedFill(a: Buf, mask: Buf, value: number, n: number, out: Buf): void {
    if (n <= 0) return;
    this.u32.fill(0, 0, 4);
    this.u32[0] = n;
    this.f32[1] = value;
    this.run(this.builder("masked_fill", () => K.maskedFillKernel(this.wg)), [this.h(a), this.h(mask), this.h(out)], 4, this.grid(n));
  }

  fill(buf: Buf, n: number, value: number): void {
    if (n <= 0) return;
    this.u32.fill(0, 0, 4);
    this.u32[0] = n;
    this.f32[1] = value;
    this.run(this.builder("fill", () => K.fillKernel(this.wg)), [this.h(buf)], 4, this.grid(n));
  }

  // ---------------------------------------------------------- linear algebra

  matmul(a: Buf, b: Buf, out: Buf, batch: number, M: number, K_: number, N: number,
         transA: boolean, transB: boolean, aBatchStride: number, bBatchStride: number): void {
    if (batch <= 0 || M <= 0 || N <= 0) return;
    if (batch > K.MAX_WORKGROUPS) throw new Error(`matmul: batch ${batch} exceeds ${K.MAX_WORKGROUPS}`);
    if (K_ <= 0) {
      this.fill(out, batch * M * N, 0);
      return;
    }
    this.u32[0] = M;
    this.u32[1] = K_;
    this.u32[2] = N;
    this.u32[3] = batch;
    this.u32[4] = transA ? 1 : 0;
    this.u32[5] = transB ? 1 : 0;
    this.u32[6] = aBatchStride;
    this.u32[7] = bBatchStride;
    const gx = Math.ceil(N / K.MATMUL_TILE);
    const gy = Math.ceil(M / K.MATMUL_TILE);
    if (gx > K.MAX_WORKGROUPS || gy > K.MAX_WORKGROUPS) throw new Error("matmul: M or N too large for one dispatch");
    this.run(this.builder("matmul_tiled", () => K.matmulKernel()), [this.h(a), this.h(b), this.h(out)], 8, gx, gy, batch);
  }

  // -------------------------------------------------------------- reductions

  reduce(op: ReduceOp, a: Buf, shape: number[], axes: number[], out: Buf): void {
    const strides = contiguousStrides(shape);
    const axisSet = new Set(axes);
    type Group = { reduced: boolean; size: number; stride: number };
    const groups: Group[] = [];
    for (let d = 0; d < shape.length; d++) {
      const size = shape[d] ?? 1;
      if (size === 1) continue;
      const reduced = axisSet.has(d);
      const last = groups[groups.length - 1];
      if (last && last.reduced === reduced) {
        last.size *= size;
        last.stride = strides[d] ?? 1;
      } else {
        groups.push({ reduced, size, stride: strides[d] ?? 1 });
      }
    }
    const kept = groups.filter((g) => !g.reduced);
    const red = groups.filter((g) => g.reduced);
    const outN = prod(kept.map((g) => g.size));
    const redN = prod(red.map((g) => g.size));
    const total = prod(shape);
    if (total === 0 || outN === 0) return;
    if (redN === 1) {
      this.batch.copyBuffer(this.h(a), 0, this.h(out), 0, outN * WORD_BYTES);
      return;
    }
    const lastGroup = groups[groups.length - 1];
    if (red.length === 1 && lastGroup && lastGroup.reduced) {
      this.reduceRows(op, this.h(a), outN, redN, this.h(out));
      return;
    }
    if (kept.length > K.MAX_RANK || red.length > K.MAX_RANK) throw new Error("reduce: too many dimension groups");
    // Leading-axis reductions (bias gradients: [B*T, C] summed over rows) would otherwise run one thread per
    // output column over the whole reduced extent. Split the reduced group into chunks that become an extra
    // kept dimension, then reduce the [chunks, outN] partials: ~chunks times more parallelism, exact same
    // per-element work, only the summation order changes.
    const lead = red[0];
    if (red.length === 1 && lead && lead.size >= 512 && kept.length + 1 <= K.MAX_RANK && outN < 4096) {
      const chunks = pickChunks(lead.size);
      if (chunks > 1) {
        const rowsPerChunk = lead.size / chunks;
        const part = this.pool.alloc(chunks * outN * WORD_BYTES);
        this.reduceGeneral(op, this.h(a), [{ size: chunks, stride: rowsPerChunk * lead.stride }, ...kept], [{ size: rowsPerChunk, stride: lead.stride }], chunks * outN, rowsPerChunk, part);
        this.reduceGeneral(op, part, kept.map((g, i) => ({ size: g.size, stride: kept.slice(i + 1).reduce((s, k) => s * k.size, 1) })), [{ size: chunks, stride: outN }], outN, chunks, this.h(out));
        this.batch.deferRelease(part);
        return;
      }
    }
    this.reduceGeneral(op, this.h(a), kept, red, outN, redN, this.h(out));
  }

  private reduceGeneral(op: ReduceOp, a: GPUBuffer, kept: { size: number; stride: number }[], red: { size: number; stride: number }[], outN: number, redN: number, out: GPUBuffer): void {
    this.u32.fill(0, 0, 36);
    this.u32[0] = outN;
    this.u32[1] = redN;
    this.u32[2] = kept.length;
    this.u32[3] = red.length;
    kept.forEach((g, i) => {
      this.u32[4 + i] = g.size;
      this.u32[12 + i] = g.stride;
    });
    red.forEach((g, i) => {
      this.u32[20 + i] = g.size;
      this.u32[28 + i] = g.stride;
    });
    this.run(this.builder(`reduce_general_${op}`, () => K.reduceGeneralKernel(op, this.wg)), [a, out], 36, this.grid(outN));
  }

  private reduceRows(op: ReduceOp, a: GPUBuffer, rows: number, cols: number, out: GPUBuffer): void {
    this.u32.fill(0, 0, 4);
    this.u32[0] = rows;
    this.u32[1] = cols;
    this.run(this.builder(`reduce_rows_${op}`, () => K.reduceRowsKernel(op, this.wg)), [a, out], 4, this.rowGrid(rows));
  }

  argmax(a: Buf, rows: number, cols: number, out: Buf): void {
    if (rows <= 0 || cols <= 0) return;
    this.u32.fill(0, 0, 4);
    this.u32[0] = rows;
    this.u32[1] = cols;
    this.run(this.builder("argmax_rows", () => K.argmaxKernel(this.wg)), [this.h(a), this.h(out)], 4, this.rowGrid(rows));
  }

  // ------------------------------------------------------- fused row kernels

  softmax(a: Buf, rows: number, cols: number, out: Buf, log: boolean): void {
    if (rows <= 0 || cols <= 0) return;
    this.u32.fill(0, 0, 4);
    this.u32[0] = rows;
    this.u32[1] = cols;
    this.u32[2] = log ? 1 : 0;
    this.run(this.builder("softmax_rows", () => K.softmaxKernel(this.wg)), [this.h(a), this.h(out)], 4, this.rowGrid(rows));
  }

  softmaxBackward(out: Buf, gradOut: Buf, rows: number, cols: number, gradIn: Buf, log: boolean): void {
    if (rows <= 0 || cols <= 0) return;
    this.u32.fill(0, 0, 4);
    this.u32[0] = rows;
    this.u32[1] = cols;
    this.u32[2] = log ? 1 : 0;
    this.run(this.builder("softmax_backward", () => K.softmaxBackwardKernel(this.wg)), [this.h(out), this.h(gradOut), this.h(gradIn)], 4, this.rowGrid(rows));
  }

  layerNorm(x: Buf, gamma: Buf, beta: Buf, rows: number, cols: number, eps: number, out: Buf, meanRstd: Buf): void {
    if (rows <= 0 || cols <= 0) return;
    this.u32.fill(0, 0, 4);
    this.u32[0] = rows;
    this.u32[1] = cols;
    this.f32[2] = eps;
    this.run(
      this.builder("layernorm_rows", () => K.layerNormKernel(this.wg)),
      [this.h(x), this.h(gamma), this.h(beta), this.h(out), this.h(meanRstd)],
      4,
      this.rowGrid(rows),
    );
  }

  layerNormBackward(x: Buf, gamma: Buf, meanRstd: Buf, gradOut: Buf, rows: number, cols: number,
                    gradX: Buf, gradGamma: Buf, gradBeta: Buf): void {
    if (rows <= 0 || cols <= 0) return;
    // 1. gradX
    this.u32.fill(0, 0, 4);
    this.u32[0] = rows;
    this.u32[1] = cols;
    this.run(
      this.builder("layernorm_backward_x", () => K.layerNormBackwardXKernel(this.wg)),
      [this.h(x), this.h(gamma), this.h(meanRstd), this.h(gradOut), this.h(gradX)],
      4,
      this.rowGrid(rows),
    );
    // 2. per-chunk column partials
    const chunks = Math.max(1, Math.min(128, Math.ceil(rows / 32)));
    const rowsPerChunk = Math.ceil(rows / chunks);
    const part = this.pool.alloc(2 * chunks * cols * WORD_BYTES);
    this.u32[0] = rows;
    this.u32[1] = cols;
    this.u32[2] = chunks;
    this.u32[3] = rowsPerChunk;
    this.run(
      this.builder("layernorm_gb_partial", () => K.layerNormGbPartialKernel(this.wg)),
      [this.h(x), this.h(meanRstd), this.h(gradOut), part],
      4,
      Math.ceil(cols / this.wg),
      chunks,
    );
    // 3. final column sums
    this.u32.fill(0, 0, 4);
    this.u32[0] = cols;
    this.u32[1] = chunks;
    this.run(
      this.builder("layernorm_gb_final", () => K.layerNormGbFinalKernel(this.wg)),
      [part, this.h(gradGamma), this.h(gradBeta)],
      4,
      Math.ceil(cols / this.wg),
    );
    this.batch.deferRelease(part);
  }

  crossEntropy(logits: Buf, targets: Buf, rows: number, cols: number, lossOut: Buf, gradOut?: Buf,
               ignoreIndex = -1, validCount?: number): void {
    if (rows <= 0 || cols <= 0) return;
    const denom = Math.max(1, validCount ?? rows); // same as the CPU backend
    this.u32[0] = rows;
    this.u32[1] = cols;
    this.i32[2] = ignoreIndex;
    this.f32[3] = 1 / denom;
    if (gradOut) {
      this.run(
        this.builder("cross_entropy_rows_grad", () => K.crossEntropyKernel(this.wg, true)),
        [this.h(logits), this.h(targets), this.h(lossOut), this.h(gradOut)],
        4,
        this.rowGrid(rows),
      );
    } else {
      this.run(
        this.builder("cross_entropy_rows", () => K.crossEntropyKernel(this.wg, false)),
        [this.h(logits), this.h(targets), this.h(lossOut)],
        4,
        this.rowGrid(rows),
      );
    }
  }

  // ---------------------------------------------------------------- indexing

  gather(table: Buf, ids: Buf, n: number, dim: number, out: Buf): void {
    if (n <= 0 || dim <= 0) return;
    this.u32.fill(0, 0, 4);
    this.u32[0] = n;
    this.u32[1] = dim;
    this.run(this.builder("gather", () => K.gatherKernel(this.wg)), [this.h(table), this.h(ids), this.h(out)], 4, this.grid(n * dim));
  }

  scatterAdd(gradOut: Buf, ids: Buf, n: number, dim: number, gradTable: Buf): void {
    if (n <= 0 || dim <= 0) return;
    const rows = Math.floor(gradTable.byteLength / (dim * WORD_BYTES));
    if (rows <= 0) return;
    const idsH = this.h(ids);
    const counts = this.pool.alloc(rows * WORD_BYTES);
    const offsets = this.pool.alloc((rows + 1) * WORD_BYTES);
    const cursor = this.pool.alloc(rows * WORD_BYTES);
    const perm = this.pool.alloc(n * WORD_BYTES);
    const perm2 = this.pool.alloc(n * WORD_BYTES);
    this.batch.clearBuffer(counts, 0, rows * WORD_BYTES);
    this.batch.clearBuffer(cursor, 0, rows * WORD_BYTES);

    // 1. histogram
    this.u32.fill(0, 0, 4);
    this.u32[0] = n;
    this.u32[1] = rows;
    this.run(this.builder("scatter_hist", () => K.scatterHistKernel(this.wg)), [idsH, counts], 4, this.grid(n));
    // 2. exclusive scan (single workgroup)
    this.u32.fill(0, 0, 4);
    this.u32[0] = rows;
    this.run(this.builder("scan_exclusive", () => K.scanExclusiveKernel(this.wg)), [counts, offsets], 4, 1);
    // 3. place
    this.u32.fill(0, 0, 4);
    this.u32[0] = n;
    this.u32[1] = rows;
    this.run(this.builder("scatter_place", () => K.scatterPlaceKernel(this.wg)), [idsH, offsets, cursor, perm], 4, this.grid(n));
    // 4. deterministic rank within segment
    this.u32.fill(0, 0, 4);
    this.u32[0] = n;
    this.u32[1] = rows;
    this.run(this.builder("scatter_rank", () => K.scatterRankKernel(this.wg)), [idsH, offsets, perm, perm2], 4, this.grid(n));
    // 5. segmented sum into the table
    this.u32.fill(0, 0, 4);
    this.u32[0] = n;
    this.u32[1] = rows;
    this.u32[2] = dim;
    this.run(
      this.builder("scatter_segsum", () => K.scatterSegsumKernel()),
      [offsets, perm2, this.h(gradOut), this.h(gradTable)],
      4,
      Math.ceil(dim / K.SEGSUM_WGX),
      Math.min(rows, K.MAX_WORKGROUPS),
    );
    for (const b of [counts, offsets, cursor, perm, perm2]) this.batch.deferRelease(b);
  }

  // --------------------------------------------------------------- optimiser

  adamw(param: Buf, grad: Buf, m: Buf, v: Buf, n: number, lr: number, beta1: number, beta2: number,
        eps: number, weightDecay: number, step: number): void {
    if (n <= 0) return;
    this.u32[0] = n;
    this.f32[1] = lr;
    this.f32[2] = beta1;
    this.f32[3] = beta2;
    this.f32[4] = eps;
    this.f32[5] = weightDecay;
    this.f32[6] = 1 - Math.pow(beta1, step);
    this.f32[7] = Math.sqrt(1 - Math.pow(beta2, step));
    this.run(this.builder("adamw", () => K.adamwKernel(this.wg)), [this.h(param), this.h(grad), this.h(m), this.h(v)], 8, this.grid(n));
  }

  sumSquares(a: Buf, n: number, out: Buf): void {
    if (n <= 0) {
      this.fill(out, 1, 0);
      return;
    }
    const nWG = Math.max(1, Math.min(1024, Math.ceil(n / (this.wg * 4))));
    const part = this.pool.alloc(nWG * WORD_BYTES);
    this.u32.fill(0, 0, 4);
    this.u32[0] = n;
    this.run(this.builder("sum_squares_partial", () => K.sumSquaresKernel(this.wg)), [this.h(a), part], 4, nWG);
    this.reduceRows("sum", part, 1, nWG, this.h(out));
    this.batch.deferRelease(part);
  }

  // -------------------------------------------------------------- randomness

  randn(buf: Buf, n: number, seed: number): void {
    this.rng("randn", buf, n, seed, 0);
  }
  rand(buf: Buf, n: number, seed: number): void {
    this.rng("rand", buf, n, seed, 0);
  }
  dropoutMask(mask: Buf, n: number, p: number, seed: number): void {
    this.rng("dropoutMask", mask, n, seed, p);
  }

  private rng(mode: K.RngMode, buf: Buf, n: number, seed: number, p: number): void {
    if (n <= 0) return;
    this.u32[0] = n;
    this.u32[1] = seed >>> 0;
    this.f32[2] = p;
    this.u32[3] = 0;
    this.run(this.builder(`rng_${mode}`, () => K.rngKernel(mode, this.wg)), [this.h(buf)], 4, this.grid(n));
  }

  // -------------------------------------------------------------------- misc

  /** Submit all recorded work and wait for the GPU to finish it. */
  async sync(): Promise<void> {
    await this.batch.sync();
  }

  /** Submit recorded work without waiting (useful before long host-side work). */
  flush(): void {
    this.batch.submit();
  }

  /** Diagnostics: pool usage, compiled pipelines, submits. */
  stats(): PoolStats & { pipelines: number; submits: number; pendingDispatches: number } {
    return { ...this.pool.stats(), pipelines: this.pipelines.size, submits: this.batch.submits, pendingDispatches: this.batch.pending };
  }

  /** Release pooled GPU memory and destroy the device. The backend is unusable afterwards. */
  destroy(): void {
    this.pool.drain();
    this.staging.drain();
    this.uniforms.destroy();
    this.gpu.destroy();
  }

  // ----------------------------------------------------------------- helpers

  private assertAlive(): void {
    if (this.lostInfo) throw new Error(`WebGPU device lost: ${this.lostInfo.message}`);
  }

  private h(buf: Buf): GPUBuffer {
    const g = buf as GpuBuf;
    if (g.device !== "webgpu") throw new Error("WebGpuBackend: buffer belongs to another device");
    if (g.freed) throw new Error("WebGpuBackend: use after free");
    return g.handle;
  }

  private grid(n: number): number {
    return Math.max(1, Math.min(K.MAX_WORKGROUPS, Math.ceil(n / this.wg)));
  }

  private rowGrid(rows: number): number {
    return Math.max(1, Math.min(K.MAX_WORKGROUPS, rows));
  }

  private builder(name: string, make: () => KernelBuilder): KernelBuilder {
    let b = this.builders.get(name);
    if (!b) {
      b = make();
      this.builders.set(name, b);
    }
    return b;
  }

  /** Promote aliased bindings to read_write so one buffer can be bound as both input and output. */
  private resolveAccess(def: readonly Access[], bufs: readonly GPUBuffer[]): readonly Access[] {
    let aliased = false;
    for (let i = 0; i < bufs.length && !aliased; i++) {
      for (let j = i + 1; j < bufs.length; j++) {
        if (bufs[i] === bufs[j] && def[i] !== def[j]) {
          aliased = true;
          break;
        }
      }
    }
    if (!aliased) return def;
    const out = [...def];
    for (let i = 0; i < bufs.length; i++) {
      for (let j = 0; j < bufs.length; j++) {
        if (i !== j && bufs[i] === bufs[j]) {
          out[i] = "read_write";
          out[j] = "read_write";
        }
      }
    }
    return out;
  }

  /** Upload the first `words` of the scratch params, build the bind group and record the dispatch. */
  private run(builder: KernelBuilder, bufs: GPUBuffer[], words: number, x: number, y = 1, z = 1): void {
    this.assertAlive();
    const access = this.resolveAccess(builder.defaultAccess, bufs);
    const kernel: CompiledKernel = this.pipelines.getOrBuild(builder.key(access), () => builder.build(access));
    const slice = this.batch.uniformSlice(this.scratch, words * WORD_BYTES);
    const bg = this.pipelines.bindGroup(kernel, slice, bufs);
    this.batch.dispatch(kernel.pipeline, bg, x, y, z, bufs);
  }
}

/**
 * Request a WebGPU device and wrap it in a {@link WebGpuBackend}.
 * Rejects when WebGPU is unavailable; check {@link isWebGpuAvailable} first for a soft fallback.
 */
export async function createWebGpuBackend(opts: WebGpuBackendOptions = {}): Promise<WebGpuBackend> {
  if (!isWebGpuAvailable()) throw new Error("WebGPU is not available (navigator.gpu is undefined).");
  const devOpts: WebGpuDeviceOptions = {};
  if (opts.powerPreference) devOpts.powerPreference = opts.powerPreference;
  if (opts.requiredLimits) devOpts.requiredLimits = opts.requiredLimits;
  const dev = await requestKilnDevice(devOpts);
  return new WebGpuBackend(dev, opts);
}
