/**
 * Backend contract shared by the CPU and WebGPU backends.
 * Autograd ops are written once against this interface. Keep it dependency-free.
 *
 * Conventions:
 *  - All inputs are contiguous row-major buffers unless an explicit strides argument exists.
 *  - Shapes/strides are plain number arrays (element units, not bytes).
 *  - Methods are synchronous: CPU computes immediately, WebGPU enqueues work. Only `read`/`sync` are async.
 *  - `Buf` is an opaque handle owned by the backend. Element dtype is tracked by the Tensor, not the Buf,
 *    except where a kernel needs to know (ids are always i32, masks are u32 0/1).
 */

export type Device = "cpu" | "webgpu";
export type DType = "f32" | "i32" | "u32";

export interface Buf {
  readonly byteLength: number;
  readonly device: Device;
  /** Backend-specific payload (Float32Array/Int32Array-backed ArrayBuffer on CPU, GPUBuffer on WebGPU). */
  readonly handle: unknown;
}

export type UnaryOp =
  | "neg" | "exp" | "log" | "sqrt" | "rsqrt" | "tanh" | "sigmoid" | "relu" | "gelu" | "square" | "abs" | "recip"
  | "sin" | "cos"
  /** out = a * alpha */ | "scale"
  /** out = a + alpha */ | "addScalar"
  /** out = a ^ alpha */ | "powScalar"
  /** out = alpha ^ a is NOT supported; use exp(a*ln(alpha)) */
  /** backward helpers */ | "geluGrad" | "tanhGrad" | "sigmoidGrad" | "reluGrad"
  /** out = clamp(a, alpha, beta) - beta passed via second scalar */ | "clamp";

export type BinaryOp = "add" | "sub" | "mul" | "div" | "pow" | "max" | "min" | "eq" | "lt" | "gt" | "le" | "ge";
export type ReduceOp = "sum" | "max" | "min";

export interface Backend {
  readonly device: Device;

  // ---- memory ----
  alloc(byteLength: number): Buf;
  free(buf: Buf): void;
  /** Copy host data into the buffer (byte offset 0). */
  write(buf: Buf, data: ArrayBufferView): void;
  /** Read `byteLength` bytes starting at byte offset 0. Forces pending work to complete. */
  read(buf: Buf, byteLength: number): Promise<ArrayBuffer>;
  /** Device-to-device copy of `byteLength` bytes. */
  copy(src: Buf, dst: Buf, byteLength: number, srcByteOffset?: number, dstByteOffset?: number): void;

  // ---- elementwise ----
  /** out[i] = op(a[i]) for i < n. `alpha`/`beta` are op-specific scalars (scale, addScalar, powScalar, clamp). */
  unary(op: UnaryOp, a: Buf, n: number, out: Buf, alpha?: number, beta?: number): void;
  /**
   * Broadcasting binary op. `outShape` is the broadcast shape; `aStrides`/`bStrides` have the same rank as
   * outShape with 0 for broadcast dimensions. Inputs may be read with those strides (so non-contiguous inputs
   * are allowed here). Comparison ops write 1.0/0.0 as f32.
   */
  binary(op: BinaryOp, a: Buf, b: Buf, out: Buf, outShape: number[], aStrides: number[], bStrides: number[]): void;
  /** Materialise a strided view into a contiguous buffer of prod(aShape) elements. */
  copyStrided(a: Buf, aShape: number[], aStrides: number[], aOffset: number, out: Buf): void;
  /** out[i] = cond[i] != 0 ? a[i] : b[i] (all contiguous, same n). cond is f32 or u32 0/1 - both must work. */
  where(cond: Buf, a: Buf, b: Buf, n: number, out: Buf): void;
  /** out[i] = mask[i] != 0 ? value : a[i]. */
  maskedFill(a: Buf, mask: Buf, value: number, n: number, out: Buf): void;
  fill(buf: Buf, n: number, value: number): void;

  // ---- linear algebra ----
  /**
   * Batched matmul: for each batch b, out[b] (MxN) = opA(a[b]) (MxK) · opB(b[b]) (KxN).
   * a[b] starts at element offset b*aBatchStride (0 means broadcast the single matrix), same for b.
   * transA means a is stored as KxM; transB means b is stored as NxK. out is always contiguous batchxMxN.
   */
  matmul(a: Buf, b: Buf, out: Buf, batch: number, M: number, K: number, N: number,
         transA: boolean, transB: boolean, aBatchStride: number, bBatchStride: number): void;

  // ---- reductions ----
  /** Reduce `a` of `shape` over `axes` (sorted ascending). out has the reduced shape with those axes removed. */
  reduce(op: ReduceOp, a: Buf, shape: number[], axes: number[], out: Buf): void;
  /** Row-wise argmax for rowsxcols -> out (i32, rows). */
  argmax(a: Buf, rows: number, cols: number, out: Buf): void;

  // ---- fused row kernels ----
  softmax(a: Buf, rows: number, cols: number, out: Buf, log: boolean): void;
  /** Given forward output `out` (softmax or log-softmax) and gradOut, compute gradIn. */
  softmaxBackward(out: Buf, gradOut: Buf, rows: number, cols: number, gradIn: Buf, log: boolean): void;
  /** meanRstd stores (mean, rstd) per row as 2*rows f32 for backward. */
  layerNorm(x: Buf, gamma: Buf, beta: Buf, rows: number, cols: number, eps: number, out: Buf, meanRstd: Buf): void;
  layerNormBackward(x: Buf, gamma: Buf, meanRstd: Buf, gradOut: Buf, rows: number, cols: number,
                    gradX: Buf, gradGamma: Buf, gradBeta: Buf): void;
  /**
   * Fused cross-entropy over rowsxcols logits with i32 targets (rows). lossOut receives per-row loss (rows f32).
   * If gradOut is given it receives d(mean loss)/d(logits) scaled by 1/rows (rowsxcols f32). Targets equal to
   * ignoreIndex contribute 0 loss and 0 grad (and are excluded from the mean denominator - pass validCount).
   */
  crossEntropy(logits: Buf, targets: Buf, rows: number, cols: number, lossOut: Buf, gradOut?: Buf,
               ignoreIndex?: number, validCount?: number): void;

  // ---- indexing ----
  /** out[i, :] = table[ids[i], :] for i < n, rows of width `dim`. ids are i32. */
  gather(table: Buf, ids: Buf, n: number, dim: number, out: Buf): void;
  /** gradTable[ids[i], :] += gradOut[i, :] for i < n. Must be correct under duplicate ids. */
  scatterAdd(gradOut: Buf, ids: Buf, n: number, dim: number, gradTable: Buf): void;

  // ---- optimiser ----
  /** In-place AdamW step (PyTorch semantics, bias-corrected) over n elements; step is 1-based. */
  adamw(param: Buf, grad: Buf, m: Buf, v: Buf, n: number, lr: number, beta1: number, beta2: number,
        eps: number, weightDecay: number, step: number): void;
  /** Returns sum of squares of n elements into out (1 f32) - used for global-norm gradient clipping. */
  sumSquares(a: Buf, n: number, out: Buf): void;

  // ---- randomness ----
  randn(buf: Buf, n: number, seed: number): void;
  rand(buf: Buf, n: number, seed: number): void;
  /** mask[i] = (u < p) ? 0 : 1 as f32 (keep-mask). */
  dropoutMask(mask: Buf, n: number, p: number, seed: number): void;

  /** Flush pending work and wait for completion. */
  sync(): Promise<void>;
}
