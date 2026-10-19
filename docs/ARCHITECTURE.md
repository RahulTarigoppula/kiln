# Kiln - architecture and module contracts

Kiln is a deep-learning framework written from scratch in TypeScript: a tensor library with a CPU backend and a WebGPU backend (hand-written WGSL kernels), reverse-mode autograd, neural-network modules, optimisers, a byte-level BPE tokenizer, a GPT (decoder-only transformer) implementation and a training loop - plus a web "studio" that trains a GPT live in a browser tab with visualisations. Zero runtime dependencies. This document is the contract every module is built against. Where code and this document disagree, fix the code or update this document in the same change.

## Layering (bottom to top)

```
src/tensor     dtype, Shape utilities (broadcast, strides, index math), Tensor class (device-agnostic handle, requiresGrad, grad), seeded RNG
src/backend    Backend interface + CPU backend (Float32Array/Int32Array kernels) + WebGPU backend (WGSL kernels, buffer pool, dispatch batching)
src/autograd   Function nodes, tape, backward(); differentiable op definitions (forward + backward for every op)
src/nn         Module base (parameters, stateDict, train/eval), Linear, Embedding, LayerNorm, Dropout, GELU, CausalSelfAttention, TransformerBlock, Sequential
src/optim      SGD (momentum, nesterov), AdamW (fused GPU kernel), gradient clipping, LR schedulers (cosine with warmup, step, constant)
src/data       tokenizer/char.ts, tokenizer/bpe.ts (byte-level BPE train + encode/decode), dataset.ts (token stream to random (x, y) batches)
src/models     gpt.ts (config, forward, loss, generate with temperature/top-k/top-p), checkpoint.ts (safetensors-compatible read/write + JSON sidecar)
src/train      trainer.ts (train loop with callbacks: onStep, onEval, onSample; tokens/s; eval loss; checkpoint cadence)
web/           Vite app: Studio (train live), Playground (generate from a shipped checkpoint), Inspect (attention + embeddings), Bench (kernel GFLOPS), Tests (in-browser GPU vs CPU parity suite)
```

Rules:
- `tensor` and `backend` know nothing about autograd. `autograd` knows about tensors and backends. `nn`/`optim`/`models` only use `autograd` ops and `Tensor`.
- **Eager execution**. Every op dispatches immediately to the backend. On CPU, results are available synchronously. On WebGPU, ops *enqueue* GPU work synchronously and return a Tensor whose storage is a `GPUBuffer`; only `await tensor.data()` / `await tensor.item()` forces a readback. The backend batches dispatches into command encoders and submits on readback or when the pending dispatch count exceeds a threshold (default 64). This gives PyTorch-like ergonomics without a graph compiler.
- dtype: `f32` for all learnable data; `i32` for token ids / indices; `u8`/`u32` for masks where convenient. No f16 (keeps parity testing exact-ish).
- Deterministic by default: a seeded PRNG (`src/tensor/random.ts`, xoshiro128**) feeds CPU init/dropout; GPU dropout and randn use a hashed counter-based RNG (Philox-lite or PCG hash) seeded from the same stream. Tests set seeds.

## Tensor
```ts
class Tensor {
  readonly shape: readonly number[]; readonly strides: readonly number[]; readonly dtype: DType; readonly device: Device /* "cpu" | "webgpu" */;
  storage: Storage /* backend buffer handle + element offset */; requiresGrad: boolean; grad: Tensor | null; gradFn: Node | null;
  static zeros/ones/full/randn/rand/arange/fromArray(data, shape, {device, dtype, requiresGrad}); static eye
  to(device): Tensor; cpu(): Promise<Tensor>; data(): Promise<Float32Array|Int32Array>; dataSync() /* cpu only */; item(): Promise<number>
  // ops (all also exported as functions in src/autograd/ops.ts):
  add sub mul div pow neg exp log sqrt tanh sigmoid relu gelu softmax logSoftmax sum mean max min argmax var std matmul transpose permute reshape view flatten squeeze unsqueeze slice narrow cat stack expand broadcastTo gather indexSelect embedding scatterAdd where maskedFill tril triu clamp contiguous
  backward(grad?: Tensor): void; detach(): Tensor; zeroGrad(): void; toString(): string
}
```
- Views: `reshape/view/transpose/permute/slice/expand` are **zero-copy** on CPU where strides allow; on WebGPU `reshape` of a contiguous tensor is zero-copy, and everything else that produces a non-contiguous layout materialises via a strided-copy kernel (`contiguous()`), because WGSL kernels assume contiguous inputs. `Tensor.isContiguous` documents this.
- Broadcasting follows NumPy rules; the backend receives explicit output shape + input strides (0 for broadcast dims).

### Tensor implementation notes (as built)
- `Tensor.isContiguous` means dense row-major strides (offset may be non-zero); `Tensor.isDense` additionally means `offset === 0`, i.e. a kernel can read the buffer directly. Ops call an internal `dense()` that materialises anything else via `copyStrided`.
- Layering: `tensor.ts` delegates its op *methods* (`t.add(...)`, `t.matmul(...)`, `t.backward()`) to `autograd/ops.ts` / `autograd/graph.ts`. The import cycle is resolved at call time; `dtype/shape/random/storage/device` and both backends stay autograd-free.
- `Storage` is refcounted: views `retain()`, `Tensor.free()` releases, the backend buffer is freed on the last release. Allocations are at least 4 bytes.
- `to(device)` is synchronous and only moves CPU to other device; use `await t.cpu()` / `await t.data()` for device to host. `to()` does not record autograd (the result is a leaf that keeps `requiresGrad`).
- `dataSync()` (CPU only) returns a *live* typed-array view for contiguous tensors and a copy for non-contiguous ones. `itemSync()` and `toArray()` are CPU-only too.
- In-place helpers without autograd for optimisers: `copy_`, `fill_`, `zero_`.
- Arithmetic kernels are f32 only. `i32` tensors (token ids, argmax output) support creation, `cat`/`stack`, views, `to()`, `embedding`/`indexSelect` indices and a CPU-only `toDtype`/`float()`/`int()` cast.
- Naming: `var`/`std` are methods (`t.var(axes, {unbiased, keepDim})`); the functional form is `variance`. `gather(x, dim, ids)` is an alias of `indexSelect`, not PyTorch's element-wise gather. `scatterAdd(target, ids, src)` is out-of-place along dim 0 (`result[ids[i], ...] += src[i, ...]`). `where(cond, a, b)` accepts numbers for `a`/`b`.
- Gradient conventions: `max`/`min` reductions split the gradient equally among ties; elementwise `maximum`/`minimum` route ties to the first argument; `abs` has gradient 0 at 0; `clamp` passes gradient where the output equals the input.
- `crossEntropy(logits, targets, {ignoreIndex = -100, reduction = "mean" | "sum" | "none", validCount?})`: the non-ignored count is computed exactly when `targets` live on the CPU; on the GPU it defaults to the number of rows unless `validCount` is passed (no readback is performed in the forward pass).
- `dropout(x, p, training, seed?)` draws its seed from the global xoshiro generator (`manualSeed` makes it reproducible); `randn`/`rand` likewise accept an explicit `seed`.
- `backward()`: iterative reverse topological order; gradients stored on leaves are always dense and never aliased with another tensor's grad; non-leaves keep their grad only after `retainGrad()`; a second `backward()` through a released node throws (no `retain_graph`).
- `Tensor.free()` is idempotent per handle (`isFreed`); `zeroGrad()` frees the grad buffer (a tensor *owns* its `.grad`); gradient accumulation across `backward()` calls frees the replaced grad.

### Memory management: scopes (`src/tensor/scope.ts`)
There is no garbage collector for device buffers - `Storage` is refcounted and a buffer returns to the backend only when every handle has called `free()`. Ops create many short-lived tensors (dense copies, broadcast masks, saved activations, gradient temporaries), so Kiln uses TF.js-style **scopes**:

```ts
const loss = tidy(() => { const { loss } = model.forward(x, y); loss.backward(); return loss; });
```
- Every `Tensor` constructed while the callback runs (the constructor calls `trackTensor`) is tracked by the innermost scope. At the end, everything tracked is freed **except**: tensors in the return value (a Tensor or any array/object/Map/Set nesting of Tensors), tensors passed to `keep(t)`, tensors reachable from those through *unreleased* autograd nodes (`gradFn.inputs` and `gradFn.saved`), and tensors removed from tracking with `escape(t)`. Kept tensors move to the parent scope, so scopes nest.
- `escape(t)` is applied automatically to things that must outlive any scope: module parameters and buffers (`Module.param/buffer`, `Module.to`), optimizer state (`Optimizer.stateTensor`), gradients stored on leaves by `backward()`, the `tril`/`triu` mask cache, GPT position-id cache and captured attention maps. Freeing them is the owner's job (`zeroGrad()`, replacing a grad, ...).
- `backward()` frees as it goes: each node's `saved` tensors (via `Node.release()`), temporaries created inside backward closures (every closure call runs in its own scope whose result is the returned gradient list), and gradients of non-leaf tensors once consumed (reference-counted because closures may return the incoming gradient for several inputs, e.g. `add`).
- `record(out, name, inputs, fn, saved)`: `saved` lists only *fresh* tensors the op created for backward (dense copies of non-dense inputs, `meanRstd`, the fused cross-entropy gradient, the dropout mask) - never its inputs or output. When no grad is needed they are freed immediately. The graph walk in `endScope` relies on this: a closure must not capture a fresh tensor that is not in `saved`.
- Scopes are synchronous. `tidy` throws if the callback returns a Promise; async code (readbacks, `generate`) wraps each synchronous stretch in its own `tidy`.
- `tests/memory.test.ts` wraps the CPU backend's `alloc`/`free` in a counting proxy and asserts the live-buffer count is flat across 20 training steps (forward+backward in `tidy`, `clipGradNorm`, AdamW, `zeroGrad`) and across `generate`.

## Backend interface (`src/backend/types.ts`)
```ts
interface Backend {
  readonly device: Device;
  alloc(byteLength: number): Buf; free(buf: Buf): void; write(buf, data: ArrayBufferView): void; read(buf, byteLength): Promise<ArrayBuffer>;
  // kernels (all take contiguous inputs unless noted; shapes passed as number[]):
  unary(op: UnaryOp, a: Buf, n: number, out: Buf, alpha?: number): void
  binary(op: BinaryOp, a: Buf, b: Buf, out: Buf, outShape: number[], aStrides: number[], bStrides: number[]): void   // broadcasting via strides
  copyStrided(a: Buf, aShape: number[], aStrides: number[], aOffset: number, out: Buf): void                       // materialise views, transposes, slices
  matmul(a: Buf, b: Buf, out: Buf, batch: number, M: number, K: number, N: number, transA: boolean, transB: boolean, aBatchStride: number, bBatchStride: number): void
  reduce(op: ReduceOp /* sum|max|min */, a: Buf, shape: number[], axes: number[], out: Buf): void
  softmax(a: Buf, rows: number, cols: number, out: Buf, log: boolean): void
  softmaxBackward(out: Buf, gradOut: Buf, rows: number, cols: number, gradIn: Buf, log: boolean): void
  layerNorm(x: Buf, gamma: Buf, beta: Buf, rows: number, cols: number, eps: number, out: Buf, meanRstd: Buf): void
  layerNormBackward(x, gamma, meanRstd, gradOut, rows, cols, gradX, gradGamma, gradBeta): void
  gather(table: Buf, ids: Buf, n: number, dim: number, out: Buf): void              // embedding forward
  scatterAdd(gradOut: Buf, ids: Buf, n: number, dim: number, gradTable: Buf): void  // embedding backward
  where(cond: Buf, a: Buf, b: Buf, n: number, out: Buf): void
  maskedFill(a: Buf, mask: Buf, value: number, n: number, out: Buf): void
  crossEntropy(logits: Buf, targets: Buf, rows: number, cols: number, lossOut: Buf, gradOut?: Buf): void // fused log-softmax + NLL (+ grad)
  adamw(param, grad, m, v, n, lr, beta1, beta2, eps, weightDecay, step): void
  fill(buf, n, value): void; randn(buf, n, seed): void; rand(buf, n, seed): void; dropoutMask(mask: Buf, n: number, p: number, seed: number): void
  argmax(a, rows, cols, out): void
  sync(): Promise<void>  // flush pending work
}
```
Every op in `autograd/ops.ts` must be implemented in terms of this interface so CPU and GPU share identical autograd code.

### Kernel semantics both backends must share
These are the exact behaviours the CPU backend implements and the autograd layer relies on; the WebGPU kernels must match them.
- **Buffers**: a CPU `Buf.handle` is an `ArrayBuffer`. `copyStrided`, `copy`, `cat` and slice-backward move raw 32-bit words, so they work for every dtype.
- **Unary backward helpers take the forward *input* x**: `geluGrad(x) = d gelu/dx`, `tanhGrad(x) = 1 − tanh²x`, `sigmoidGrad(x) = σ(x)(1 − σ(x))`, `reluGrad(x) = x > 0 ? 1 : 0`. `gelu` is the tanh approximation `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. `clamp` uses `alpha` = min and `beta` = max (±Infinity allowed). `powScalar(x, alpha) = x^alpha`.
- **binary**: comparison ops write 1.0/0.0 as f32. `max`/`min` return the first argument on ties (`x >= y ? x : y`).
- **reduce**: `axes` sorted ascending and never empty (the autograd layer short-circuits the no-op case). Identities: sum 0, max −∞, min +∞.
- **argmax**: index of the *first* maximal element per row; output is i32.
- **where / maskedFill**: `cond`/`mask` are read as u32 bit patterns, "true" = any non-zero word (so f32 1.0 and u32 1 both work).
- **gather**: ids outside `[0, rows)` produce a zero row. **scatterAdd**: ids outside the table are skipped; duplicate ids accumulate.
- **softmaxBackward**: softmax `gradIn = out·(g − Σ g·out)`; log-softmax `gradIn = g − exp(out)·Σ g` (row-wise sums).
- **layerNorm**: biased variance, `rstd = 1/sqrt(var + eps)`, `meanRstd[2r] = mean`, `meanRstd[2r+1] = rstd`. **layerNormBackward** *overwrites* `gradGamma`/`gradBeta` (it does not accumulate).
- **crossEntropy**: `lossOut[r] = logsumexp(row) − logit[r, target]`; rows whose target equals `ignoreIndex` *or* is out of range contribute 0 loss and 0 grad. `gradOut = (softmax − onehot) / denom` with `denom = max(1, validCount ?? rows)`. The autograd layer passes `validCount = 1` for `sum`/`none` reductions and scales itself.
- **adamw** (PyTorch): `p *= 1 − lr·wd; m = β1·m + (1−β1)·g; v = β2·v + (1−β2)·g²; p −= lr · (m / (1−β1^t)) / (sqrt(v / (1−β2^t)) + eps)`.
- **matmul**: `out` is always dense `batchxMxN`; `transA` means A is stored KxM, `transB` means B is stored NxK; a batch stride of 0 broadcasts a single matrix.
- **Counter-based RNG** (`rand`, `randn`, `dropoutMask`) - reference implementation in `src/tensor/random.ts`:
  ```
  mix(x)           : x ^= x >>> 16; x *= 0x21f0aaad; x ^= x >>> 15; x *= 0x735a2d97; x ^= x >>> 15      (u32, wrapping)
  splitmix32(x)    : mix(x + 0x9E3779B9)
  hash(seed, i)    : splitmix32(seed ^ (i * 0x9E3779B9))
  uniform(seed, i) : (hash(seed, i) >>> 8) / 16777216                      to [0, 1), 24-bit
  rand[i]          = uniform(seed, i)
  dropoutMask[i]   = uniform(seed, i) < p ? 0 : 1
  randn[i]         : j = i & ~1; u1 = uniform(seed, j); u2 = uniform(seed, j | 1)
                     r = sqrt(−2·ln(1 − u1)); t = 2π·u2;  randn[i] = (i even) ? r·cos t : r·sin t
  ```
  WGSL equivalent:
  ```wgsl
  fn splitmix32(x0: u32) -> u32 { var x = x0 + 0x9E3779B9u; x ^= x >> 16u; x *= 0x21f0aaadu; x ^= x >> 15u; x *= 0x735a2d97u; x ^= x >> 15u; return x; }
  fn hash_u32(seed: u32, i: u32) -> u32 { return splitmix32(seed ^ (i * 0x9E3779B9u)); }
  fn uniform01(seed: u32, i: u32) -> f32 { return f32(hash_u32(seed, i) >> 8u) / 16777216.0; }
  ```
  `seed` is passed as a u32 (`seed >>> 0` on the host). GPU normals differ from the CPU only by f32 transcendental rounding.

### WebGPU backend details
- `device.ts`: request adapter/device with raised `maxStorageBufferBindingSize`/`maxBufferSize`, the `timestamp-query` feature when available (for the Bench page); a graceful `isWebGpuAvailable()`.
- `bufferpool.ts`: size-class pooling of `GPUBuffer`s (STORAGE | COPY_SRC | COPY_DST); freed buffers return to the pool; a ring of mappable staging buffers for readback. `stats()` reports `liveBytes`, `liveBuffers`, `pooledBytes` and `created`; the Studio shows `liveBytes` per step and the count must plateau across steps.
- `kernels/*.wgsl.ts`: WGSL as template strings. Required kernels: `elementwise_unary`, `elementwise_binary_broadcast` (strides via uniform), `copy_strided`, `matmul_tiled` (shared-memory tiles with register blocking, handles transA/transB and batch, non-multiple-of-tile sizes), `reduce_axis`, `softmax_rows` (one workgroup per row, online max/sum), `layernorm_rows` + backward, `gather`, `scatter_add` (f32 atomics are unavailable in WGSL: implement with `atomicAdd` on `u32` fixed-point *or* a deterministic per-row segmented approach; pick one, document it, test it), `cross_entropy_rows`, `adamw`, `fill`, `rng_hash`, `argmax_rows`.
- Dispatch batching: a `CommandBatch` collects `dispatchWorkgroups` calls into one `GPUCommandEncoder`; `submit()` on readback, on `sync()`, or after 64 dispatches. Uniform parameters go through a ring of small uniform buffers (`writeBuffer` before encoding).
- Pipeline cache keyed by (kernel name, constants). Workgroup sizes chosen from device limits.

Implementation notes (as built in `src/backend/webgpu/`):
- Kernel modules are `kernels/{unary,binary,elementwise,matmul,reduce,softmax,layernorm,indexing,loss_optim,rng}.ts` (plain `.ts`, not `.wgsl.ts`); `kernels/index.ts` exports `KERNEL_TEMPLATES` for the syntax test. Each kernel is a template with `{{WG}}`/`{{EXPR}}`/`{{A<i>}}` placeholders; the pipeline cache key is (kernel name, option vars, storage access modes). When one `GPUBuffer` is bound as both input and output of a dispatch, all aliased bindings are promoted to `read_write` so WebGPU usage validation passes.
- `scatter_add` uses the **deterministic counting-sort approach**: `scatter_hist` (atomicAdd<u32> histogram of ids) to `scan_exclusive` (single-workgroup scan to segment offsets) to `scatter_place` (atomic cursor; arbitrary order inside a segment) to `scatter_rank` (re-orders each segment ascending by source index) to `scatter_segsum` (one thread per (table row, column) sums its segment in index order and accumulates into `gradTable`). Exact f32 sums in a fixed order, so results are bit-reproducible run to run; cost is O(Σ segment²) in `scatter_rank`, fine for realistic token distributions.
- `layerNormBackward` is three dispatches: per-row `gradX`, per-(column, row-chunk) partial sums of `g·x̂` and `g` into a scratch buffer, then a per-column sum of the chunk partials for `gradGamma`/`gradBeta`.
- `reduce` merges adjacent kept/reduced dims on the host; a trailing reduced block uses the shared-memory row kernel (`reduce_rows`), anything else the general strided kernel. A single leading reduced group of 512 or more rows (bias gradients) is split into `pickChunks(rows)` chunks (the largest divisor in [2, 128] leaving at least 8 rows per chunk) that become an extra kept dimension, and the `[chunks, outN]` partials are reduced by a second general dispatch; prime row counts fall back to the single pass. `sumSquares` is two-stage (per-workgroup partials, then `reduce_rows`).
- The counter-based RNG is `splitmix32(seed ^ (i * 0x9E3779B9))`, uniform = `(h >>> 8) / 2^24`, normal via Box - Muller on `(i & ~1, i | 1)` with `r = sqrt(-2 ln(1 - u1))` - identical to `src/tensor/random.ts`, so `rand`/`dropoutMask` match the CPU bit-for-bit.
- Unary backward helpers (`tanhGrad`, `sigmoidGrad`, `geluGrad`, `reluGrad`) all take the forward *input* x, as the CPU backend does.
- `register.ts` exposes `initWebGpu()` which creates the backend once and calls `registerBackend("webgpu", ...)`; `parity.ts` holds the backend-level parity suite (`runParity`, `runParityAndPublish` to `window.__kilnParityDone`) consumed by the Tests page and `scripts/gpu-parity.mjs` (`npm run test:gpu`). Known limits: strided kernels support rank ≤ 6 after dim merging; matmul batch ≤ 65535; `fill` writes f32 bit patterns (zero is dtype-agnostic).

## Autograd
- `Node { inputs: Tensor[]; backward(gradOutput: Tensor): (Tensor | null)[]; saved; released }`. `Tensor.backward()` topologically sorts the graph (reverse DFS), accumulates grads (`grad = grad ? grad + g : g`), frees saved tensors after use (`released = true`, which also tells `endScope` not to walk through the node). `noGrad(fn)` scope disables recording.
- Every op in the Tensor list has a tested backward. `tests/gradcheck.test.ts` does central-difference numerical gradient checks (CPU, eps 1e-3, rtol 1e-2, atol 1e-3) for every op on small random tensors, including broadcasting cases and non-contiguous inputs.
- Backward of views: `slice` writes the gradient into zeros via contiguous `copy` runs (any step), `cat` slices the gradient, `expand`/broadcasting sum it back, `permute` applies the inverse permutation. `cat` forward is also built from `copy` runs (one per outer index per input) so no strided-write kernel is needed.
- CPU GEMM: NN/TN use a 4-row register-blocked i-k-j kernel (K blocked by 256, N by 512; transposed A is addressed by strides, no copy), NT uses a 2x2 register-blocked dot kernel, TT runs NN on the swapped operands and transposes the result. Transposed *views* feeding `matmul` are fused into `transA`/`transB` (also for batched operands with contiguous batch strides); other layouts are materialised. `tests/matmul-bench.test.ts` is a sanity bound, `scripts/bench-cpu.ts` prints GFLOPS/GB/s tables.

## nn / optim / models
- `Module`: `parameters(): Tensor[]`, `namedParameters()`, `stateDict()`, `loadStateDict()`, `train()/eval()`, `to(device)`, `zeroGrad()`.
- `GPT` config: `{ vocabSize, blockSize, nLayer, nHead, nEmbd, dropout, bias }` (nanoGPT-compatible naming). Weight tying between token embedding and LM head. Causal attention via a `tril` mask + `maskedFill(-inf)` + softmax (a fused attention kernel is a stretch goal, not required). `generate(idx, maxNewTokens, {temperature, topK, topP, seed})`.
- Checkpoints: `safetensors` format (8-byte little-endian header length + JSON header + raw tensors) + a sidecar JSON (`config`, `tokenizer`, `trainingLog`). Must round-trip exactly on CPU and be loadable from a URL in the browser.
- Tokenizers: `CharTokenizer` and `ByteLevelBPE` (train merges from text with a target vocab size, GPT-2-style byte->unicode mapping, regex pre-tokenisation, exact encode/decode round trip). Serialise to JSON.
- Trainer: AdamW, cosine LR with warmup, gradient clipping (global norm), gradient accumulation, eval every N steps on a held-out split, sample-generation callback, `tokens/s` and `ms/step` metrics, resumable from checkpoint.

### nn implementation notes (as built in `src/nn/`)
- Registration is explicit inside constructors - `this.param(name, t)`, `this.buffer(name, t, {persistent})`, `this.child(name, m)` (public aliases `registerParameter/Buffer/Module`) - with auto-discovery of own properties holding a `Module`, a `Module[]` (`h.0`, `h.1`, ...) or a requires-grad `Tensor` as a fallback. `ModuleList` names its children `"0"`, `"1"`, ...; `Sequential` likewise. Names are dotted exactly like nanoGPT (`transformer.h.0.attn.c_attn.weight`, `lm_head.weight`).
- `namedParameters()`/`parameters()` list each tensor once (first name wins, so the tied `lm_head.weight` is reported as `transformer.wte.weight`); `stateDict()` lists it under both names (PyTorch semantics). `loadStateDict(strict)` copies in place, throws on shape mismatch / missing / unexpected keys, and accepts a missing alias whose tensor was already loaded under another name. Non-persistent buffers (attention mask, LayerNorm's zero beta when `bias: false`) are moved by `to()` but kept out of `stateDict()`.
- `to(device)` moves CPU to device, keeps tied tensors tied (memoised), rebinds the properties that held the old tensor (`onTensorMoved` hook for unregistered references) and returns `this`.
- `Linear` stores `weight` as `[out, in]` and computes `x.matmul(weight.transpose())` - the transposed view is fused into the GEMM's `transB`, no copy. Init `N(0, 0.02)` (`scaleInit: nLayer` to `0.02/√(2·nLayer)` for residual projections), zero biases; `Embedding` `N(0, 0.02)`; `LayerNorm` ones/zeros.
- `CausalSelfAttention`: fused `c_attn` `[C to 3C]`, slice to reshape `[B, T, nh, hd]` to transpose; `q·kᵀ/√hd`; a non-persistent `mask` buffer `[1, 1, blockSize, blockSize]` holding 1 **above** the diagonal is sliced to `T` and applied with `maskedFill(ATTENTION_MASK_VALUE = -Infinity)` (exact zeros after softmax; switch to −1e9 if a backend ever yields NaN); softmax, attention dropout, `att·v`, merge heads (one copy), `c_proj`, residual dropout. `captureAttention = true` keeps the last softmax output detached in `lastAttention` (`GPT.captureAttention` toggles all layers, `GPT.attentionWeights()` collects them).
- `Block`: pre-LN, `x + attn(ln_1(x))`, `x + mlp(ln_2(x))`; `MLP` = `c_fc (4x) to GELU to c_proj to Dropout`.

### optim implementation notes (`src/optim/`)
- `Optimizer<H>`: param groups `{params, hyper}` merged over `defaults`; `step()` runs under `noGrad`; `lr` getter/setter fans out to every group (schedulers set it each step); `stateTensors()` exposes per-parameter state for checkpoints (`optim.<param name>.<key>` records); `stateDict()/loadStateDict()` carry hyper-parameters and the step count.
- `AdamW` is one fused `backend.adamw` dispatch per parameter (PyTorch semantics, decoupled decay, bias correction); `configureAdamW(model, {lr, weightDecay, betas})` builds nanoGPT's two groups (decay for ≥2-D parameters, none for biases/LayerNorm). `SGD` (momentum, Nesterov, coupled weight decay) is written with tensor ops inside `tidy`.
- `clipGradNorm(params, maxNorm)` performs **no readback**: `backend.sumSquares` per grad into `[1]` tensors to `cat` to `sum` to `sqrt`, `scale = min(1, maxNorm / (norm + 1e-6))` with broadcasting ops, grads replaced by `grad · scale` (old grads freed). It returns the pre-clip norm as a 0-d tensor; `await norm.item()` only when the number is needed (the Trainer reads it together with the loss).
- Schedules are pure functions of the step: `cosineWithWarmup(step, {warmup, total, lrMax, lrMin})` (nanoGPT: `lrMax·(step+1)/(warmup+1)` during warm-up), `stepSchedule`, `constantSchedule`; `make*` variants curry them into `LRSchedule = (step) => lr`.

### models / train implementation notes
- `GPT.forward(idx, targets?, {allPositions, ignoreIndex = -1})`: with targets, logits for every position and the fused mean cross-entropy over `B·T` rows; without, logits for the **last position only** (`[B, 1, V]`) unless `allPositions`. Position ids are an escaped per-`(T, device)` cache. `numParameters(nonEmbedding = true)` subtracts `wpe` like nanoGPT; `flopsPerToken(T)` = `6N + 12·L·H·Q·T`.
- `generate()` is async: each token is one `tidy(() => forward(cond).logits)` enqueue plus a single small readback; sampling (temperature, top-k, top-p, seeded `Random`) happens on the CPU in `sampleLogits`; `temperature ≤ 0` is greedy. The context is cropped to `blockSize`; `onToken(id, probs, row)` streams tokens for the Playground and may return a promise that is awaited before the next token (back-pressure); an `AbortSignal` in `signal` ends generation after the current token and the tokens so far are returned.
- `io.ts`: `saveGPT(model, tokenizerJson, trainingLog, {optimizer, step})` is **async** (GPU weights must be read back) and returns `{bytes, meta}`. Tied parameters are written once; the alias map lives in `meta.extra.tied`; optimizer state goes in as `optim.<param>.<m|v>` with hyper-parameters in `meta.extra.optimizer`. `loadGPT(bytes, meta, device)` rebuilds the model with `strict` loading; `restoreOptimizer(optimizer, model, records, meta)` restores state. Node helpers (`saveGPTNode`, `loadGPTNode`) live in `io-node.ts`; browser bundles import `io.ts` only.
- `Trainer`: each micro-step is `tidy(() => { forward; (loss / gradAccumSteps).backward(); return loss.detach() })`, batches are uploaded as i32 `[B, T]` and freed after the step, then optional `clipGradNorm`, `optimizer.step()`, `zeroGrad()`. Losses (and grad norms) are read back in one `stack(...).data()` every `lossReadbackInterval` steps (default 1) - the only host sync in the loop; `tokens/s` and `ms/step` are wall-clock over that interval. After each flush the loop yields one macrotask (`MessageChannel`) so `pause()/stop()` and UI rendering get a turn even when every await resolves immediately (CPU backend). Eval runs `evalIters` random val/train batches under `noGrad` in `eval()` mode with one readback; samples use the tokenizer and `generate`; `checkpoint()`/`loadCheckpoint()` round-trip weights, optimizer state, step and log (resume continues the step count).

## Shipped artefacts
- Shipped checkpoints live in `web/public/checkpoints/` and are listed in `web/src/lib/model-store.ts` (`SHIPPED_CHECKPOINTS`, first entry is the default). `shakespeare-micro` (default, 2026-10-19) was trained in the Studio on WebGPU (AMD integrated GPU, Chromium): preset micro (nLayer 4, nHead 4, nEmbd 128, blockSize 128, batch 32, dropout 0.1), char tokenizer, AdamW lr 1e-3 cosine to 1e-4 with 60 warm-up steps, weight decay 0.1, clip 1.0, 984 (of 1200 planned, stopped by the time budget) steps in 17 minutes at about 8-10 k tokens/s: val loss 2.01, train loss 2.00, 3.3 MB safetensors (weights only) plus a sidecar with the tokenizer, config, compacted training log with every `valLoss` point, and `extra.{hyper, finalValLoss, finalTrainLoss, device}`. `shakespeare-tiny` remains as the second option.
- `data/tinyshakespeare.txt` (public domain) is bundled. `scripts/train-tiny.ts` trains a **char-level GPT** on CPU for a wall-clock budget (`--minutes`, default 25) and writes `web/public/checkpoints/shakespeare-tiny.safetensors` + `.json` (weights only, no optimizer state; the sidecar carries the char tokenizer, the per-step training log with periodic `valLoss`, and `extra.{finalValLoss, finalTrainLoss, sample, config}`). The Playground page loads it so visitors can generate text immediately even without WebGPU.
  - The script **calibrates** throughput first and picks the largest of three configs that fits enough steps into the budget: A `nLayer 4, nHead 4, nEmbd 128, T 128` (the original target), B `3 / 3 / 96 / 128`, C `2 / 2 / 64 / 64`, all batch 16, AdamW lr 1e-3 to 1e-4 cosine with 50 warm-up steps, weight decay 0.1, clip 1.0, no dropout. The cosine horizon is re-estimated from the measured ms/step every 25 steps so the lr still reaches `lrMin` at the deadline.
  - Shipped checkpoint (2026-10-18, single-threaded JS on a heavily loaded CPU): config **C** was chosen because A ran at ~10 s/step (≈145 steps in budget) and B at ~4.8 s/step; C ran 2272 steps at ~0.5-1 s/step to **val loss 2.02, train loss 1.95**, 104 256 non-embedding parameters, 0.44 MB safetensors + 0.3 MB sidecar. Re-run `npm run train:tiny` on an idle machine (or `-- --preset A --minutes 60`) for a stronger model; the Playground is config-agnostic.

## Testing strategy (required)
1. `tests/shape.test.ts`: broadcasting, strides, index math.
2. `tests/ops.test.ts`: every CPU op against hand-computed expected values (golden constants embedded).
3. `tests/gradcheck.test.ts`: numerical gradient checks for every differentiable op.
4. `tests/nn.test.ts`: Linear/LayerNorm/Attention forward against reference computations; GPT forward shape + loss ≈ ln(vocab) at init; one training run on a toy problem (learn to copy / modular addition / sort) must reach a loss threshold in ≤ N steps - this is the end-to-end correctness gate.
5. `tests/tokenizer.test.ts`: BPE trains, encodes, decodes round trip on unicode text; char tokenizer.
6. `tests/checkpoint.test.ts`: safetensors round trip; header compatible with the reference format.
7. **GPU parity** (browser only): `web/src/pages/tests.ts` runs every op with random inputs on both backends and reports max abs/rel error; thresholds 1e-4 (1e-3 for matmul with large K). Also runnable headless via `scripts/gpu-parity.mjs` with Playwright + Chromium flags `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=swiftshader` (best effort; CI job `continue-on-error: true`).
8. `scripts/bench-cpu.ts`: matmul GFLOPS, tokens/s for the tiny GPT on CPU - numbers go in README.

## Web app (`web/`)
Design identity: *"the firing room"* - near-black `#0e0d0b` background with a warm ember accent `#ff7a1a` and soft ash greys; display face Space Grotesk, UI Inter, numerals/code JetBrains Mono; glowing loss curve; a GPU/CPU badge pulses while training. No UI framework (vanilla TS + a tiny signal store), canvas-drawn charts (no chart libs).
Pages: (1) **Studio** - pick dataset (TinyShakespeare, or paste/upload text), tokenizer (char/BPE with vocab size), model preset (nano/micro/mini), batch/context/lr; Start/Pause/Resume; live loss chart (train + eval), tokens/s, step time, ETA, memory; periodic generated samples stream into a side panel; download checkpoint; (2) **Playground** - load the shipped checkpoint or your own; prompt, temperature/top-k/top-p sliders, token-by-token streaming with per-token probability shading; (3) **Inspect** - attention heatmaps per layer/head for the current prompt, token-embedding 2-D projection (PCA computed in-browser), weight histograms; (4) **Bench** - matmul GFLOPS CPU vs GPU across sizes, kernel timings (timestamp queries when available), bar charts; (5) **Tests** - the parity suite with a pass/fail table. Training runs in a Web Worker (WebGPU in workers is supported in Chromium) so the UI never stalls; falls back to the main thread if `navigator.gpu` is missing in workers.

### Engine adapter (`web/src/lib/engine.ts`, as built)
- The only module in `web/` that imports `src/models`, `src/optim` and `src/train`. `PRESETS` layers UI defaults (block size, batch, lr, max steps) over `GPT_PRESETS`.
- `instantiateModel` is `loadGPTFromRecords(records, meta, device)` + `eval()`; `generateTokens` is an async generator over `GPT.generate` whose `onToken` returns a promise resolved when the consumer has taken the token, so the page paints per token (one macrotask between tokens) and an abort stops the engine after the current token; `attentionMaps` toggles `captureAttention`, runs one forward under `noGrad` inside `tidy`, reads every layer's `[1, nHead, T, T]` weights and frees them.
- `createTrainer` builds `new GPT(config, device)`, `configureAdamW` (betas 0.9/0.95), a cosine schedule (`lrMin = lr / 10`) and a `Trainer` with `lossReadbackInterval` 1 on the CPU and 2 on WebGPU; `exportCheckpoint` is `saveGPT` without optimizer state plus a compacted training log (at most 1500 entries, every `valLoss` point kept) and `extra.{hyper, finalTrainLoss, finalValLoss, device}`. The worker reports `backend.stats().liveBytes` with every step.
- `scripts/lint-text.mjs` (`npm run lint:text`, in CI) rejects typographic dashes, curly quotes and arrows in tracked text files outside the corpora; prose and strings use plain ASCII punctuation.

