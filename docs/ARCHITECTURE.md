# Kiln — architecture and module contracts

Kiln is a deep-learning framework written from scratch in TypeScript: a tensor library with a CPU backend and a WebGPU backend (hand-written WGSL kernels), reverse-mode autograd, neural-network modules, optimisers, a byte-level BPE tokenizer, a GPT (decoder-only transformer) implementation and a training loop — plus a web "studio" that trains a GPT live in a browser tab with visualisations. Zero runtime dependencies. This document is the contract every module is built against. Where code and this document disagree, fix the code or update this document in the same change.

## Layering (bottom → top)

```
src/tensor     dtype, Shape utilities (broadcast, strides, index math), Tensor class (device-agnostic handle, requiresGrad, grad), seeded RNG
src/backend    Backend interface + CPU backend (Float32Array/Int32Array kernels) + WebGPU backend (WGSL kernels, buffer pool, dispatch batching)
src/autograd   Function nodes, tape, backward(); differentiable op definitions (forward + backward for every op)
src/nn         Module base (parameters, stateDict, train/eval), Linear, Embedding, LayerNorm, Dropout, GELU, CausalSelfAttention, TransformerBlock, Sequential
src/optim      SGD (momentum, nesterov), AdamW (fused GPU kernel), gradient clipping, LR schedulers (cosine with warmup, step, constant)
src/data       tokenizer/char.ts, tokenizer/bpe.ts (byte-level BPE train + encode/decode), dataset.ts (token stream → random (x, y) batches)
src/models     gpt.ts (config, forward, loss, generate with temperature/top-k/top-p), checkpoint.ts (safetensors-compatible read/write + JSON sidecar)
src/train      trainer.ts (train loop with callbacks: onStep, onEval, onSample; tokens/s; eval loss; checkpoint cadence)
web/           Vite app: Studio (train live), Playground (generate from a shipped checkpoint), Inspect (attention + embeddings), Bench (kernel GFLOPS), Tests (in-browser GPU↔CPU parity suite)
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

### WebGPU backend details
- `device.ts`: request adapter/device with raised `maxStorageBufferBindingSize`/`maxBufferSize`, the `timestamp-query` feature when available (for the Bench page); a graceful `isWebGpuAvailable()`.
- `bufferpool.ts`: size-class pooling of `GPUBuffer`s (STORAGE | COPY_SRC | COPY_DST); freed buffers return to the pool; a ring of mappable staging buffers for readback.
- `kernels/*.wgsl.ts`: WGSL as template strings. Required kernels: `elementwise_unary`, `elementwise_binary_broadcast` (strides via uniform), `copy_strided`, `matmul_tiled` (shared-memory tiles with register blocking, handles transA/transB and batch, non-multiple-of-tile sizes), `reduce_axis`, `softmax_rows` (one workgroup per row, online max/sum), `layernorm_rows` + backward, `gather`, `scatter_add` (f32 atomics are unavailable in WGSL: implement with `atomicAdd` on `u32` fixed-point *or* a deterministic per-row segmented approach; pick one, document it, test it), `cross_entropy_rows`, `adamw`, `fill`, `rng_hash`, `argmax_rows`.
- Dispatch batching: a `CommandBatch` collects `dispatchWorkgroups` calls into one `GPUCommandEncoder`; `submit()` on readback, on `sync()`, or after 64 dispatches. Uniform parameters go through a ring of small uniform buffers (`writeBuffer` before encoding).
- Pipeline cache keyed by (kernel name, constants). Workgroup sizes chosen from device limits.

## Autograd
- `Node { inputs: Tensor[]; backward(gradOutput: Tensor): (Tensor | null)[]; saved }`. `Tensor.backward()` topologically sorts the graph (reverse DFS), accumulates grads (`grad = grad ? grad + g : g`), frees saved tensors after use. `noGrad(fn)` scope disables recording.
- Every op in the Tensor list has a tested backward. `tests/gradcheck.test.ts` does central-difference numerical gradient checks (CPU, eps 1e-3, rtol 1e-2) for every op on small random tensors, including broadcasting cases and non-contiguous inputs.

## nn / optim / models
- `Module`: `parameters(): Tensor[]`, `namedParameters()`, `stateDict()`, `loadStateDict()`, `train()/eval()`, `to(device)`, `zeroGrad()`.
- `GPT` config: `{ vocabSize, blockSize, nLayer, nHead, nEmbd, dropout, bias }` (nanoGPT-compatible naming). Weight tying between token embedding and LM head. Causal attention via a `tril` mask + `maskedFill(-inf)` + softmax (a fused attention kernel is a stretch goal, not required). `generate(idx, maxNewTokens, {temperature, topK, topP, seed})`.
- Checkpoints: `safetensors` format (8-byte little-endian header length + JSON header + raw tensors) + a sidecar JSON (`config`, `tokenizer`, `trainingLog`). Must round-trip exactly on CPU and be loadable from a URL in the browser.
- Tokenizers: `CharTokenizer` and `ByteLevelBPE` (train merges from text with a target vocab size, GPT-2-style byte→unicode mapping, regex pre-tokenisation, exact encode/decode round trip). Serialise to JSON.
- Trainer: AdamW, cosine LR with warmup, gradient clipping (global norm), gradient accumulation, eval every N steps on a held-out split, sample-generation callback, `tokens/s` and `ms/step` metrics, resumable from checkpoint.

## Shipped artefacts
- `data/tinyshakespeare.txt` (public domain) is bundled. `scripts/train-tiny.ts` trains a **char-level GPT** (`nLayer 4, nHead 4, nEmbd 128, blockSize 128`) on CPU for a time budget and writes `web/public/checkpoints/shakespeare-tiny.safetensors` + `.json` (a few MB). The Playground page loads it so visitors can generate text immediately even without WebGPU.

## Testing strategy (required)
1. `tests/shape.test.ts`: broadcasting, strides, index math.
2. `tests/ops.test.ts`: every CPU op against hand-computed expected values (golden constants embedded).
3. `tests/gradcheck.test.ts`: numerical gradient checks for every differentiable op.
4. `tests/nn.test.ts`: Linear/LayerNorm/Attention forward against reference computations; GPT forward shape + loss ≈ ln(vocab) at init; one training run on a toy problem (learn to copy / modular addition / sort) must reach a loss threshold in ≤ N steps — this is the end-to-end correctness gate.
5. `tests/tokenizer.test.ts`: BPE trains, encodes, decodes round trip on unicode text; char tokenizer.
6. `tests/checkpoint.test.ts`: safetensors round trip; header compatible with the reference format.
7. **GPU parity** (browser only): `web/src/pages/tests.ts` runs every op with random inputs on both backends and reports max abs/rel error; thresholds 1e-4 (1e-3 for matmul with large K). Also runnable headless via `scripts/gpu-parity.mjs` with Playwright + Chromium flags `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=swiftshader` (best effort; CI job `continue-on-error: true`).
8. `scripts/bench-cpu.ts`: matmul GFLOPS, tokens/s for the tiny GPT on CPU — numbers go in README.

## Web app (`web/`)
Design identity: *"the firing room"* — near-black `#0e0d0b` background with a warm ember accent `#ff7a1a` and soft ash greys; display face Space Grotesk, UI Inter, numerals/code JetBrains Mono; glowing loss curve; a GPU/CPU badge pulses while training. No UI framework (vanilla TS + a tiny signal store), canvas-drawn charts (no chart libs).
Pages: (1) **Studio** — pick dataset (TinyShakespeare, or paste/upload text), tokenizer (char/BPE with vocab size), model preset (nano/micro/mini), batch/context/lr; Start/Pause/Resume; live loss chart (train + eval), tokens/s, step time, ETA, memory; periodic generated samples stream into a side panel; download checkpoint; (2) **Playground** — load the shipped checkpoint or your own; prompt, temperature/top-k/top-p sliders, token-by-token streaming with per-token probability shading; (3) **Inspect** — attention heatmaps per layer/head for the current prompt, token-embedding 2-D projection (PCA computed in-browser), weight histograms; (4) **Bench** — matmul GFLOPS CPU vs GPU across sizes, kernel timings (timestamp queries when available), bar charts; (5) **Tests** — the parity suite with a pass/fail table. Training runs in a Web Worker (WebGPU in workers is supported in Chromium) so the UI never stalls; falls back to the main thread if `navigator.gpu` is missing in workers.
