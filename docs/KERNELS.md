# WGSL kernels

Every GPU operation in Kiln is one of the 27 WGSL templates below (`src/backend/webgpu/kernels/*.ts`). Templates are
plain TypeScript strings with `{{WG}}`, `{{EXPR}}`, `{{A<i>}}` placeholders; `makeBuilder` substitutes them and the
`PipelineCache` compiles each (template, options, access modes) combination once. Binding 0 is always a small uniform
with the dispatch parameters, bindings 1..k are storage buffers. When one buffer is bound as both input and output the
aliased bindings are promoted to `read_write` so WebGPU usage validation passes.

Workgroup size is `min(256, device limits)`. Rank-generic kernels take up to 6 dimensions after the host merges
contiguous dims. All arithmetic is f32; integer ids are i32/u32.

| Kernel | File | What it does |
| --- | --- | --- |
| `unary` | `unary.ts` | `out[i] = op(a[i])` for 22 ops (neg, exp, log, sqrt, rsqrt, tanh, sigmoid, relu, gelu, square, abs, recip, sin, cos, scale, addScalar, powScalar, clamp and the backward helpers geluGrad, tanhGrad, sigmoidGrad, reluGrad); `alpha`/`beta` scalars travel in the uniform. |
| `binary_broadcast` | `binary.ts` | `out = a op b` with NumPy broadcasting via per-input strides (0 for broadcast dims); 12 ops (add, sub, mul, div, pow, max, min, eq, lt, gt, le, ge); comparisons write 1.0/0.0. |
| `copy_strided` | `elementwise.ts` | Materialises a strided view (transposes, slices, expands) into a contiguous buffer; moves raw 32-bit words so it works for every dtype. |
| `fill` | `elementwise.ts` | Writes one f32 bit pattern to every element (zero is dtype-agnostic). |
| `where` | `elementwise.ts` | `out[i] = cond[i] != 0 ? a[i] : b[i]`; the condition is read as a u32 bit pattern with the sign bit masked so `-0.0` is false. |
| `masked_fill` | `elementwise.ts` | `out[i] = mask[i] != 0 ? value : a[i]`; used for the causal attention mask with `-inf`. |
| `matmul_tiled` | `matmul.ts` | Batched f32 GEMM with 64x64 output tiles, BK = 16, 16x16 threads each owning a 4x4 register block; `transA`/`transB` by load-index math, zero-padded edges, batch via workgroup z with stride 0 broadcasting one operand. |
| `reduce_rows` | `reduce.ts` | Sum/max/min of a trailing contiguous block: one workgroup per row, lane-strided loads, shared-memory tree reduction (fixed order). |
| `reduce_general` | `reduce.ts` | Sum/max/min over an arbitrary set of axes: one thread per output element iterating the reduced index space. Leading-axis reductions over 512 or more rows run in two stages (chunk partials, then a reduce over the partials) for parallelism. |
| `argmax_rows` | `reduce.ts` | Index of the first maximal element per row, i32 output. |
| `sum_squares_partial` | `reduce.ts` | Stage 1 of a two-stage sum of squares (per-workgroup partials); stage 2 is `reduce_rows`. Used by gradient-norm clipping without any readback. |
| `softmax_rows` | `softmax.ts` | Row-wise softmax or log-softmax with an online (max, sum-exp) pass, one workgroup per row, any column count. |
| `softmax_backward` | `softmax.ts` | `gradIn = out * (g - sum(g * out))` or the log-softmax form `g - exp(out) * sum(g)`. |
| `layernorm_rows` | `layernorm.ts` | LayerNorm forward over rows; writes `mean` and `rstd` per row for the backward pass. |
| `layernorm_backward_x` | `layernorm.ts` | Input gradient per row (two row reductions). |
| `layernorm_gb_partial` | `layernorm.ts` | Per (column, row-chunk) partial sums of `g * xhat` and `g`. |
| `layernorm_gb_final` | `layernorm.ts` | Sums the chunk partials into `gradGamma` / `gradBeta` in a fixed order (overwrites, does not accumulate). |
| `gather` | `indexing.ts` | Embedding lookup `out[i, :] = table[ids[i], :]`; out-of-range ids produce a zero row. |
| `scatter_hist` | `indexing.ts` | Step 1 of the deterministic scatter-add: `atomicAdd<u32>` histogram of the ids. |
| `scan_exclusive` | `indexing.ts` | Step 2: single-workgroup exclusive scan of the histogram into segment offsets. |
| `scatter_place` | `indexing.ts` | Step 3: places each source row index into its id's segment (atomic cursor, arbitrary order inside the segment). |
| `scatter_rank` | `indexing.ts` | Step 4: re-orders every segment ascending by source index so the summation order is fixed. |
| `scatter_segsum` | `indexing.ts` | Step 5: one thread per (table row, column) sums its segment in index order and accumulates into the gradient table. Exact f32 sums, bit-reproducible run to run; no f32 atomics needed. |
| `cross_entropy_rows` | `loss_optim.ts` | Fused log-sum-exp + NLL per row; `ignoreIndex` and out-of-range targets contribute 0. |
| `cross_entropy_rows_grad` | `loss_optim.ts` | Same, plus `grad = (softmax - onehot) / denom` written in the same pass. |
| `adamw` | `loss_optim.ts` | Fused AdamW update with decoupled weight decay and bias correction (PyTorch semantics), one dispatch per parameter. |
| `rng` | `rng.ts` | Counter-based generator (`splitmix32(seed ^ (i * 0x9E3779B9))`) in three modes: `rand` (uniform, 24-bit), `randn` (Box-Muller on index pairs), `dropoutMask` (0/1 keep mask). Matches the CPU reference bit-for-bit for `rand` and `dropoutMask`. |

Shared WGSL snippets live in `common.ts`: `WGSL_MATH` (tanh/sigmoid/gelu/pow helpers that avoid overflow and
define `pow` for negative bases with integral exponents), `WGSL_HASH` (the RNG) and `WGSL_WG_REDUCE`
(workgroup-wide sum/max/min and an online max/sum-exp merge).

## Dispatch model

Kernels are recorded into one `GPUCommandEncoder` by `CommandBatch` (`batch.ts`) and submitted on readback, on
`sync()`, when the 512-slot uniform ring is exhausted, or after 64 dispatches. Buffers come from a size-class pool
(`bufferpool.ts`); a buffer freed while in flight is returned to the pool only after `queue.onSubmittedWorkDone()`
for the submission that last touched it.

## Parity

Every kernel runs on both backends with random inputs in the parity suite (`src/backend/webgpu/parity.ts`, the
Tests page, `npm run test:gpu`). The threshold is 1e-4 absolute or relative, 1e-3 for matmul with K > 256, and
scaled by `sqrt(n)` for reductions. `tests/wgsl-syntax.test.ts` checks every template for balanced braces, resolved
placeholders and declared bindings without a GPU.
