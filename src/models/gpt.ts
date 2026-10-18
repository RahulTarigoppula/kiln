/**
 * GPT: a decoder-only transformer with nanoGPT's module layout and parameter names
 * (`transformer.wte.weight`, `transformer.h.0.attn.c_attn.weight`, `lm_head.weight` tied to `wte`).
 */
import type { Device } from "../backend/types";
import { Tensor } from "../tensor/tensor";
import { escape, tidy } from "../tensor/scope";
import { Random, nextSeed } from "../tensor/random";
import { noGrad } from "../autograd/graph";
import { cat, crossEntropy } from "../autograd/ops";
import { Module, ModuleList } from "../nn/module";
import { Embedding } from "../nn/embedding";
import { Linear } from "../nn/linear";
import { LayerNorm } from "../nn/layernorm";
import { Dropout } from "../nn/dropout";
import { Block } from "../nn/transformer";

/** Model configuration (nanoGPT-compatible naming). */
export interface GPTConfig {
  vocabSize: number;
  /** Maximum context length. */
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  dropout: number;
  /** Biases in Linear and LayerNorm layers. */
  bias: boolean;
}

/** Size presets (layers, heads, width). */
export const GPT_PRESETS = {
  nano: { nLayer: 2, nHead: 2, nEmbd: 64 },
  micro: { nLayer: 4, nHead: 4, nEmbd: 128 },
  mini: { nLayer: 6, nHead: 6, nEmbd: 192 },
  small: { nLayer: 8, nHead: 8, nEmbd: 256 },
} as const satisfies Record<string, Pick<GPTConfig, "nLayer" | "nHead" | "nEmbd">>;

/** Preset name. */
export type GPTPreset = keyof typeof GPT_PRESETS;

/** Build a full config from a preset plus overrides (`vocabSize` and `blockSize` are required). */
export function gptConfig(preset: GPTPreset, overrides: Pick<GPTConfig, "vocabSize" | "blockSize"> & Partial<GPTConfig>): GPTConfig {
  return { dropout: 0, bias: true, ...GPT_PRESETS[preset], ...overrides };
}

/** Validate an untrusted config object (e.g. from a checkpoint sidecar). */
export function parseGPTConfig(raw: unknown): GPTConfig {
  const r = raw as Record<string, unknown>;
  const num = (k: string): number => {
    const v = r[k];
    if (typeof v !== "number" || !Number.isFinite(v)) throw new Error(`GPTConfig: "${k}" must be a number`);
    return v;
  };
  const cfg: GPTConfig = {
    vocabSize: num("vocabSize"), blockSize: num("blockSize"), nLayer: num("nLayer"), nHead: num("nHead"), nEmbd: num("nEmbd"),
    dropout: typeof r.dropout === "number" ? r.dropout : 0,
    bias: typeof r.bias === "boolean" ? r.bias : true,
  };
  if (cfg.nEmbd % cfg.nHead !== 0) throw new Error("GPTConfig: nEmbd must be divisible by nHead");
  return cfg;
}

/** Output of {@link GPT.forward}. */
export interface GPTOutput {
  /** [B, T, V] (or [B, 1, V] when only the last position is computed). */
  logits: Tensor;
  /** Mean cross-entropy when targets were given. */
  loss?: Tensor;
}

/** Options for {@link GPT.forward}. */
export interface GPTForwardOptions {
  /** Compute logits for every position even without targets (default: last position only). */
  allPositions?: boolean;
  /** Target id ignored by the loss (default −1, as in nanoGPT). */
  ignoreIndex?: number;
}

/** Options for {@link GPT.generate}. */
export interface GenerateOptions {
  /** Softmax temperature; 0 (or less) means greedy. Default 1. */
  temperature?: number;
  /** Keep only the k most likely tokens. */
  topK?: number;
  /** Nucleus sampling: keep the smallest set of tokens whose probability mass ≥ topP. */
  topP?: number;
  /** Sampler seed (default: drawn from the global generator). */
  seed?: number;
  /** Called after each generated token with its id, the sampling distribution and the batch row. */
  onToken?: (id: number, probs: Float32Array, row: number) => void;
}

/** The embedding + blocks + final norm stack (`transformer.*`). */
class Transformer extends Module {
  wte: Embedding;
  wpe: Embedding;
  drop: Dropout;
  h: ModuleList<Block>;
  ln_f: LayerNorm;

  constructor(cfg: GPTConfig, device?: Device) {
    super();
    this.wte = this.child("wte", new Embedding(cfg.vocabSize, cfg.nEmbd, { device }));
    this.wpe = this.child("wpe", new Embedding(cfg.blockSize, cfg.nEmbd, { device }));
    this.drop = this.child("drop", new Dropout(cfg.dropout));
    const blocks: Block[] = [];
    for (let i = 0; i < cfg.nLayer; i++) {
      blocks.push(new Block({ nEmbd: cfg.nEmbd, nHead: cfg.nHead, blockSize: cfg.blockSize, dropout: cfg.dropout, bias: cfg.bias, nLayer: cfg.nLayer, device }));
    }
    this.h = this.child("h", new ModuleList(blocks));
    this.ln_f = this.child("ln_f", new LayerNorm(cfg.nEmbd, { bias: cfg.bias, device }));
  }
}

/** GPT language model. */
export class GPT extends Module {
  readonly config: GPTConfig;
  transformer: Transformer;
  /** Output head; its weight *is* `transformer.wte.weight`. */
  lm_head: Linear;
  private posCache = new Map<string, Tensor>();

  constructor(config: GPTConfig, device?: Device) {
    super();
    if (config.nEmbd % config.nHead !== 0) throw new Error("GPT: nEmbd must be divisible by nHead");
    this.config = { ...config };
    this.transformer = this.child("transformer", new Transformer(config, device));
    this.lm_head = this.child("lm_head", new Linear(config.nEmbd, config.vocabSize, { bias: false, device }));
    this.lm_head.weight.free();
    this.lm_head.weight = this.lm_head.registerParameter("weight", this.transformer.wte.weight); // weight tying
  }

  /** Build from a preset. */
  static fromPreset(preset: GPTPreset, overrides: Pick<GPTConfig, "vocabSize" | "blockSize"> & Partial<GPTConfig>, device?: Device): GPT {
    return new GPT(gptConfig(preset, overrides), device);
  }

  /** i32 [T] position ids on the model device (cached). */
  private positions(T: number, device: Device): Tensor {
    const key = `${T}@${device}`;
    let p = this.posCache.get(key);
    if (!p) {
      p = escape(Tensor.arange(0, T, 1, { dtype: "i32", device: "cpu" }).to(device));
      this.posCache.set(key, p);
    }
    return p;
  }

  /**
   * idx: i32 [B, T] token ids; targets: optional i32 [B, T]. With targets, logits cover every position and
   * `loss` is the fused mean cross-entropy over B·T rows; without them only the last position's logits are
   * computed (cheap generation) unless `allPositions` is set.
   */
  forward(idx: Tensor, targets?: Tensor, opts: GPTForwardOptions = {}): GPTOutput {
    if (idx.dtype !== "i32" || idx.ndim !== 2) throw new Error("GPT.forward: idx must be an i32 [B, T] tensor");
    const [B, T] = idx.shape as [number, number];
    if (T > this.config.blockSize) throw new Error(`GPT.forward: sequence length ${T} exceeds blockSize ${this.config.blockSize}`);
    const tr = this.transformer;
    const tok = tr.wte.forward(idx); // [B, T, C]
    const pos = tr.wpe.forward(this.positions(T, idx.device)); // [T, C]
    let x = tr.drop.forward(tok.add(pos));
    for (const block of tr.h) x = block.forward(x);
    x = tr.ln_f.forward(x);
    if (targets) {
      const logits = this.lm_head.forward(x);
      const V = this.config.vocabSize;
      const loss = crossEntropy(logits.reshape(B * T, V), targets.reshape(B * T), { ignoreIndex: opts.ignoreIndex ?? -1 });
      return { logits, loss };
    }
    const h = opts.allPositions ? x : x.slice(1, T - 1, T);
    return { logits: this.lm_head.forward(h) };
  }

  /**
   * Autoregressive sampling under `noGrad`: feeds the last `blockSize` tokens, reads back the last-position
   * logits (one small readback per token) and samples on the CPU. Returns i32 [B, T + maxNewTokens].
   * Put the model in `eval()` mode first to disable dropout.
   */
  async generate(idx: Tensor, maxNewTokens: number, opts: GenerateOptions = {}): Promise<Tensor> {
    if (idx.dtype !== "i32" || idx.ndim !== 2) throw new Error("GPT.generate: idx must be an i32 [B, T] tensor");
    const device = this.device;
    const rng = new Random(opts.seed ?? nextSeed());
    const B = idx.shape[0] as number;
    const V = this.config.vocabSize;
    let cur = idx.device === device ? idx : idx.to(device);
    for (let i = 0; i < maxNewTokens; i++) {
      const T = cur.shape[1] as number;
      const seq = cur;
      const logits = noGrad(() => tidy(() => {
        const cond = T <= this.config.blockSize ? seq : seq.slice(1, T - this.config.blockSize, T);
        return this.forward(cond).logits.reshape(B, V);
      }));
      const data = (await logits.data()) as Float32Array;
      logits.free();
      const next = new Int32Array(B);
      for (let b = 0; b < B; b++) {
        const { id, probs } = sampleLogits(data.subarray(b * V, (b + 1) * V), rng, opts);
        next[b] = id;
        opts.onToken?.(id, probs, b);
      }
      const nextCpu = Tensor.fromArray(next, [B, 1], { dtype: "i32", device: "cpu" });
      const nextT = nextCpu.to(device);
      const appended = noGrad(() => cat([cur, nextT], 1));
      if (nextT !== nextCpu) nextCpu.free();
      nextT.free();
      if (cur !== idx) cur.free();
      cur = appended;
    }
    return cur;
  }

  /** Parameter count; `nonEmbedding` (default true) subtracts the position embeddings like nanoGPT. */
  override numParameters(nonEmbedding = true): number {
    let n = super.numParameters();
    if (nonEmbedding) n -= this.transformer.wpe.weight.numel;
    return n;
  }

  /** Forward+backward FLOPs per token (PaLM appendix B / nanoGPT `estimate_mfu`): 6N + 12·L·H·Q·T. */
  flopsPerToken(T = this.config.blockSize): number {
    const { nLayer: L, nHead: H, nEmbd } = this.config;
    const Q = nEmbd / H;
    return 6 * this.numParameters() + 12 * L * H * Q * T;
  }

  /** Toggle attention-weight capture on every layer (see `CausalSelfAttention.lastAttention`). */
  set captureAttention(on: boolean) {
    for (const b of this.transformer.h) b.attn.captureAttention = on;
  }

  get captureAttention(): boolean {
    const first = this.transformer.h.items[0];
    return first ? first.attn.captureAttention : false;
  }

  /** Last captured attention weights per layer ([B, nHead, T, T] or null). */
  attentionWeights(): (Tensor | null)[] {
    return this.transformer.h.items.map((b) => b.attn.lastAttention);
  }

  override extraRepr(): string {
    const c = this.config;
    return `vocab=${c.vocabSize}, block=${c.blockSize}, L=${c.nLayer}, H=${c.nHead}, C=${c.nEmbd}, params=${this.numParameters()}`;
  }
}

/** Sample one token id from raw logits on the CPU; also returns the (post temperature/top-k/top-p) distribution. */
export function sampleLogits(logits: Float32Array, rng: Random, opts: GenerateOptions = {}): { id: number; probs: Float32Array } {
  const V = logits.length;
  const probs = new Float32Array(V);
  const temperature = opts.temperature ?? 1;
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < V; i++) if ((logits[i] as number) > (logits[best] as number)) best = i;
    probs[best] = 1;
    return { id: best, probs };
  }
  const scaled = new Float64Array(V);
  let max = -Infinity;
  for (let i = 0; i < V; i++) {
    scaled[i] = (logits[i] as number) / temperature;
    if ((scaled[i] as number) > max) max = scaled[i] as number;
  }
  let order: number[] | null = null;
  if ((opts.topK !== undefined && opts.topK < V) || opts.topP !== undefined) {
    order = Array.from({ length: V }, (_, i) => i).sort((a, b) => (scaled[b] as number) - (scaled[a] as number));
  }
  if (opts.topK !== undefined && opts.topK < V && order) {
    const k = Math.max(1, Math.floor(opts.topK));
    for (let r = k; r < V; r++) scaled[order[r] as number] = -Infinity;
  }
  let sum = 0;
  for (let i = 0; i < V; i++) {
    const e = (scaled[i] as number) === -Infinity ? 0 : Math.exp((scaled[i] as number) - max);
    probs[i] = e;
    sum += e;
  }
  for (let i = 0; i < V; i++) probs[i] = (probs[i] as number) / sum;
  if (opts.topP !== undefined && opts.topP < 1 && order) {
    let cum = 0;
    let cut = V;
    for (let r = 0; r < V; r++) {
      cum += probs[order[r] as number] as number;
      if (cum >= opts.topP) {
        cut = r + 1;
        break;
      }
    }
    let kept = 0;
    for (let r = cut; r < V; r++) probs[order[r] as number] = 0;
    for (let i = 0; i < V; i++) kept += probs[i] as number;
    for (let i = 0; i < V; i++) probs[i] = (probs[i] as number) / kept;
  }
  const u = rng.uniform();
  let acc = 0;
  let id = V - 1;
  for (let i = 0; i < V; i++) {
    acc += probs[i] as number;
    if (u < acc) {
      id = i;
      break;
    }
  }
  if ((probs[id] as number) === 0) {
    // Floating-point tail: fall back to the most likely token.
    let best = 0;
    for (let i = 1; i < V; i++) if ((probs[i] as number) > (probs[best] as number)) best = i;
    id = best;
  }
  return { id, probs };
}
