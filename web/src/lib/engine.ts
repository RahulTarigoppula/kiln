/**
 * Engine adapter — the ONLY place the web app touches the model/training layer.
 *
 * Everything that exists in phase 1 (tokenizers, dataset, safetensors checkpoints, backends) is wired
 * for real. Calls that need the phase-2 modules (`src/nn`, `src/optim`, `src/models/gpt.ts`,
 * `src/models/io.ts`, `src/train/trainer.ts`) are isolated behind the functions marked `TODO(phase-2)`
 * below and currently throw {@link EngineNotWiredError}; the UI renders that as an inline notice.
 *
 * Planned phase-2 API this adapter targets:
 *   new GPT(config) · GPT_PRESETS.{nano,micro,mini,small} · model.forward(idx, targets?) → {logits, loss?}
 *   model.generate(idx, maxNewTokens, {temperature, topK, topP, seed, onToken}) · model.captureAttention
 *   new Trainer({...}, {onStep, onEval, onSample}) with run()/pause()/resume()/stop() · saveGPT/loadGPT
 */
import type { Device } from "../../../src/backend/types";
import {
  ByteLevelBPE,
  CharTokenizer,
  TokenDataset,
  tokenizerFromJSON,
  type Tokenizer,
  type TokenizerJSON,
} from "../../../src/data/index";
import {
  fetchCheckpoint,
  parseSafetensors,
  isKilnCheckpointMeta,
  type FetchedCheckpoint,
  type KilnCheckpointMeta,
  type TensorRecord,
} from "../../../src/models/checkpoint";

export type { Tokenizer, TokenizerJSON, FetchedCheckpoint, KilnCheckpointMeta, TensorRecord };

/** Set to true once the phase-2 modules are imported below. Pages use it to label scaffolded actions. */
export const ENGINE_WIRED = false;

export class EngineNotWiredError extends Error {
  constructor(readonly feature: string) {
    super(`${feature} needs the phase-2 model engine (src/models/gpt.ts, src/train/trainer.ts). The UI is ready; wire it in web/src/lib/engine.ts.`);
    this.name = "EngineNotWiredError";
  }
}

export function isEngineNotWired(err: unknown): err is EngineNotWiredError {
  return err instanceof EngineNotWiredError || (err instanceof Error && err.name === "EngineNotWiredError");
}

// ---------------------------------------------------------------- model config & presets

/** nanoGPT-compatible configuration (see ARCHITECTURE.md § nn / optim / models). */
export interface GPTConfig {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  dropout: number;
  bias: boolean;
}

export type PresetName = "nano" | "micro" | "mini" | "small";

export interface Preset {
  name: PresetName;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  blockSize: number;
  /** Suggested hyper-parameters. */
  batchSize: number;
  lr: number;
  maxSteps: number;
  blurb: string;
}

/**
 * Model presets. TODO(phase-2): replace with `GPT_PRESETS` from `src/models/gpt.ts` once it lands
 * (keep the same names so Studio's picker needs no change).
 */
export const PRESETS: Record<PresetName, Preset> = {
  nano: { name: "nano", nLayer: 3, nHead: 3, nEmbd: 48, blockSize: 64, batchSize: 32, lr: 1e-3, maxSteps: 1500, blurb: "3 layers · d48" },
  micro: { name: "micro", nLayer: 4, nHead: 4, nEmbd: 128, blockSize: 128, batchSize: 32, lr: 6e-4, maxSteps: 3000, blurb: "4 layers · d128" },
  mini: { name: "mini", nLayer: 6, nHead: 6, nEmbd: 192, blockSize: 256, batchSize: 32, lr: 5e-4, maxSteps: 5000, blurb: "6 layers · d192" },
  small: { name: "small", nLayer: 8, nHead: 8, nEmbd: 256, blockSize: 256, batchSize: 32, lr: 3e-4, maxSteps: 8000, blurb: "8 layers · d256" },
};

export const PRESET_NAMES: PresetName[] = ["nano", "micro", "mini", "small"];

export function presetConfig(preset: Preset, vocabSize: number, overrides: Partial<GPTConfig> = {}): GPTConfig {
  return {
    vocabSize,
    blockSize: preset.blockSize,
    nLayer: preset.nLayer,
    nHead: preset.nHead,
    nEmbd: preset.nEmbd,
    dropout: 0.1,
    bias: true,
    ...overrides,
  };
}

/**
 * Parameter count for a weight-tied GPT: per block 12·d² (+ biases + 2 LayerNorms), final LayerNorm,
 * token embedding V·d (tied with the LM head) and positional embedding T·d.
 */
export function estimateParams(c: GPTConfig): number {
  const d = c.nEmbd;
  const perBlock = 12 * d * d + (c.bias ? 13 * d : 0) + 4 * d; // attn+mlp weights, biases, 2 LN (γ, β)
  return c.nLayer * perBlock + 2 * d + c.vocabSize * d + c.blockSize * d;
}

/** Rough f32 training-memory estimate (params + grads + 2 Adam moments + activations). */
export function estimateTrainingBytes(c: GPTConfig, batchSize: number): number {
  const params = estimateParams(c);
  const states = params * 4 * 4; // p, g, m, v
  const T = c.blockSize;
  const act = batchSize * T * (c.nLayer * (12 * c.nEmbd + 2 * c.nHead * T) + c.vocabSize) * 4;
  return states + act;
}

// ---------------------------------------------------------------- tokenizers & data (live)

export type TokenizerKind = "char" | "bpe";

export interface TokenizerSpec {
  kind: TokenizerKind;
  /** BPE target vocabulary size (ignored for char). */
  vocabSize: number;
}

/** Build a tokenizer from text (char vocab or trained byte-level BPE). Live in phase 1. */
export function buildTokenizer(spec: TokenizerSpec, text: string, onProgress?: (done: number, total: number) => void): Tokenizer {
  if (spec.kind === "char") return CharTokenizer.fromText(text);
  return ByteLevelBPE.train(text, spec.vocabSize, { onProgress });
}

export function reviveTokenizer(json: TokenizerJSON): Tokenizer {
  return tokenizerFromJSON(json);
}

export interface DatasetSummary {
  chars: number;
  tokens: number;
  vocabSize: number;
  trainTokens: number;
  valTokens: number;
  tokensPerChar: number;
}

/** Encode `text` and split 90/10 into train/val token datasets. Live in phase 1. */
export function buildDataset(tokenizer: Tokenizer, text: string, blockSize: number): { ids: Int32Array; train: TokenDataset; val: TokenDataset; summary: DatasetSummary } {
  const ids = tokenizer.encode(text);
  const ds = new TokenDataset(ids, blockSize);
  const { train, val } = ds.trainValSplit(0.9);
  const summary: DatasetSummary = {
    chars: text.length,
    tokens: ids.length,
    vocabSize: tokenizer.vocabSize,
    trainTokens: train.length,
    valTokens: val.length,
    tokensPerChar: text.length ? ids.length / text.length : 0,
  };
  return { ids, train, val, summary };
}

/** Quick stats without building a tokenizer (char vocab is cheap; BPE token count is estimated). */
export function quickTextStats(text: string, spec: TokenizerSpec): { chars: number; charVocab: number; estTokens: number } {
  const set = new Set<string>();
  const limit = Math.min(text.length, 2_000_000);
  for (let i = 0; i < limit; i++) set.add(text[i] ?? "");
  const chars = text.length;
  const estTokens = spec.kind === "char" ? chars : Math.round(chars / Math.min(4.2, 1.6 + Math.log2(Math.max(2, spec.vocabSize / 256)) * 0.55));
  return { chars, charVocab: set.size, estTokens };
}

// ---------------------------------------------------------------- checkpoints (live)

export interface LoadedCheckpoint extends FetchedCheckpoint {
  /** Where it came from (URL or file name). */
  source: string;
  config: GPTConfig | null;
  tokenizer: Tokenizer | null;
  byteLength: number;
}

function configFromMeta(meta: KilnCheckpointMeta | null): GPTConfig | null {
  const c = meta?.config;
  if (!c) return null;
  const num = (k: string): number | null => (typeof c[k] === "number" ? (c[k] as number) : null);
  const vocabSize = num("vocabSize") ?? num("vocab_size");
  const blockSize = num("blockSize") ?? num("block_size");
  const nLayer = num("nLayer") ?? num("n_layer");
  const nHead = num("nHead") ?? num("n_head");
  const nEmbd = num("nEmbd") ?? num("n_embd");
  if (vocabSize === null || blockSize === null || nLayer === null || nHead === null || nEmbd === null) return null;
  return { vocabSize, blockSize, nLayer, nHead, nEmbd, dropout: num("dropout") ?? 0, bias: c["bias"] !== false };
}

function tokenizerFromMeta(meta: KilnCheckpointMeta | null): Tokenizer | null {
  const t = meta?.tokenizer as { type?: string } | undefined;
  if (!t || (t.type !== "char" && t.type !== "bpe")) return null;
  try {
    return tokenizerFromJSON(t as TokenizerJSON);
  } catch {
    return null;
  }
}

function finishCheckpoint(fetched: FetchedCheckpoint, source: string): LoadedCheckpoint {
  const byteLength = fetched.records.reduce((n, r) => n + r.data.byteLength, 0);
  return { ...fetched, source, config: configFromMeta(fetched.meta), tokenizer: tokenizerFromMeta(fetched.meta), byteLength };
}

export async function loadCheckpointFromUrl(url: string, signal?: AbortSignal): Promise<LoadedCheckpoint> {
  // Dev servers / GitHub Pages SPA fallbacks answer missing files with index.html — treat that as 404.
  const probe = await fetch(url, { method: "HEAD", signal }).catch(() => null);
  if (!probe || !probe.ok || (probe.headers.get("content-type") ?? "").includes("text/html")) {
    throw new Error(`fetchCheckpoint: ${url} → HTTP ${probe?.ok ? 404 : probe?.status ?? 404}`);
  }
  const fetched = await fetchCheckpoint(url, { signal });
  return finishCheckpoint(fetched, url);
}

export async function loadCheckpointFromFiles(safetensors: File, metaFile: File | null): Promise<LoadedCheckpoint> {
  const parsed = parseSafetensors(new Uint8Array(await safetensors.arrayBuffer()));
  let meta: KilnCheckpointMeta | null = null;
  if (metaFile) {
    const json: unknown = JSON.parse(await metaFile.text());
    if (isKilnCheckpointMeta(json)) meta = json;
  }
  return finishCheckpoint({ ...parsed, meta }, safetensors.name);
}

/** The token-embedding table (vocab × nEmbd) of a checkpoint, if recognisable. */
export function findEmbeddingRecord(ckpt: LoadedCheckpoint): TensorRecord | null {
  const named = ckpt.records.find((r) => /(^|\.)(wte|tok_emb|token_embedding|embedding)(\.weight)?$/i.test(r.name) && r.shape.length === 2);
  if (named) return named;
  const V = ckpt.config?.vocabSize;
  if (V) {
    const byShape = ckpt.records.find((r) => r.shape.length === 2 && r.shape[0] === V);
    if (byShape) return byShape;
  }
  return ckpt.records.find((r) => r.shape.length === 2) ?? null;
}

// ---------------------------------------------------------------- generation & inspection (phase-2)

export interface GenerateOptions {
  maxNewTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  seed?: number;
}

export interface TokenAlternative {
  id: number;
  text: string;
  prob: number;
}

/** One streamed token with its sampling probability and the top-5 alternatives. */
export interface TokenEvent {
  id: number;
  text: string;
  prob: number;
  top: TokenAlternative[];
}

export interface AttentionMaps {
  /** weights[layer] is nHead × T × T, row-major. */
  weights: Float32Array[];
  nLayer: number;
  nHead: number;
  T: number;
}

/** Opaque handle to an instantiated model (phase-2 GPT instance). */
export interface ModelHandle {
  readonly config: GPTConfig;
  readonly tokenizer: Tokenizer;
  readonly device: Device;
  readonly paramCount: number;
  /** TODO(phase-2): the `GPT` instance. */
  readonly impl: unknown;
}

/**
 * TODO(phase-2): `const model = loadGPT(ckpt.records, ckpt.meta)` → `model.to(device)`.
 */
export async function instantiateModel(_ckpt: LoadedCheckpoint, _device: Device): Promise<ModelHandle> {
  throw new EngineNotWiredError("Model instantiation");
}

/**
 * TODO(phase-2): `model.generate(idx, maxNewTokens, {temperature, topK, topP, seed, onToken})`, yielding one
 * {@link TokenEvent} per sampled token (probability of the sampled id + top-5 of the softmax row).
 */
export async function* generateTokens(_model: ModelHandle, _promptIds: Int32Array, _opts: GenerateOptions, _signal?: AbortSignal): AsyncGenerator<TokenEvent> {
  throw new EngineNotWiredError("Generation");
}

/**
 * TODO(phase-2): `model.captureAttention = true; model.forward(idx)` → read the per-layer attention weights.
 */
export async function attentionMaps(_model: ModelHandle, _ids: Int32Array): Promise<AttentionMaps> {
  throw new EngineNotWiredError("Attention capture");
}

// ---------------------------------------------------------------- training (phase-2)

export interface TrainHyper {
  batchSize: number;
  lr: number;
  maxSteps: number;
  warmupSteps: number;
  weightDecay: number;
  gradClip: number;
  evalInterval: number;
  evalIters: number;
  sampleInterval: number;
  sampleTokens: number;
  seed: number;
}

export interface TrainJob {
  config: GPTConfig;
  tokenizer: Tokenizer;
  train: TokenDataset;
  val: TokenDataset;
  hyper: TrainHyper;
  device: Device;
}

export interface StepMetrics {
  step: number;
  loss: number;
  lr: number;
  tokensPerSec: number;
  msPerStep: number;
}

export interface EvalMetrics {
  step: number;
  valLoss: number;
  trainLoss?: number;
}

export interface SampleEvent {
  step: number;
  text: string;
}

export interface TrainerCallbacks {
  onStep(m: StepMetrics): void;
  onEval(m: EvalMetrics): void;
  onSample(s: SampleEvent): void;
}

export interface TrainerHandle {
  /** Resolves when training finishes or is stopped. */
  run(): Promise<void>;
  pause(): void;
  resume(): void;
  stop(): void;
  /** Serialise the current weights + sidecar meta. */
  exportCheckpoint(): Promise<{ safetensors: Uint8Array; meta: KilnCheckpointMeta }>;
  /** Sample from the in-training model. */
  generate(promptIds: Int32Array, opts: GenerateOptions, signal?: AbortSignal): AsyncGenerator<TokenEvent>;
}

/**
 * TODO(phase-2): `new Trainer({ model: new GPT(job.config).to(job.device), train, val, ...job.hyper }, callbacks)`.
 */
export async function createTrainer(_job: TrainJob, _callbacks: TrainerCallbacks): Promise<TrainerHandle> {
  throw new EngineNotWiredError("Training");
}
