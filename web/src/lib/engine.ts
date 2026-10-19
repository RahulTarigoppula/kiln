/**
 * Engine adapter - the ONLY place the web app touches the model/training layer.
 *
 * Tokenizers, datasets and safetensors parsing come from `src/data` and `src/models/checkpoint`; model
 * instantiation, generation, attention capture and training are thin wrappers over `GPT`, `saveGPT`/`loadGPT`
 * and `Trainer` (`src/models`, `src/train`). Every forward/backward pass runs inside the engine's `tidy`
 * scopes, so device memory stays flat across steps (see ARCHITECTURE.md § Memory management).
 */
import type { Device } from "../../../src/backend/types";
import { Tensor } from "../../../src/tensor/tensor";
import { manualSeed } from "../../../src/tensor/random";
import { tidy } from "../../../src/tensor/scope";
import { noGrad } from "../../../src/autograd/graph";
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
  type TrainingLogEntry,
} from "../../../src/models/checkpoint";
import { GPT, GPT_PRESETS, type GPTConfig, type GPTPreset } from "../../../src/models/gpt";
import { loadGPTFromRecords, saveGPT } from "../../../src/models/io";
import { configureAdamW } from "../../../src/optim/adamw";
import { makeCosineWithWarmup } from "../../../src/optim/schedulers";
import { Trainer } from "../../../src/train/trainer";

export type { Tokenizer, TokenizerJSON, FetchedCheckpoint, KilnCheckpointMeta, TensorRecord, GPTConfig };

/** True: the model engine is imported and every adapter function below is live. */
export const ENGINE_WIRED = true;

/** Kept for the UI's error classification; nothing throws it any more. */
export class EngineNotWiredError extends Error {
  constructor(readonly feature: string) {
    super(`${feature} is not available in this build.`);
    this.name = "EngineNotWiredError";
  }
}

export function isEngineNotWired(err: unknown): err is EngineNotWiredError {
  return err instanceof EngineNotWiredError || (err instanceof Error && err.name === "EngineNotWiredError");
}

// ---------------------------------------------------------------- model config & presets

export type PresetName = GPTPreset;

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

/** UI defaults layered over the engine's size presets (`GPT_PRESETS` in src/models/gpt.ts). */
const PRESET_UI: Record<PresetName, Pick<Preset, "blockSize" | "batchSize" | "lr" | "maxSteps">> = {
  nano: { blockSize: 64, batchSize: 32, lr: 1e-3, maxSteps: 1500 },
  micro: { blockSize: 128, batchSize: 32, lr: 1e-3, maxSteps: 2000 },
  mini: { blockSize: 256, batchSize: 32, lr: 5e-4, maxSteps: 5000 },
  small: { blockSize: 256, batchSize: 32, lr: 3e-4, maxSteps: 8000 },
};

export const PRESET_NAMES: PresetName[] = Object.keys(GPT_PRESETS) as PresetName[];

export const PRESETS: Record<PresetName, Preset> = Object.fromEntries(
  PRESET_NAMES.map((name) => {
    const size = GPT_PRESETS[name];
    const ui = PRESET_UI[name];
    return [name, { name, ...size, ...ui, blurb: `${size.nLayer} layers · d${size.nEmbd}` } satisfies Preset];
  }),
) as Record<PresetName, Preset>;

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

// ---------------------------------------------------------------- tokenizers & data

export type TokenizerKind = "char" | "bpe";

export interface TokenizerSpec {
  kind: TokenizerKind;
  /** BPE target vocabulary size (ignored for char). */
  vocabSize: number;
}

/** Build a tokenizer from text (char vocab or trained byte-level BPE). */
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

/** Encode `text` and split 90/10 into train/val token datasets. */
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

// ---------------------------------------------------------------- checkpoints

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
  // Dev servers / GitHub Pages SPA fallbacks answer missing files with index.html - treat that as 404.
  const probe = await fetch(url, { method: "HEAD", signal }).catch(() => null);
  if (!probe || !probe.ok || (probe.headers.get("content-type") ?? "").includes("text/html")) {
    throw new Error(`fetchCheckpoint: ${url} -> HTTP ${probe?.ok ? 404 : probe?.status ?? 404}`);
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

/** The token-embedding table (vocab x nEmbd) of a checkpoint, if recognisable. */
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

// ---------------------------------------------------------------- generation & inspection

export interface GenerateOptions {
  maxNewTokens: number;
  temperature: number;
  /** 0 = off. */
  topK: number;
  /** ≥ 1 = off. */
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
  /** weights[layer] is nHead x T x T, row-major. */
  weights: Float32Array[];
  nLayer: number;
  nHead: number;
  T: number;
}

/** Handle to an instantiated model. */
export interface ModelHandle {
  readonly config: GPTConfig;
  readonly tokenizer: Tokenizer;
  readonly device: Device;
  /** Non-embedding parameter count (nanoGPT convention). */
  readonly paramCount: number;
  readonly impl: GPT;
}

/** Rebuild the GPT from a parsed checkpoint on `device` (eval mode; dropout off). */
export async function instantiateModel(ckpt: LoadedCheckpoint, device: Device): Promise<ModelHandle> {
  if (!ckpt.meta) throw new Error("This checkpoint has no .json sidecar - the model config and tokenizer are needed to rebuild it.");
  if (!ckpt.tokenizer) throw new Error("The sidecar has no recognisable tokenizer (expected type \"char\" or \"bpe\").");
  const loaded = loadGPTFromRecords(ckpt.records, ckpt.meta, device);
  loaded.model.eval();
  if (device !== "cpu") await loaded.model.parameters()[0]?.backend.sync(); // surface GPU errors here, not mid-generation
  return { config: loaded.config, tokenizer: ckpt.tokenizer, device, paramCount: loaded.model.numParameters(), impl: loaded.model };
}

/** Upload a flat i32 id array as a [1, T] tensor on `device`. */
function uploadIds(ids: ArrayLike<number>, device: Device): Tensor {
  const arr = ids.length ? Int32Array.from(ids) : Int32Array.from([0]);
  const cpu = Tensor.fromArray(arr, [1, arr.length], { dtype: "i32", device: "cpu" });
  const dev = cpu.to(device);
  if (dev !== cpu) cpu.free();
  return dev;
}

const TOP_N = 5;

/** The top-N entries of a probability row. */
function topAlternatives(probs: Float32Array, tokenizer: Tokenizer, n = TOP_N): TokenAlternative[] {
  const best: number[] = [];
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i] as number;
    if (p <= 0) continue;
    let pos = best.length;
    while (pos > 0 && (probs[best[pos - 1] as number] as number) < p) pos--;
    if (pos < n) {
      best.splice(pos, 0, i);
      if (best.length > n) best.pop();
    }
  }
  return best.map((id) => ({ id, text: tokenizer.decode([id]), prob: probs[id] as number }));
}

function engineGenerateOptions(opts: GenerateOptions, vocabSize: number): { temperature: number; topK?: number; topP?: number; seed?: number } {
  const out: { temperature: number; topK?: number; topP?: number; seed?: number } = { temperature: opts.temperature };
  if (opts.topK > 0 && opts.topK < vocabSize) out.topK = Math.floor(opts.topK);
  if (opts.topP > 0 && opts.topP < 1) out.topP = opts.topP;
  if (opts.seed !== undefined) out.seed = opts.seed;
  return out;
}

/** Yield one macrotask so the page can paint between tokens (readbacks on the CPU backend resolve immediately). */
function nextMacrotask(): Promise<void> {
  if (typeof MessageChannel !== "undefined") {
    return new Promise((resolve) => {
      const ch = new MessageChannel();
      ch.port1.onmessage = () => {
        ch.port1.close();
        resolve();
      };
      ch.port2.postMessage(null);
    });
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Stream tokens from `model.generate`. Each `onToken` callback is turned into a {@link TokenEvent} (probability of
 * the sampled id + the top-5 of the softmax row) and handed to the consumer; the engine waits for the consumer
 * before computing the next token, so an abort takes effect immediately and the UI gets a paint per token.
 */
async function* streamTokens(model: GPT, tokenizer: Tokenizer, promptIds: Int32Array, opts: GenerateOptions, signal?: AbortSignal): AsyncGenerator<TokenEvent> {
  const device = model.device;
  const wasTraining = model.training;
  model.eval();
  const idx = uploadIds(promptIds, device);
  const queue: TokenEvent[] = [];
  // Handshake between the engine's onToken (producer) and this generator (consumer).
  const hs: { release: (() => void) | null; wake: (() => void) | null; finished: boolean } = { release: null, wake: null, finished: false };
  const ctrl = new AbortController();
  const onAbort = (): void => {
    ctrl.abort();
    hs.release?.();
    hs.wake?.();
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  if (signal?.aborted) onAbort();
  const run = model
    .generate(idx, opts.maxNewTokens, {
      ...engineGenerateOptions(opts, model.config.vocabSize),
      signal: ctrl.signal,
      onToken: (id, probs) => {
        queue.push({ id, text: tokenizer.decode([id]), prob: probs[id] as number, top: topAlternatives(probs, tokenizer) });
        hs.wake?.();
        hs.wake = null;
        return new Promise<void>((r) => {
          hs.release = r;
        });
      },
    })
    .then((out) => out.free());
  void run.finally(() => {
    hs.finished = true;
    hs.wake?.();
    hs.wake = null;
  });
  try {
    for (;;) {
      if (!queue.length) {
        if (hs.finished) break;
        await new Promise<void>((r) => {
          hs.wake = r;
        });
        continue;
      }
      const ev = queue.shift() as TokenEvent;
      await nextMacrotask();
      yield ev;
      if (ctrl.signal.aborted) break;
      const r = hs.release;
      hs.release = null;
      r?.();
    }
    await run;
  } finally {
    ctrl.abort();
    hs.release?.();
    signal?.removeEventListener("abort", onAbort);
    await run.catch(() => undefined);
    idx.free();
    model.train(wasTraining);
  }
}

/** Stream tokens from an instantiated model. */
export function generateTokens(model: ModelHandle, promptIds: Int32Array, opts: GenerateOptions, signal?: AbortSignal): AsyncGenerator<TokenEvent> {
  return streamTokens(model.impl, model.tokenizer, promptIds, opts, signal);
}

/** One forward pass with attention capture; returns every layer's [nHead, T, T] weights for the prompt. */
export async function attentionMaps(model: ModelHandle, ids: Int32Array): Promise<AttentionMaps> {
  const gpt = model.impl;
  if (ids.length === 0) throw new Error("attentionMaps: empty prompt");
  if (ids.length > model.config.blockSize) throw new Error(`attentionMaps: ${ids.length} tokens exceed blockSize ${model.config.blockSize}`);
  const wasTraining = gpt.training;
  gpt.eval();
  gpt.captureAttention = true;
  const idx = uploadIds(ids, gpt.device);
  try {
    tidy(() => noGrad(() => gpt.forward(idx).logits)).free();
    const layers = gpt.attentionWeights();
    const weights: Float32Array[] = [];
    for (const t of layers) {
      if (!t) throw new Error("attentionMaps: a layer produced no attention weights");
      weights.push((await t.data()) as Float32Array);
    }
    return { weights, nLayer: model.config.nLayer, nHead: model.config.nHead, T: ids.length };
  } finally {
    for (const block of gpt.transformer.h.items) {
      block.attn.lastAttention?.free();
      block.attn.lastAttention = null;
    }
    gpt.captureAttention = false;
    idx.free();
    gpt.train(wasTraining);
  }
}

// ---------------------------------------------------------------- training

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
  gradNorm?: number;
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
  /** Serialise the current weights + sidecar meta (weights only; the sidecar carries tokenizer, config and log). */
  exportCheckpoint(): Promise<{ safetensors: Uint8Array; meta: KilnCheckpointMeta }>;
  /** Sample from the in-training model. */
  generate(promptIds: Int32Array, opts: GenerateOptions, signal?: AbortSignal): AsyncGenerator<TokenEvent>;
  /** Steps completed so far. */
  readonly step: number;
}

/** Steps between loss readbacks on each device (the only host sync in the loop). */
const LOSS_READBACK_INTERVAL: Record<Device, number> = { cpu: 1, webgpu: 2 };

/** Thin a long per-step log for the sidecar (every entry with a valLoss is kept). */
export function compactTrainingLog(log: TrainingLogEntry[], maxEntries = 1500): TrainingLogEntry[] {
  const round = (e: TrainingLogEntry): TrainingLogEntry => {
    const out: TrainingLogEntry = { step: e.step, loss: Number(e.loss.toFixed(4)) };
    if (e.valLoss !== undefined) out.valLoss = Number(e.valLoss.toFixed(4));
    if (e.lr !== undefined) out.lr = Number(e.lr.toPrecision(4));
    if (e.tokensPerSec !== undefined) out.tokensPerSec = Math.round(e.tokensPerSec);
    if (e.msPerStep !== undefined) out.msPerStep = Number(e.msPerStep.toFixed(1));
    return out;
  };
  if (log.length <= maxEntries) return log.map(round);
  const stride = Math.ceil(log.length / maxEntries);
  return log.filter((e, i) => e.valLoss !== undefined || i % stride === 0 || i === log.length - 1).map(round);
}

/** Build a GPT + AdamW + Trainer for a Studio job. */
export async function createTrainer(job: TrainJob, callbacks: TrainerCallbacks): Promise<TrainerHandle> {
  const { hyper, device, config } = job;
  manualSeed(hyper.seed);
  const model = new GPT(config, device);
  const optimizer = configureAdamW(model, { lr: hyper.lr, weightDecay: hyper.weightDecay, betas: [0.9, 0.95] });
  const lrSchedule = makeCosineWithWarmup({ warmup: hyper.warmupSteps, total: hyper.maxSteps, lrMax: hyper.lr, lrMin: hyper.lr / 10 });
  const trainer = new Trainer({
    model,
    optimizer,
    train: job.train,
    val: job.val,
    tokenizer: job.tokenizer,
    batchSize: hyper.batchSize,
    maxSteps: hyper.maxSteps,
    lrSchedule,
    evalInterval: hyper.evalInterval,
    evalIters: hyper.evalIters,
    clipNorm: hyper.gradClip,
    sampleInterval: hyper.sampleInterval,
    sampleTokens: hyper.sampleTokens,
    samplePrompt: "\n",
    sampleTemperature: 0.8,
    sampleTopK: 40,
    device,
    seed: hyper.seed,
    lossReadbackInterval: LOSS_READBACK_INTERVAL[device],
    callbacks: {
      onStep: (s) => {
        const m: StepMetrics = { step: s.step, loss: s.loss, lr: s.lr, tokensPerSec: s.tokensPerSec, msPerStep: s.msPerStep };
        if (s.gradNorm !== undefined) m.gradNorm = s.gradNorm;
        callbacks.onStep(m);
      },
      onEval: (e) => callbacks.onEval({ step: e.step, valLoss: e.valLoss, trainLoss: e.trainLoss }),
      onSample: (s) => callbacks.onSample(s),
    },
  });
  if (device !== "cpu") await model.parameters()[0]?.backend.sync(); // init kernels ran; surface GPU errors early

  return {
    get step() {
      return trainer.state.step;
    },
    async run() {
      await trainer.run();
    },
    pause: () => trainer.pause(),
    resume: () => trainer.resume(),
    stop: () => trainer.stop(),
    async exportCheckpoint() {
      const st = trainer.state;
      const lastVal = [...st.trainingLog].reverse().find((e) => e.valLoss !== undefined)?.valLoss;
      const { bytes, meta } = await saveGPT(model, job.tokenizer.toJSON(), compactTrainingLog(st.trainingLog), {
        step: st.step,
        extra: {
          device,
          hyper: { ...hyper },
          finalTrainLoss: st.lastLoss,
          finalValLoss: lastVal ?? null,
          trainTokens: job.train.length,
          valTokens: job.val.length,
        },
      });
      return { safetensors: bytes, meta };
    },
    generate: (promptIds, opts, signal) => streamTokens(model, job.tokenizer, promptIds, opts, signal),
  };
}
