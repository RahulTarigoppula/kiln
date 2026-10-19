/**
 * GPT checkpoints: model weights (+ optional optimizer state as `optim.<param>.<key>` records) in safetensors
 * plus the Kiln sidecar meta. Browser-safe; Node file helpers live in `io-node.ts`.
 *
 * Tied parameters are written once (under the first name `namedParameters()` reports) and the alias is
 * recorded in `meta.extra.tied` so `loadStateDict` can treat the missing alias as loaded.
 */
import type { Device } from "../backend/types";
import { Tensor } from "../tensor/tensor";
import type { Optimizer } from "../optim/optimizer";
import type { Module, StateDict } from "../nn/module";
import { GPT, parseGPTConfig, type GPTConfig } from "./gpt";
import {
  makeCheckpointMeta, parseSafetensors, serializeSafetensors,
  type KilnCheckpointMeta, type TensorRecord, type TrainingLogEntry,
} from "./checkpoint";

/** Extra sidecar fields Kiln writes. */
export interface GPTCheckpointExtra {
  /** Training step the checkpoint was taken at. */
  step?: number;
  /** alias name -> canonical name for tied parameters. */
  tied?: Record<string, string>;
  /** Optimizer hyper-parameters when `optim.*` records are present. */
  optimizer?: { type: string; step: number; defaults: Record<string, number>; groups: { size: number; hyper: Record<string, number> }[] };
  [key: string]: unknown;
}

/** Options for {@link saveGPT}. */
export interface SaveGPTOptions {
  /** Include this optimizer's state tensors and hyper-parameters. */
  optimizer?: Optimizer;
  step?: number;
  extra?: Record<string, unknown>;
}

/** Result of {@link saveGPT}. */
export interface SavedGPT {
  bytes: Uint8Array;
  meta: KilnCheckpointMeta;
}

async function toRecord(name: string, t: Tensor): Promise<TensorRecord> {
  const data = await t.data();
  return { name, dtype: t.dtype === "i32" ? "I32" : "F32", shape: [...t.shape], data: data as Float32Array | Int32Array };
}

/** Deduplicated model records + tied-name map (async: GPU tensors are read back). */
export async function moduleRecords(model: Module): Promise<{ records: TensorRecord[]; tied: Record<string, string> }> {
  const records: TensorRecord[] = [];
  const tied: Record<string, string> = {};
  const canonical = new Map<Tensor, string>();
  for (const [name, t] of model.stateDict()) {
    const seen = canonical.get(t);
    if (seen) {
      tied[name] = seen;
      continue;
    }
    canonical.set(t, name);
    records.push(await toRecord(name, t));
  }
  return { records, tied };
}

/** Optimizer state records named `optim.<param name>.<key>`. */
export async function optimizerRecords(optimizer: Optimizer, model: Module): Promise<TensorRecord[]> {
  const records: TensorRecord[] = [];
  const names = new Map<Tensor, string>(model.namedParameters().map(([n, p]) => [p, n]));
  for (const [p, state] of optimizer.stateTensors()) {
    const name = names.get(p);
    if (!name) continue;
    for (const [key, t] of Object.entries(state)) records.push(await toRecord(`optim.${name}.${key}`, t));
  }
  return records;
}

/** Serialise a GPT (and optionally its optimizer) to safetensors bytes + sidecar meta. */
export async function saveGPT(model: GPT, tokenizer: unknown, trainingLog: TrainingLogEntry[] = [], opts: SaveGPTOptions = {}): Promise<SavedGPT> {
  const { records, tied } = await moduleRecords(model);
  const extra: GPTCheckpointExtra = { ...(opts.extra ?? {}), tied };
  if (opts.step !== undefined) extra.step = opts.step;
  if (opts.optimizer) {
    records.push(...(await optimizerRecords(opts.optimizer, model)));
    extra.optimizer = opts.optimizer.stateDict() as GPTCheckpointExtra["optimizer"];
  }
  const meta = makeCheckpointMeta({ ...model.config } as unknown as Record<string, unknown>, tokenizer, trainingLog, extra);
  const bytes = serializeSafetensors(records, { format: "kiln", model: "gpt" });
  return { bytes, meta };
}

/** Records -> CPU tensors keyed by name (model records only, `optim.*` excluded). */
export function recordsToStateDict(records: readonly TensorRecord[]): StateDict {
  const sd: StateDict = new Map();
  for (const r of records) {
    if (r.name.startsWith("optim.")) continue;
    sd.set(r.name, Tensor.fromArray(r.data, r.shape, { device: "cpu", dtype: r.dtype === "I32" ? "i32" : "f32" }));
  }
  return sd;
}

/** Result of {@link loadGPT}. */
export interface LoadedGPT {
  model: GPT;
  config: GPTConfig;
  tokenizer: unknown;
  trainingLog: TrainingLogEntry[];
  /** Step stored in the sidecar (0 when absent). */
  step: number;
  /** Raw records so a trainer can restore optimizer state (`restoreOptimizer`). */
  records: TensorRecord[];
  meta: KilnCheckpointMeta;
}

/** Rebuild a GPT from safetensors bytes and its sidecar meta on `device`. */
export function loadGPT(bytes: Uint8Array | ArrayBuffer, meta: KilnCheckpointMeta, device: Device = "cpu"): LoadedGPT {
  return loadGPTFromRecords(parseSafetensors(bytes).records, meta, device);
}

/** Like {@link loadGPT} but from already-parsed records. */
export function loadGPTFromRecords(records: TensorRecord[], meta: KilnCheckpointMeta, device: Device = "cpu"): LoadedGPT {
  const config = parseGPTConfig(meta.config);
  const model = new GPT(config, device);
  const sd = recordsToStateDict(records);
  // Tied aliases: make them resolvable so strict loading succeeds even when the alias record is absent.
  const tied = (meta.extra as GPTCheckpointExtra | undefined)?.tied ?? {};
  for (const [alias, canonical] of Object.entries(tied)) {
    const t = sd.get(canonical);
    if (t && !sd.has(alias)) sd.set(alias, t);
  }
  model.loadStateDict(sd, true);
  for (const t of new Set(sd.values())) t.free();
  const extra = meta.extra as GPTCheckpointExtra | undefined;
  return { model, config, tokenizer: meta.tokenizer, trainingLog: [...meta.trainingLog], step: extra?.step ?? 0, records, meta };
}

/** Restore `optim.*` records (and hyper-parameters from `meta.extra.optimizer`) into `optimizer`. */
export function restoreOptimizer(optimizer: Optimizer, model: Module, records: readonly TensorRecord[], meta?: KilnCheckpointMeta): boolean {
  const byName = new Map<string, Tensor>(model.namedParameters());
  let found = false;
  const grouped = new Map<string, Record<string, Tensor>>();
  for (const r of records) {
    if (!r.name.startsWith("optim.")) continue;
    const dot = r.name.lastIndexOf(".");
    const paramName = r.name.slice("optim.".length, dot);
    const key = r.name.slice(dot + 1);
    let g = grouped.get(paramName);
    if (!g) {
      g = {};
      grouped.set(paramName, g);
    }
    g[key] = Tensor.fromArray(r.data, r.shape, { device: "cpu", dtype: r.dtype === "I32" ? "i32" : "f32" });
    found = true;
  }
  for (const [paramName, state] of grouped) {
    const p = byName.get(paramName);
    if (!p) throw new Error(`restoreOptimizer: checkpoint has state for unknown parameter "${paramName}"`);
    optimizer.loadParamState(p, state);
    for (const t of Object.values(state)) t.free();
  }
  const hyper = (meta?.extra as GPTCheckpointExtra | undefined)?.optimizer;
  if (hyper) optimizer.loadStateDict(hyper as Parameters<typeof optimizer.loadStateDict>[0]);
  return found;
}
