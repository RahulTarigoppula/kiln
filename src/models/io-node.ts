/**
 * Node-only GPT checkpoint helpers (browser bundles must import `io.ts` instead).
 */
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { Device } from "../backend/types";
import type { GPT } from "./gpt";
import { loadGPTFromRecords, saveGPT, type LoadedGPT, type SaveGPTOptions } from "./io";
import { readCheckpointNode } from "./checkpoint-node";
import { sidecarUrl, type TrainingLogEntry } from "./checkpoint";

/** Save a GPT to `path` (`.safetensors`) and its sidecar `.json`. Returns the byte sizes. */
export async function saveGPTNode(path: string, model: GPT, tokenizer: unknown, trainingLog: TrainingLogEntry[] = [], opts: SaveGPTOptions = {}): Promise<{ path: string; metaPath: string; bytes: number }> {
  const { bytes, meta } = await saveGPT(model, tokenizer, trainingLog, opts);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, bytes);
  const metaPath = sidecarUrl(path);
  await writeFile(metaPath, JSON.stringify(meta));
  return { path, metaPath, bytes: bytes.byteLength };
}

/** Load a GPT checkpoint written by {@link saveGPTNode}. */
export async function loadGPTNode(path: string, device: Device = "cpu"): Promise<LoadedGPT> {
  const { records, meta } = await readCheckpointNode(path);
  if (!meta) throw new Error(`loadGPTNode: sidecar ${sidecarUrl(path)} missing or invalid`);
  return loadGPTFromRecords(records, meta, device);
}
