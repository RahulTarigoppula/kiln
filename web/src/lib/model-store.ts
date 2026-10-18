/**
 * Shared checkpoint / model state for Playground and Inspect (main thread).
 */
import { signal } from "./signals";
import { gpuState } from "./gpu";
import {
  instantiateModel,
  isEngineNotWired,
  loadCheckpointFromFiles,
  loadCheckpointFromUrl,
  type LoadedCheckpoint,
  type ModelHandle,
} from "./engine";

export const SHIPPED_CHECKPOINT = {
  id: "shakespeare-tiny",
  label: "shakespeare-tiny (shipped)",
  url: `${import.meta.env.BASE_URL}checkpoints/shakespeare-tiny.safetensors`,
  blurb: "char-level · 4 layers · d128 · trained on TinyShakespeare",
};

export const checkpoint = signal<LoadedCheckpoint | null>(null);
export const model = signal<ModelHandle | null>(null);
export const loading = signal<string | null>(null);
export const loadError = signal<string | null>(null);
/** Set when the checkpoint loaded but the model engine is not wired yet. */
export const modelUnavailable = signal<string | null>(null);

async function afterLoad(ckpt: LoadedCheckpoint): Promise<void> {
  checkpoint.set(ckpt);
  model.set(null);
  modelUnavailable.set(null);
  try {
    const dev = gpuState.peek().backend ? "webgpu" : "cpu";
    model.set(await instantiateModel(ckpt, dev));
  } catch (err: unknown) {
    if (isEngineNotWired(err)) modelUnavailable.set(err.message);
    else throw err;
  }
}

export async function loadShipped(): Promise<void> {
  loading.set("Fetching shipped checkpoint…");
  loadError.set(null);
  try {
    await afterLoad(await loadCheckpointFromUrl(SHIPPED_CHECKPOINT.url));
  } catch (err: unknown) {
    loadError.set(describeError(err));
  } finally {
    loading.set(null);
  }
}

export async function loadFromFiles(files: FileList | File[]): Promise<void> {
  const list = Array.from(files);
  const st = list.find((f) => f.name.endsWith(".safetensors"));
  const meta = list.find((f) => f.name.endsWith(".json")) ?? null;
  if (!st) {
    loadError.set("Pick a .safetensors file (plus its .json sidecar).");
    return;
  }
  loading.set(`Reading ${st.name}…`);
  loadError.set(null);
  try {
    await afterLoad(await loadCheckpointFromFiles(st, meta));
  } catch (err: unknown) {
    loadError.set(describeError(err));
  } finally {
    loading.set(null);
  }
}

function describeError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  if (/HTTP 404/.test(msg)) return "The shipped checkpoint is not published yet (scripts/train-tiny.ts writes web/public/checkpoints/). Load your own .safetensors instead.";
  return msg;
}
