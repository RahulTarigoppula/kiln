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

export interface ShippedCheckpoint {
  id: string;
  label: string;
  url: string;
  blurb: string;
}

const base = import.meta.env.BASE_URL;

/** Checkpoints bundled with the site (web/public/checkpoints). The first entry is the default. */
export const SHIPPED_CHECKPOINTS: ShippedCheckpoint[] = [
  {
    id: "shakespeare-micro",
    label: "shakespeare-micro",
    url: `${base}checkpoints/shakespeare-micro.safetensors`,
    blurb: "char-level · 4 layers · 4 heads · d128 · T128 · trained on WebGPU",
  },
  {
    id: "shakespeare-tiny",
    label: "shakespeare-tiny",
    url: `${base}checkpoints/shakespeare-tiny.safetensors`,
    blurb: "char-level · 2 layers · 2 heads · d64 · T64 · trained on CPU",
  },
];

export const DEFAULT_CHECKPOINT = SHIPPED_CHECKPOINTS[0] as ShippedCheckpoint;

/** @deprecated use {@link DEFAULT_CHECKPOINT} / {@link SHIPPED_CHECKPOINTS}. */
export const SHIPPED_CHECKPOINT = DEFAULT_CHECKPOINT;

export const checkpoint = signal<LoadedCheckpoint | null>(null);
export const model = signal<ModelHandle | null>(null);
export const loading = signal<string | null>(null);
export const loadError = signal<string | null>(null);
/** Set when the checkpoint loaded but the model could not be instantiated. */
export const modelUnavailable = signal<string | null>(null);

async function afterLoad(ckpt: LoadedCheckpoint): Promise<void> {
  // Release the previous model's device buffers before building the next one.
  const prev = model.peek();
  checkpoint.set(ckpt);
  model.set(null);
  modelUnavailable.set(null);
  if (prev) for (const p of prev.impl.parameters()) p.free();
  try {
    const dev = gpuState.peek().backend ? "webgpu" : "cpu";
    model.set(await instantiateModel(ckpt, dev));
  } catch (err: unknown) {
    if (isEngineNotWired(err)) modelUnavailable.set(err.message);
    else throw err;
  }
}

export async function loadShipped(which: ShippedCheckpoint = DEFAULT_CHECKPOINT): Promise<void> {
  loading.set(`Fetching ${which.label}...`);
  loadError.set(null);
  try {
    await afterLoad(await loadCheckpointFromUrl(which.url));
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
  loading.set(`Reading ${st.name}...`);
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
  if (/HTTP 404/.test(msg)) return "That checkpoint is not published in this build (web/public/checkpoints/). Load your own .safetensors instead.";
  return msg;
}
