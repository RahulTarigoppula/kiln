/**
 * Node-only checkpoint helpers (kept separate so browser bundles never import `node:fs`).
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import {
  isKilnCheckpointMeta,
  parseSafetensors,
  serializeSafetensors,
  sidecarUrl,
  type KilnCheckpointMeta,
  type ParsedSafetensors,
  type TensorRecord,
} from "./checkpoint";

/** Write `records` to `path` (`.safetensors`) and, when given, `meta` to the sidecar `.json`. */
export async function writeCheckpointNode(
  path: string,
  records: readonly TensorRecord[],
  meta?: KilnCheckpointMeta,
  metadata?: Record<string, string>,
): Promise<{ path: string; metaPath: string | null; bytes: number }> {
  await mkdir(dirname(path), { recursive: true });
  const bytes = serializeSafetensors(records, metadata);
  await writeFile(path, bytes);
  let metaPath: string | null = null;
  if (meta) {
    metaPath = sidecarUrl(path);
    await writeFile(metaPath, JSON.stringify(meta, null, 2));
  }
  return { path, metaPath, bytes: bytes.byteLength };
}

/** Read a `.safetensors` file plus its sidecar (if present). */
export async function readCheckpointNode(path: string): Promise<ParsedSafetensors & { meta: KilnCheckpointMeta | null }> {
  const buf = await readFile(path);
  const parsed = parseSafetensors(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength));
  let meta: KilnCheckpointMeta | null = null;
  try {
    const json: unknown = JSON.parse(await readFile(sidecarUrl(path), "utf8"));
    if (isKilnCheckpointMeta(json)) meta = json;
  } catch {
    meta = null;
  }
  return { ...parsed, meta };
}
