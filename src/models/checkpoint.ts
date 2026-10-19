/**
 * safetensors-compatible checkpoint format (browser-safe; no Node imports).
 *
 * Layout: `u64 LE header length` + JSON header (padded with spaces to a multiple of 8 bytes) + raw tensor
 * bytes. The header maps each tensor name to `{dtype, shape, data_offsets: [start, end]}` (offsets relative
 * to the data section) and may hold `__metadata__` (string -> string).
 *
 * Kiln writes a sidecar `.json` next to the `.safetensors` file ({@link KilnCheckpointMeta}) with the model
 * config, tokenizer and training log.
 */

/** safetensors dtypes Kiln reads/writes. */
export type SafeDType = "F32" | "I32";

/** One tensor to store. */
export interface TensorRecord {
  name: string;
  dtype: SafeDType;
  shape: number[];
  data: Float32Array | Int32Array;
}

/** Parsed safetensors file. */
export interface ParsedSafetensors {
  records: TensorRecord[];
  metadata: Record<string, string>;
}

/** One training-log row stored in the sidecar. */
export interface TrainingLogEntry {
  step: number;
  loss: number;
  valLoss?: number;
  lr?: number;
  tokensPerSec?: number;
  msPerStep?: number;
}

/** Sidecar JSON schema. */
export interface KilnCheckpointMeta {
  format: "kiln.v1";
  /** Model configuration (e.g. GPT config). */
  config: Record<string, unknown>;
  /** Serialised tokenizer (see `src/data`). */
  tokenizer: unknown;
  trainingLog: TrainingLogEntry[];
  /** ISO-8601 timestamp. */
  createdAt: string;
  /** Optional free-form extras (optimizer state, step count, ...). */
  extra?: Record<string, unknown>;
}

const BYTES_PER_ELEMENT: Record<SafeDType, number> = { F32: 4, I32: 4 };
const HEADER_LIMIT = 100 * 1024 * 1024;

function utf8(s: string): Uint8Array {
  return new TextEncoder().encode(s);
}

/** Serialise records (+ optional string metadata) to safetensors bytes. */
export function serializeSafetensors(records: readonly TensorRecord[], metadata?: Record<string, string>): Uint8Array {
  const header: Record<string, unknown> = {};
  if (metadata && Object.keys(metadata).length) header.__metadata__ = metadata;
  let offset = 0;
  const seen = new Set<string>();
  for (const r of records) {
    if (seen.has(r.name)) throw new Error(`serializeSafetensors: duplicate tensor name "${r.name}"`);
    seen.add(r.name);
    const numel = r.shape.reduce((a, b) => a * b, 1);
    if (numel !== r.data.length) throw new Error(`serializeSafetensors: "${r.name}" shape ${JSON.stringify(r.shape)} has ${numel} elements but data has ${r.data.length}`);
    const bytes = numel * BYTES_PER_ELEMENT[r.dtype];
    header[r.name] = { dtype: r.dtype, shape: r.shape, data_offsets: [offset, offset + bytes] };
    offset += bytes;
  }
  const json = utf8(JSON.stringify(header));
  const padded = Math.ceil(json.length / 8) * 8;
  const out = new Uint8Array(8 + padded + offset);
  const view = new DataView(out.buffer);
  view.setUint32(0, padded >>> 0, true);
  view.setUint32(4, Math.floor(padded / 2 ** 32), true);
  out.set(json, 8);
  out.fill(0x20, 8 + json.length, 8 + padded);
  let pos = 8 + padded;
  for (const r of records) {
    const src = new Uint8Array(r.data.buffer, r.data.byteOffset, r.data.byteLength);
    out.set(src, pos);
    pos += src.byteLength;
  }
  return out;
}

/** Parse safetensors bytes. Tensor data is copied into fresh, aligned typed arrays. */
export function parseSafetensors(input: Uint8Array | ArrayBuffer): ParsedSafetensors {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes.byteLength < 8) throw new Error("parseSafetensors: file too small");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const lo = view.getUint32(0, true);
  const hi = view.getUint32(4, true);
  const headerLen = hi * 2 ** 32 + lo;
  if (headerLen > HEADER_LIMIT || 8 + headerLen > bytes.byteLength) throw new Error("parseSafetensors: invalid header length");
  const headerText = new TextDecoder("utf-8").decode(bytes.subarray(8, 8 + headerLen));
  let header: unknown;
  try {
    header = JSON.parse(headerText);
  } catch (err: unknown) {
    throw new Error(`parseSafetensors: header is not valid JSON (${err instanceof Error ? err.message : String(err)}); is this a safetensors file?`);
  }
  if (typeof header !== "object" || header === null || Array.isArray(header)) throw new Error("parseSafetensors: header must be a JSON object");
  const dataStart = 8 + headerLen;
  const dataLen = bytes.byteLength - dataStart;
  const records: TensorRecord[] = [];
  let metadata: Record<string, string> = {};
  for (const [name, entry] of Object.entries(header as Record<string, unknown>)) {
    if (name === "__metadata__") {
      if (entry && typeof entry === "object") {
        for (const [k, v] of Object.entries(entry as Record<string, unknown>)) metadata[k] = String(v);
      }
      continue;
    }
    const e = entry as { dtype?: unknown; shape?: unknown; data_offsets?: unknown };
    if (!e || typeof e !== "object" || typeof e.dtype !== "string" || !Array.isArray(e.shape) || !Array.isArray(e.data_offsets) || e.data_offsets.length !== 2) {
      throw new Error(`parseSafetensors: malformed entry "${name}"`);
    }
    if (!e.shape.every((d) => Number.isInteger(d) && (d as number) >= 0)) throw new Error(`parseSafetensors: shape of "${name}" must be non-negative integers`);
    const [s, eEnd] = e.data_offsets as [unknown, unknown];
    if (!Number.isInteger(s) || !Number.isInteger(eEnd)) throw new Error(`parseSafetensors: data_offsets of "${name}" must be integers`);
    const start = s as number;
    const end = eEnd as number;
    if (start < 0 || end < start || end > dataLen) throw new Error(`parseSafetensors: offsets out of range for "${name}"`);
    const dtype = e.dtype as SafeDType;
    if (!(dtype in BYTES_PER_ELEMENT)) throw new Error(`parseSafetensors: unsupported dtype ${e.dtype} for "${name}"`);
    const shape = e.shape as number[];
    const numel = shape.reduce((a, b) => a * b, 1);
    if (numel * BYTES_PER_ELEMENT[dtype] !== end - start) throw new Error(`parseSafetensors: size mismatch for "${name}"`);
    // Copy into a fresh, aligned ArrayBuffer. `ArrayBuffer.prototype.slice` is used on purpose: a Node `Buffer`
    // overrides `Uint8Array.prototype.slice` to return a *view* into its shared pool, so `bytes.slice(...).buffer`
    // would be the pool and offset 0 would read someone else's bytes.
    const begin = bytes.byteOffset + dataStart + start;
    const copy = bytes.buffer.slice(begin, begin + (end - start));
    const data = dtype === "F32" ? new Float32Array(copy, 0, numel) : new Int32Array(copy, 0, numel);
    records.push({ name, dtype, shape: [...shape], data });
  }
  metadata = Object.freeze({ ...metadata });
  return { records, metadata };
}

/** Build a sidecar meta object with a fresh timestamp. */
export function makeCheckpointMeta(config: Record<string, unknown>, tokenizer: unknown, trainingLog: TrainingLogEntry[] = [], extra?: Record<string, unknown>): KilnCheckpointMeta {
  const meta: KilnCheckpointMeta = { format: "kiln.v1", config, tokenizer, trainingLog, createdAt: new Date().toISOString() };
  if (extra) meta.extra = extra;
  return meta;
}

/** Runtime check that `x` looks like a {@link KilnCheckpointMeta}. */
export function isKilnCheckpointMeta(x: unknown): x is KilnCheckpointMeta {
  return typeof x === "object" && x !== null && (x as { format?: unknown }).format === "kiln.v1" && typeof (x as { config?: unknown }).config === "object";
}

/** Result of {@link fetchCheckpoint}. */
export interface FetchedCheckpoint extends ParsedSafetensors {
  /** Sidecar meta if `<url without .safetensors>.json` (or `metaUrl`) was found. */
  meta: KilnCheckpointMeta | null;
}

/** Derive the sidecar URL for a `.safetensors` URL. */
export function sidecarUrl(url: string): string {
  return url.replace(/\.safetensors(\?.*)?$/, ".json$1");
}

/** Fetch a checkpoint (and its sidecar when present) in the browser or any `fetch`-capable runtime. */
export async function fetchCheckpoint(url: string, opts: { metaUrl?: string; signal?: AbortSignal } = {}): Promise<FetchedCheckpoint> {
  const res = await fetch(url, { signal: opts.signal });
  if (!res.ok) throw new Error(`fetchCheckpoint: ${url} -> HTTP ${res.status}`);
  const parsed = parseSafetensors(new Uint8Array(await res.arrayBuffer()));
  let meta: KilnCheckpointMeta | null = null;
  const metaUrl = opts.metaUrl ?? sidecarUrl(url);
  if (metaUrl !== url) {
    try {
      const mres = await fetch(metaUrl, { signal: opts.signal });
      if (mres.ok) {
        const json: unknown = await mres.json();
        if (isKilnCheckpointMeta(json)) meta = json;
      }
    } catch {
      meta = null;
    }
  }
  return { ...parsed, meta };
}
