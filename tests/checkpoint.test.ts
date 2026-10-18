import { describe, expect, it } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  makeCheckpointMeta,
  parseSafetensors,
  serializeSafetensors,
  sidecarUrl,
  type TensorRecord,
} from "../src/models/checkpoint";
import { readCheckpointNode, writeCheckpointNode } from "../src/models/checkpoint-node";

describe("safetensors", () => {
  it("writes the exact header bytes of the reference format", () => {
    const records: TensorRecord[] = [
      { name: "a", dtype: "F32", shape: [2, 2], data: Float32Array.from([1, 2, 3, 4]) },
      { name: "b", dtype: "I32", shape: [3], data: Int32Array.from([7, -8, 9]) },
    ];
    const bytes = serializeSafetensors(records, { format: "pt" });

    // Hand-built reference: JSON header, space-padded to 8 bytes, u64 LE length prefix, raw tensors.
    const headerJson = JSON.stringify({
      __metadata__: { format: "pt" },
      a: { dtype: "F32", shape: [2, 2], data_offsets: [0, 16] },
      b: { dtype: "I32", shape: [3], data_offsets: [16, 28] },
    });
    const enc = new TextEncoder().encode(headerJson);
    const padded = Math.ceil(enc.length / 8) * 8;
    const expected = new Uint8Array(8 + padded + 28);
    new DataView(expected.buffer).setBigUint64(0, BigInt(padded), true);
    expected.set(enc, 8);
    expected.fill(0x20, 8 + enc.length, 8 + padded);
    expected.set(new Uint8Array(Float32Array.from([1, 2, 3, 4]).buffer), 8 + padded);
    expected.set(new Uint8Array(Int32Array.from([7, -8, 9]).buffer), 8 + padded + 16);

    expect(bytes.length).toBe(expected.length);
    expect(Array.from(bytes)).toEqual(Array.from(expected));
    expect(bytes.length % 4).toBe(0);
    // header length prefix is a multiple of 8 and the header is pure ASCII JSON + spaces
    expect(Number(new DataView(bytes.buffer).getBigUint64(0, true)) % 8).toBe(0);
    expect(new TextDecoder().decode(bytes.subarray(8, 8 + padded)).trimEnd()).toBe(headerJson);
  });

  it("round-trips records and metadata", () => {
    const w = new Float32Array(1000);
    for (let i = 0; i < w.length; i++) w[i] = Math.sin(i) * 1e3 + (i % 7) * 1e-7;
    const records: TensorRecord[] = [
      { name: "transformer.wte.weight", dtype: "F32", shape: [100, 10], data: w },
      { name: "ids", dtype: "I32", shape: [2, 3], data: Int32Array.from([1, 2, 3, -4, 2 ** 31 - 1, -(2 ** 31)]) },
      { name: "scalar", dtype: "F32", shape: [], data: Float32Array.from([3.5]) },
    ];
    const bytes = serializeSafetensors(records, { format: "kiln", step: "12" });
    // Parse from an unaligned view to exercise the copy path.
    const shifted = new Uint8Array(bytes.length + 3);
    shifted.set(bytes, 3);
    const parsed = parseSafetensors(shifted.subarray(3));
    expect(parsed.metadata).toEqual({ format: "kiln", step: "12" });
    expect(parsed.records.map((r) => r.name)).toEqual(records.map((r) => r.name));
    for (let i = 0; i < records.length; i++) {
      const a = records[i] as TensorRecord;
      const b = parsed.records[i] as TensorRecord;
      expect(b.dtype).toBe(a.dtype);
      expect(b.shape).toEqual(a.shape);
      expect(Array.from(b.data)).toEqual(Array.from(a.data));
    }
  });

  it("rejects malformed inputs", () => {
    expect(() => parseSafetensors(new Uint8Array(4))).toThrow();
    expect(() =>
      serializeSafetensors([{ name: "x", dtype: "F32", shape: [2], data: Float32Array.from([1, 2, 3]) }]),
    ).toThrow();
    expect(() =>
      serializeSafetensors([
        { name: "x", dtype: "F32", shape: [1], data: Float32Array.from([1]) },
        { name: "x", dtype: "F32", shape: [1], data: Float32Array.from([2]) },
      ]),
    ).toThrow();
    const ok = serializeSafetensors([{ name: "x", dtype: "F32", shape: [1], data: Float32Array.from([1]) }]);
    const bad = ok.slice();
    bad[0] = 0xff; // absurd header length
    expect(() => parseSafetensors(bad)).toThrow();
  });

  it("handles an empty file", () => {
    const bytes = serializeSafetensors([]);
    const parsed = parseSafetensors(bytes);
    expect(parsed.records).toEqual([]);
    expect(parsed.metadata).toEqual({});
  });

  it("derives sidecar paths", () => {
    expect(sidecarUrl("/kiln/checkpoints/shakespeare-tiny.safetensors")).toBe("/kiln/checkpoints/shakespeare-tiny.json");
    expect(sidecarUrl("https://x/y.safetensors?v=2")).toBe("https://x/y.json?v=2");
  });
});

describe("checkpoint-node", () => {
  it("writes and reads a checkpoint with its sidecar", async () => {
    const dir = await mkdtemp(join(tmpdir(), "kiln-ckpt-"));
    try {
      const path = join(dir, "nested", "model.safetensors");
      const records: TensorRecord[] = [{ name: "w", dtype: "F32", shape: [2], data: Float32Array.from([0.5, -0.25]) }];
      const meta = makeCheckpointMeta({ nLayer: 2 }, { type: "char", chars: ["a", "b"] }, [{ step: 1, loss: 2.5 }]);
      const info = await writeCheckpointNode(path, records, meta, { format: "kiln" });
      expect(info.metaPath).toBe(join(dir, "nested", "model.json"));
      const back = await readCheckpointNode(path);
      expect(back.metadata).toEqual({ format: "kiln" });
      expect(Array.from(back.records[0]!.data)).toEqual([0.5, -0.25]);
      expect(back.meta?.format).toBe("kiln.v1");
      expect(back.meta?.config).toEqual({ nLayer: 2 });
      expect(back.meta?.trainingLog).toEqual([{ step: 1, loss: 2.5 }]);
      expect(typeof back.meta?.createdAt).toBe("string");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
