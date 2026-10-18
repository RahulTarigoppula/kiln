/**
 * Kernel micro-benchmarks run directly against the Backend interface (CPU vs WebGPU).
 * Timing is wall-clock around enqueue + `sync()` (timestamp queries are not surfaced by the backend).
 */
import type { Backend, Buf } from "../../../src/backend/types";
import { cpuBackend } from "../../../src/backend/cpu/index";

export type BenchGroup = "matmul" | "elementwise" | "softmax" | "layernorm";

export interface BenchRow {
  group: BenchGroup;
  kernel: string;
  size: string;
  unit: "GFLOPS" | "GB/s" | "rows/s";
  /** Work per call (FLOP, bytes or rows) — metric = work / seconds. */
  work: number;
  cpuMs: number | null;
  gpuMs: number | null;
  cpu: number | null;
  gpu: number | null;
}

export interface BenchOptions {
  gpu: Backend | null;
  includeCpu: boolean;
  /** Largest square matmul the CPU is asked to run. */
  cpuMaxMatmul: number;
  onProgress?: (label: string, frac: number) => void;
  onRow?: (row: BenchRow) => void;
  signal?: AbortSignal;
}

export const MATMUL_SIZES = [128, 256, 512, 1024, 2048];
const EW_N = 1 << 22; // 4M floats
const ROWS = 8192;
const COLS = 1024;

function yieldToUi(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}

async function timeIt(backend: Backend, fn: () => void, targetMs = 220, maxIters = 40): Promise<number> {
  fn();
  await backend.sync(); // warm-up (pipeline compilation, pool growth)
  let t0 = performance.now();
  fn();
  await backend.sync();
  const once = performance.now() - t0;
  const iters = Math.max(1, Math.min(maxIters, Math.round(targetMs / Math.max(once, 0.05))));
  t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  await backend.sync();
  return (performance.now() - t0) / iters;
}

function upload(backend: Backend, n: number, seed: number): Buf {
  const buf = backend.alloc(n * 4);
  const data = new Float32Array(n);
  let s = seed >>> 0 || 1;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    data[i] = (s / 4294967296) * 2 - 1;
  }
  backend.write(buf, data);
  return buf;
}

function metric(work: number, ms: number | null, unit: BenchRow["unit"]): number | null {
  if (ms === null || ms <= 0) return null;
  const perSec = work / (ms / 1000);
  return unit === "GFLOPS" ? perSec / 1e9 : unit === "GB/s" ? perSec / 1e9 : perSec;
}

async function matmulCase(backend: Backend, n: number): Promise<number> {
  const a = upload(backend, n * n, 1);
  const b = upload(backend, n * n, 2);
  const out = backend.alloc(n * n * 4);
  try {
    return await timeIt(backend, () => backend.matmul(a, b, out, 1, n, n, n, false, false, 0, 0), n >= 1024 ? 300 : 200, n >= 1024 ? 12 : 40);
  } finally {
    backend.free(a);
    backend.free(b);
    backend.free(out);
  }
}

async function elementwiseCase(backend: Backend, kind: "add" | "gelu"): Promise<number> {
  const a = upload(backend, EW_N, 3);
  const b = kind === "add" ? upload(backend, EW_N, 4) : null;
  const out = backend.alloc(EW_N * 4);
  try {
    if (kind === "add" && b) return await timeIt(backend, () => backend.binary("add", a, b, out, [EW_N], [1], [1]));
    return await timeIt(backend, () => backend.unary("gelu", a, EW_N, out));
  } finally {
    backend.free(a);
    if (b) backend.free(b);
    backend.free(out);
  }
}

async function softmaxCase(backend: Backend): Promise<number> {
  const a = upload(backend, ROWS * COLS, 5);
  const out = backend.alloc(ROWS * COLS * 4);
  try {
    return await timeIt(backend, () => backend.softmax(a, ROWS, COLS, out, false));
  } finally {
    backend.free(a);
    backend.free(out);
  }
}

async function layerNormCase(backend: Backend): Promise<number> {
  const x = upload(backend, ROWS * COLS, 6);
  const gamma = backend.alloc(COLS * 4);
  const beta = backend.alloc(COLS * 4);
  backend.fill(gamma, COLS, 1);
  backend.fill(beta, COLS, 0);
  const out = backend.alloc(ROWS * COLS * 4);
  const meanRstd = backend.alloc(ROWS * 2 * 4);
  try {
    return await timeIt(backend, () => backend.layerNorm(x, gamma, beta, ROWS, COLS, 1e-5, out, meanRstd));
  } finally {
    for (const b of [x, gamma, beta, out, meanRstd]) backend.free(b);
  }
}

interface Case {
  group: BenchGroup;
  kernel: string;
  size: string;
  unit: BenchRow["unit"];
  work: number;
  cpu: boolean;
  gpu: boolean;
  run(backend: Backend): Promise<number>;
}

export function buildCases(opts: BenchOptions): Case[] {
  const cases: Case[] = [];
  for (const n of MATMUL_SIZES) {
    cases.push({
      group: "matmul",
      kernel: "matmul f32",
      size: `${n}×${n}×${n}`,
      unit: "GFLOPS",
      work: 2 * n * n * n,
      cpu: opts.includeCpu && n <= opts.cpuMaxMatmul,
      gpu: opts.gpu !== null,
      run: (b) => matmulCase(b, n),
    });
  }
  cases.push({ group: "elementwise", kernel: "add (a+b)", size: `${(EW_N / 1e6).toFixed(0)}M f32`, unit: "GB/s", work: EW_N * 4 * 3, cpu: opts.includeCpu, gpu: opts.gpu !== null, run: (b) => elementwiseCase(b, "add") });
  cases.push({ group: "elementwise", kernel: "gelu", size: `${(EW_N / 1e6).toFixed(0)}M f32`, unit: "GB/s", work: EW_N * 4 * 2, cpu: opts.includeCpu, gpu: opts.gpu !== null, run: (b) => elementwiseCase(b, "gelu") });
  cases.push({ group: "softmax", kernel: "softmax rows", size: `${ROWS}×${COLS}`, unit: "rows/s", work: ROWS, cpu: opts.includeCpu, gpu: opts.gpu !== null, run: softmaxCase });
  cases.push({ group: "layernorm", kernel: "layernorm rows", size: `${ROWS}×${COLS}`, unit: "rows/s", work: ROWS, cpu: opts.includeCpu, gpu: opts.gpu !== null, run: layerNormCase });
  return cases;
}

export async function runBench(opts: BenchOptions): Promise<BenchRow[]> {
  const cases = buildCases(opts).filter((c) => c.cpu || c.gpu);
  const rows: BenchRow[] = [];
  const total = cases.reduce((n, c) => n + (c.cpu ? 1 : 0) + (c.gpu ? 1 : 0), 0);
  let done = 0;
  for (const c of cases) {
    if (opts.signal?.aborted) break;
    const row: BenchRow = { group: c.group, kernel: c.kernel, size: c.size, unit: c.unit, work: c.work, cpuMs: null, gpuMs: null, cpu: null, gpu: null };
    if (c.gpu && opts.gpu) {
      opts.onProgress?.(`WebGPU · ${c.kernel} ${c.size}`, done / total);
      await yieldToUi();
      row.gpuMs = await c.run(opts.gpu);
      row.gpu = metric(c.work, row.gpuMs, c.unit);
      done++;
    }
    if (c.cpu) {
      opts.onProgress?.(`CPU · ${c.kernel} ${c.size}`, done / total);
      await yieldToUi();
      row.cpuMs = await c.run(cpuBackend);
      row.cpu = metric(c.work, row.cpuMs, c.unit);
      done++;
    }
    rows.push(row);
    opts.onRow?.(row);
  }
  opts.onProgress?.("done", 1);
  return rows;
}

export function formatMetric(v: number | null, unit: BenchRow["unit"]): string {
  if (v === null || !Number.isFinite(v)) return "—";
  if (unit === "rows/s") return v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}k` : v.toFixed(0);
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}

export function toMarkdown(rows: BenchRow[], adapter: string | null): string {
  const lines = [
    `## Kiln kernel bench${adapter ? ` — ${adapter}` : ""}`,
    "",
    `Measured in-browser (${navigator.userAgent.includes("Chrome") ? "Chromium" : "browser"}), wall-clock around enqueue + sync. ${new Date().toISOString().slice(0, 10)}.`,
    "",
    "| Kernel | Size | Unit | CPU | WebGPU | Speedup |",
    "|---|---|---|---:|---:|---:|",
  ];
  for (const r of rows) {
    const speed = r.cpu !== null && r.gpu !== null && r.cpu > 0 ? `${(r.gpu / r.cpu).toFixed(1)}×` : "—";
    lines.push(`| ${r.kernel} | ${r.size} | ${r.unit} | ${formatMetric(r.cpu, r.unit)} | ${formatMetric(r.gpu, r.unit)} | ${speed} |`);
  }
  return lines.join("\n");
}
