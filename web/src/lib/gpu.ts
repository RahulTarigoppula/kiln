/**
 * WebGPU initialisation for the web app: one shared backend, adapter info for the device badge,
 * availability flags and a "busy" counter that makes the badge pulse while a job runs.
 */
import { initWebGpu } from "../../../src/backend/webgpu/register";
import { isWebGpuAvailable, type WebGpuBackend, type WebGpuAdapterInfo } from "../../../src/backend/webgpu/index";
import { cpuBackend } from "../../../src/backend/cpu/index";
import type { Backend } from "../../../src/backend/types";
import { signal } from "./signals";

export type GpuStatus = "idle" | "initializing" | "ready" | "unavailable";

export interface GpuState {
  status: GpuStatus;
  backend: WebGpuBackend | null;
  info: WebGpuAdapterInfo | null;
  /** navigator.gpu exists (adapter may still be missing). */
  apiPresent: boolean;
  error: string | null;
}

export const gpuState = signal<GpuState>({ status: "idle", backend: null, info: null, apiPresent: isWebGpuAvailable(), error: null });

/** Number of running jobs (bench, parity, training...). The badge pulses when > 0. */
export const busyCount = signal(0);

export function markBusy<T>(p: Promise<T>): Promise<T> {
  busyCount.update((n) => n + 1);
  return p.finally(() => busyCount.update((n) => Math.max(0, n - 1)));
}

let initPromise: Promise<WebGpuBackend | null> | null = null;

/** Initialise WebGPU once; resolves to the backend or null (CPU fallback). Never throws. */
export function ensureGpu(): Promise<WebGpuBackend | null> {
  if (initPromise) return initPromise;
  if (!isWebGpuAvailable()) {
    gpuState.set({ status: "unavailable", backend: null, info: null, apiPresent: false, error: "navigator.gpu is missing" });
    initPromise = Promise.resolve(null);
    return initPromise;
  }
  gpuState.update((s) => ({ ...s, status: "initializing" }));
  initPromise = initWebGpu({
    onError: (msg) => console.error("[kiln/webgpu]", msg),
  }).then((backend) => {
    if (!backend) {
      gpuState.set({ status: "unavailable", backend: null, info: null, apiPresent: true, error: "No compatible GPU adapter / device creation failed" });
      return null;
    }
    backend.onDeviceLost = (info) => {
      console.warn("[kiln] GPU device lost:", info.message);
      gpuState.set({ status: "unavailable", backend: null, info: backend.info, apiPresent: true, error: `Device lost: ${info.message}` });
    };
    gpuState.set({ status: "ready", backend, info: backend.info, apiPresent: true, error: null });
    return backend;
  });
  return initPromise;
}

/** The best available backend right now (GPU when ready, CPU otherwise). */
export function activeBackend(): Backend {
  return gpuState.peek().backend ?? cpuBackend;
}

/** Human label for the adapter, e.g. "NVIDIA GeForce RTX 3060" or "Apple M2". */
export function adapterLabel(info: WebGpuAdapterInfo | null): string {
  if (!info) return "WebGPU";
  const desc = info.description.trim();
  if (desc) return cleanDescription(desc);
  const parts = [info.vendor, info.architecture].filter((p) => p && p !== "unknown");
  return parts.length ? parts.map(capitalize).join(" · ") : "WebGPU adapter";
}

function cleanDescription(d: string): string {
  // Chrome often reports e.g. "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 (0x00002503) Direct3D11 vs_5_0 ps_5_0, D3D11)"
  const angle = /ANGLE \(([^,]+), ([^(]+?)(?: \(0x[0-9a-fA-F]+\))?[^,]*,/.exec(d);
  if (angle?.[2]) return angle[2].trim();
  return d.replace(/\s*\(0x[0-9a-fA-F]+\)/g, "").trim();
}

function capitalize(s: string): string {
  return s.length ? s[0]!.toUpperCase() + s.slice(1) : s;
}

export interface BrowserGuide {
  name: string;
  version: string;
  note: string;
}

/** Browsers that ship WebGPU, for the unavailable state. */
export const BROWSER_GUIDANCE: BrowserGuide[] = [
  { name: "Chrome / Edge", version: "113+", note: "Windows, macOS, ChromeOS; Linux behind chrome://flags" },
  { name: "Safari", version: "26+", note: "macOS Tahoe, iOS 26, iPadOS 26" },
  { name: "Firefox", version: "141+", note: "Windows; other platforms behind dom.webgpu.enabled" },
];
