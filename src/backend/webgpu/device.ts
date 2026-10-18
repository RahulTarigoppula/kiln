/**
 * WebGPU adapter/device acquisition for the Kiln WebGPU backend.
 *
 * Requests raised storage/buffer limits and the optional `timestamp-query` feature, and surfaces
 * adapter info so the UI can show which GPU is running the kernels.
 */

/** Options accepted by {@link requestKilnDevice}. */
export interface WebGpuDeviceOptions {
  /** Passed straight through to `requestAdapter`. */
  powerPreference?: GPUPowerPreference;
  /** Extra limits to request (merged over the raised defaults; clamped to what the adapter supports). */
  requiredLimits?: Record<string, number>;
}

/** Adapter description surfaced as `backend.info`. */
export interface WebGpuAdapterInfo {
  vendor: string;
  architecture: string;
  description: string;
  /** Whether the device was created with the `timestamp-query` feature. */
  timestampQuery: boolean;
}

/** Result of {@link requestKilnDevice}. */
export interface KilnGpuDevice {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: WebGpuAdapterInfo;
  /** Resolves (never rejects) when the device is lost; carries the reason/message. */
  lost: Promise<GPUDeviceLostInfo>;
}

/** True when `navigator.gpu` exists in the current realm (main thread or worker). */
export function isWebGpuAvailable(): boolean {
  const nav = (globalThis as { navigator?: { gpu?: unknown } }).navigator;
  return typeof nav === "object" && nav !== null && typeof nav.gpu === "object" && nav.gpu !== null;
}

/** Limits we try to raise, in priority order; each is clamped to the adapter's maximum. */
const RAISED_LIMITS: readonly (keyof GPUSupportedLimits)[] = [
  "maxStorageBufferBindingSize",
  "maxBufferSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupStorageSize",
  "maxComputeWorkgroupSizeX",
  "maxStorageBuffersPerShaderStage",
];

function readAdapterInfo(adapter: GPUAdapter): { vendor: string; architecture: string; description: string } {
  const a = adapter as unknown as { info?: { vendor?: string; architecture?: string; description?: string } };
  const inf = a.info;
  if (inf && typeof inf === "object") {
    return {
      vendor: inf.vendor ?? "unknown",
      architecture: inf.architecture ?? "unknown",
      description: inf.description ?? "",
    };
  }
  return { vendor: "unknown", architecture: "unknown", description: "" };
}

/**
 * Request an adapter and device suitable for Kiln's kernels.
 * Throws if WebGPU is unavailable or no adapter can be obtained.
 */
export async function requestKilnDevice(opts: WebGpuDeviceOptions = {}): Promise<KilnGpuDevice> {
  if (!isWebGpuAvailable()) throw new Error("WebGPU is not available in this environment (navigator.gpu missing).");
  const gpu = (globalThis as unknown as { navigator: { gpu: GPU } }).navigator.gpu;
  const adapterOpts: GPURequestAdapterOptions = {};
  if (opts.powerPreference) adapterOpts.powerPreference = opts.powerPreference;
  const adapter = await gpu.requestAdapter(adapterOpts);
  if (!adapter) throw new Error("WebGPU: requestAdapter() returned null (no compatible GPU adapter).");

  const requiredLimits: Record<string, number> = {};
  const sup = adapter.limits as unknown as Record<string, number | undefined>;
  for (const key of RAISED_LIMITS) {
    const max = sup[key as string];
    if (typeof max === "number" && Number.isFinite(max)) requiredLimits[key as string] = max;
  }
  if (opts.requiredLimits) {
    for (const [k, v] of Object.entries(opts.requiredLimits)) {
      const max = sup[k];
      requiredLimits[k] = typeof max === "number" ? Math.min(v, max) : v;
    }
  }

  const requiredFeatures: GPUFeatureName[] = [];
  const hasTimestamp = adapter.features.has("timestamp-query");
  if (hasTimestamp) requiredFeatures.push("timestamp-query");

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({ requiredLimits, requiredFeatures, label: "kiln" });
  } catch {
    // Some drivers reject a subset of raised limits; fall back to defaults so the backend still works.
    device = await adapter.requestDevice({ requiredFeatures, label: "kiln-fallback" });
  }

  const lost = device.lost.then((l) => l, (e: unknown) => ({ reason: "unknown", message: String(e) }) as GPUDeviceLostInfo);
  const base = readAdapterInfo(adapter);
  const info: WebGpuAdapterInfo = { ...base, timestampQuery: device.features.has("timestamp-query") };
  return { adapter, device, info, lost };
}
