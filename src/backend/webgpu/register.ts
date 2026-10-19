/**
 * Registration shim: creates the WebGPU backend (once) and hands it to the tensor layer's device registry
 * (`registerBackend` from `src/tensor/device.ts`). A custom registrar can be injected for tests.
 *
 * ```ts
 * import { initWebGpu } from "../../src/backend/webgpu/register";
 * const backend = await initWebGpu(); // null when WebGPU is unavailable
 * ```
 */

import type { Backend } from "../types";
import { registerBackend } from "../../tensor/device";
import { createWebGpuBackend, isWebGpuAvailable, type WebGpuBackend, type WebGpuBackendOptions } from "./index";

/** Shape of the tensor layer's `registerBackend` hook. */
export type RegisterBackendFn = (name: "webgpu", backend: Backend) => void;

let singleton: Promise<WebGpuBackend | null> | null = null;

/**
 * Create (once) the WebGPU backend and register it under the name `"webgpu"`.
 * Resolves to `null` - without throwing - when WebGPU is unavailable or device creation fails.
 */
export function initWebGpu(opts: WebGpuBackendOptions = {}, register: RegisterBackendFn = registerBackend): Promise<WebGpuBackend | null> {
  if (singleton) return singleton;
  singleton = (async () => {
    if (!isWebGpuAvailable()) return null;
    try {
      const backend = await createWebGpuBackend(opts);
      register("webgpu", backend);
      return backend;
    } catch (err: unknown) {
      console.warn("[kiln] WebGPU backend unavailable:", err instanceof Error ? err.message : String(err));
      return null;
    }
  })();
  return singleton;
}

/** Forget the cached backend promise (tests / device-lost recovery). */
export function resetWebGpuRegistration(): void {
  singleton = null;
}
