/**
 * Device registry. The CPU backend is always registered; the WebGPU backend registers itself at runtime
 * via `registerBackend("webgpu", backend)` once a GPU device has been acquired.
 */
import type { Backend, Device } from "../backend/types";
import { cpuBackend } from "../backend/cpu/index";

const registry: Partial<Record<Device, Backend>> = { cpu: cpuBackend };
let defaultDevice: Device = "cpu";

/** Register (or replace) the backend for `device`. */
export function registerBackend(device: Device, backend: Backend): void {
  registry[device] = backend;
}

/** Remove a registered backend (used when a GPU device is lost). */
export function unregisterBackend(device: Device): void {
  if (device === "cpu") throw new Error("The CPU backend cannot be unregistered");
  delete registry[device];
  if (defaultDevice === device) defaultDevice = "cpu";
}

/** True when a backend exists for `device`. */
export function hasBackend(device: Device): boolean {
  return registry[device] !== undefined;
}

/** Backend for `device`; throws when none is registered. */
export function getBackend(device: Device): Backend {
  const b = registry[device];
  if (!b) throw new Error(`No backend registered for device "${device}"`);
  return b;
}

/** Device used when a creation function does not specify one. */
export function getDefaultDevice(): Device {
  return defaultDevice;
}

/** Set the default device (must be registered). */
export function setDefaultDevice(device: Device): void {
  getBackend(device);
  defaultDevice = device;
}
