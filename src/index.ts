/**
 * Kiln public API: tensors, autograd ops and backends.
 * Importing this module registers the CPU backend; the WebGPU backend registers itself when initialised.
 */
export * from "./tensor/index";
export * from "./autograd/graph";
export * as ops from "./autograd/ops";
export type { Backend, Buf, BinaryOp, UnaryOp, ReduceOp } from "./backend/types";
export { cpuBackend, CpuBackend } from "./backend/cpu/index";
