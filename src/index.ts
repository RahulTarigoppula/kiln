/**
 * Kiln public API: tensors, autograd ops, backends, nn layers, optimisers, data utilities, models and the trainer.
 * Importing this module registers the CPU backend; the WebGPU backend registers itself when initialised.
 * Browser-safe: Node-only helpers live in `models/checkpoint-node.ts` and `models/io-node.ts`.
 */
export * from "./tensor/index";
export * from "./autograd/graph";
export * as ops from "./autograd/ops";
export type { Backend, Buf, BinaryOp, UnaryOp, ReduceOp } from "./backend/types";
export { cpuBackend, CpuBackend } from "./backend/cpu/index";
export * as nn from "./nn/index";
export * as optim from "./optim/index";
export * from "./data/index";
export * from "./models/index";
export * from "./train/index";
