/**
 * Message protocol between the Studio page and the training worker (or its main-thread fallback).
 */
import type { Device } from "../../../src/backend/types";
import type {
  EvalMetrics,
  GenerateOptions,
  GPTConfig,
  KilnCheckpointMeta,
  SampleEvent,
  StepMetrics,
  TokenEvent,
  TokenizerJSON,
  TokenizerSpec,
  TrainHyper,
} from "../lib/engine";

export type SessionState = "idle" | "preparing" | "running" | "paused" | "stopped" | "done" | "error";

export interface StartJob {
  text: string;
  tokenizer: TokenizerSpec;
  /** Config without vocabSize (filled from the tokenizer). */
  model: Omit<GPTConfig, "vocabSize">;
  hyper: TrainHyper;
  preferGpu: boolean;
}

export type TrainRequest =
  | { type: "init"; preferGpu: boolean }
  | { type: "start"; job: StartJob }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "stop" }
  | { type: "export" }
  | { type: "generate"; id: number; prompt: string; opts: GenerateOptions }
  | { type: "cancelGenerate"; id: number };

export type TrainEvent =
  | { type: "ready"; device: Device; adapter: string | null; inWorker: boolean }
  | { type: "status"; phase: "tokenizer" | "encode" | "model" | "training" | "idle"; message: string; progress?: number }
  | { type: "dataset"; tokens: number; vocabSize: number; trainTokens: number; valTokens: number; tokenizer: TokenizerJSON; paramCount: number }
  | { type: "state"; state: SessionState }
  | { type: "step"; metrics: StepMetrics; memBytes: number | null; etaMs: number | null }
  | { type: "eval"; metrics: EvalMetrics }
  | { type: "sample"; sample: SampleEvent }
  | { type: "token"; id: number; token: TokenEvent }
  | { type: "generated"; id: number }
  | { type: "checkpoint"; safetensors: ArrayBuffer; meta: KilnCheckpointMeta }
  | { type: "error"; message: string; notWired?: boolean };

export type PostEvent = (ev: TrainEvent, transfer?: Transferable[]) => void;
