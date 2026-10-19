/**
 * Training session handler - shared by `train.worker.ts` (Web Worker) and the main-thread fallback.
 * Owns the lifecycle: init backend -> build tokenizer/dataset -> create trainer -> stream events.
 */
import type { Device } from "../../../src/backend/types";
import { initWebGpu } from "../../../src/backend/webgpu/register";
import { isWebGpuAvailable, type WebGpuBackend } from "../../../src/backend/webgpu/index";
import {
  buildDataset,
  buildTokenizer,
  createTrainer,
  estimateParams,
  isEngineNotWired,
  type GenerateOptions,
  type GPTConfig,
  type Tokenizer,
  type TrainerHandle,
} from "../lib/engine";
import type { PostEvent, SessionState, StartJob, TrainRequest } from "./protocol";

export interface TrainHandler {
  handle(req: TrainRequest): void;
  dispose(): void;
}

export function createTrainHandler(post: PostEvent, inWorker: boolean): TrainHandler {
  let backend: WebGpuBackend | null = null;
  let device: Device = "cpu";
  let trainer: TrainerHandle | null = null;
  let tokenizer: Tokenizer | null = null;
  let state: SessionState = "idle";
  let maxSteps = 0;
  let startedAt = 0;
  let stepsDone = 0;
  const generations = new Map<number, AbortController>();

  const setState = (s: SessionState): void => {
    state = s;
    post({ type: "state", state: s });
  };
  /** Read the current state without TypeScript's control-flow narrowing (it is mutated by callbacks). */
  const current = (): SessionState => state;

  const fail = (err: unknown): void => {
    const message = err instanceof Error ? err.message : String(err);
    post({ type: "error", message, notWired: isEngineNotWired(err) });
    setState("error");
  };

  async function init(preferGpu: boolean): Promise<void> {
    if (preferGpu && isWebGpuAvailable()) {
      backend = await initWebGpu();
      device = backend ? "webgpu" : "cpu";
    } else {
      device = "cpu";
    }
    post({ type: "ready", device, adapter: backend ? adapterName(backend) : null, inWorker });
  }

  async function start(job: StartJob): Promise<void> {
    if (state === "running" || state === "preparing") return;
    setState("preparing");
    try {
      if (!backend && job.preferGpu && isWebGpuAvailable()) await init(true);
      const dev: Device = job.preferGpu && backend ? "webgpu" : "cpu";

      post({ type: "status", phase: "tokenizer", message: job.tokenizer.kind === "bpe" ? `Training byte-level BPE (vocab ${job.tokenizer.vocabSize})…` : "Building character vocabulary…" });
      await tick();
      let lastReport = 0;
      tokenizer = buildTokenizer(job.tokenizer, job.text, (done, total) => {
        const now = Date.now();
        if (now - lastReport > 120 || done === total) {
          lastReport = now;
          post({ type: "status", phase: "tokenizer", message: `Training BPE merges ${done}/${total}`, progress: total ? done / total : 0 });
        }
      });

      post({ type: "status", phase: "encode", message: `Encoding ${job.text.length.toLocaleString()} characters…` });
      await tick();
      const config: GPTConfig = { ...job.model, vocabSize: tokenizer.vocabSize };
      const { train, val, summary } = buildDataset(tokenizer, job.text, config.blockSize);
      const paramCount = estimateParams(config);
      post({
        type: "dataset",
        tokens: summary.tokens,
        vocabSize: summary.vocabSize,
        trainTokens: summary.trainTokens,
        valTokens: summary.valTokens,
        tokenizer: tokenizer.toJSON(),
        paramCount,
      });

      post({ type: "status", phase: "model", message: `Initialising GPT (${paramCount.toLocaleString()} params) on ${dev}…` });
      await tick();
      maxSteps = job.hyper.maxSteps;
      stepsDone = 0;
      startedAt = performance.now();
      trainer = await createTrainer(
        { config, tokenizer, train, val, hyper: job.hyper, device: dev },
        {
          onStep: (m) => {
            stepsDone = m.step;
            const elapsed = performance.now() - startedAt;
            const etaMs = m.step > 0 ? (elapsed / m.step) * Math.max(0, maxSteps - m.step) : null;
            post({ type: "step", metrics: m, memBytes: backend ? backend.stats().liveBytes : null, etaMs });
          },
          onEval: (m) => post({ type: "eval", metrics: m }),
          onSample: (s) => post({ type: "sample", sample: s }),
        },
      );
      post({ type: "status", phase: "training", message: "Training" });
      setState("running");
      await trainer.run();
      const after = current();
      if (after === "running" || after === "paused") setState(stepsDone >= maxSteps ? "done" : "stopped");
    } catch (err: unknown) {
      fail(err);
    }
  }

  async function exportCheckpoint(): Promise<void> {
    if (!trainer) {
      post({ type: "error", message: "No model to export yet." });
      return;
    }
    try {
      const { safetensors, meta } = await trainer.exportCheckpoint();
      const buf = safetensors.buffer.slice(safetensors.byteOffset, safetensors.byteOffset + safetensors.byteLength) as ArrayBuffer;
      post({ type: "checkpoint", safetensors: buf, meta }, [buf]);
    } catch (err: unknown) {
      fail(err);
    }
  }

  async function generate(id: number, prompt: string, opts: GenerateOptions): Promise<void> {
    if (!trainer || !tokenizer) {
      post({ type: "error", message: "Start training first - the in-training model is what gets sampled." });
      post({ type: "generated", id });
      return;
    }
    const ctrl = new AbortController();
    generations.set(id, ctrl);
    try {
      const ids = tokenizer.encode(prompt);
      for await (const tok of trainer.generate(ids, opts, ctrl.signal)) {
        if (ctrl.signal.aborted) break;
        post({ type: "token", id, token: tok });
      }
    } catch (err: unknown) {
      if (!ctrl.signal.aborted) fail(err);
    } finally {
      generations.delete(id);
      post({ type: "generated", id });
    }
  }

  return {
    handle(req: TrainRequest): void {
      switch (req.type) {
        case "init":
          void init(req.preferGpu);
          break;
        case "start":
          void start(req.job);
          break;
        case "pause":
          if (trainer && state === "running") {
            trainer.pause();
            setState("paused");
          }
          break;
        case "resume":
          if (trainer && state === "paused") {
            trainer.resume();
            setState("running");
          }
          break;
        case "stop":
          if (trainer && (state === "running" || state === "paused")) {
            trainer.stop();
            setState("stopped");
          } else if (state === "preparing") {
            setState("stopped");
          }
          break;
        case "export":
          void exportCheckpoint();
          break;
        case "generate":
          void generate(req.id, req.prompt, req.opts);
          break;
        case "cancelGenerate":
          generations.get(req.id)?.abort();
          break;
      }
    },
    dispose(): void {
      trainer?.stop();
      for (const c of generations.values()) c.abort();
      generations.clear();
    },
  };
}

function adapterName(b: WebGpuBackend): string {
  const d = b.info.description.trim();
  if (d) return d;
  return [b.info.vendor, b.info.architecture].filter((s) => s && s !== "unknown").join(" ") || "WebGPU";
}

function tick(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}
