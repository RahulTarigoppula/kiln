/**
 * Training loop for GPT models: AdamW/any optimizer, LR schedule, gradient accumulation, global-norm clipping,
 * periodic eval/sampling/checkpoints, pause/resume/stop, and resumable state.
 *
 * Memory: each micro-step runs forward + backward inside `tidy()`, so every activation and gradient temporary
 * is freed before the optimizer step; parameters, their grads and optimizer state live outside scopes. The
 * only per-step host sync is the loss readback, batched every `lossReadbackInterval` steps (one `data()` call
 * for all pending losses and grad norms).
 */
import type { Device } from "../backend/types";
import { Tensor } from "../tensor/tensor";
import { tidy } from "../tensor/scope";
import { Random } from "../tensor/random";
import { noGrad } from "../autograd/graph";
import { stack } from "../autograd/ops";
import type { TokenDataset } from "../data/dataset";
import type { Tokenizer } from "../data/index";
import type { Optimizer } from "../optim/optimizer";
import { clipGradNorm } from "../optim/clip";
import type { LRSchedule } from "../optim/schedulers";
import type { GPT } from "../models/gpt";
import { recordsToStateDict, restoreOptimizer, saveGPT, type GPTCheckpointExtra, type SavedGPT } from "../models/io";
import { parseSafetensors, type KilnCheckpointMeta, type TrainingLogEntry } from "../models/checkpoint";

/** Per-step metrics passed to `onStep`. */
export interface StepInfo {
  /** 1-based step just completed. */
  step: number;
  /** Mean training loss of the step (averaged over accumulation micro-steps). */
  loss: number;
  lr: number;
  /** Pre-clip global gradient norm (when clipping is enabled). */
  gradNorm?: number;
  tokensPerSec: number;
  msPerStep: number;
}

/** Eval metrics passed to `onEval`. */
export interface EvalInfo {
  step: number;
  trainLoss: number;
  valLoss: number;
}

/** Callbacks. */
export interface TrainerCallbacks {
  onStep?: (info: StepInfo) => void;
  onEval?: (info: EvalInfo) => void;
  onSample?: (info: { step: number; text: string }) => void;
  onCheckpoint?: (info: { step: number } & SavedGPT) => void | Promise<void>;
}

/** Trainer configuration. */
export interface TrainerOptions {
  model: GPT;
  optimizer: Optimizer;
  train: TokenDataset;
  val?: TokenDataset;
  /** Needed for sample generation and for storing the tokenizer in checkpoints. */
  tokenizer?: Tokenizer;
  batchSize: number;
  /** Micro-batches per optimizer step (default 1). */
  gradAccumSteps?: number;
  /** Learning rate per step (default: the optimizer's current lr, constant). */
  lrSchedule?: LRSchedule;
  maxSteps: number;
  /** Evaluate every N steps (0 = never). */
  evalInterval?: number;
  /** Batches per evaluation (default 10). */
  evalIters?: number;
  /** Also evaluate before the first step (default false). */
  evalAtStart?: boolean;
  /** Global grad-norm clip (0 = off). */
  clipNorm?: number;
  /** Generate a sample every N steps (0 = never; needs `tokenizer`). */
  sampleInterval?: number;
  sampleTokens?: number;
  samplePrompt?: string;
  sampleTemperature?: number;
  sampleTopK?: number;
  /** Save a checkpoint (via `onCheckpoint`) every N steps (0 = never). */
  checkpointInterval?: number;
  device?: Device;
  /** Seed for batch sampling and generation. */
  seed?: number;
  /** Read losses back from the device every N steps (default 1). */
  lossReadbackInterval?: number;
  callbacks?: TrainerCallbacks;
}

/** Lifecycle status. */
export type TrainerStatus = "idle" | "running" | "paused" | "stopped" | "done";

/** Snapshot of the trainer. */
export interface TrainerState {
  step: number;
  status: TrainerStatus;
  lastLoss: number | null;
  lastValLoss: number | null;
  lr: number;
  tokensPerSec: number;
  msPerStep: number;
  trainingLog: TrainingLogEntry[];
}

/** Yield one macrotask so timers/UI events (pause, stop, rendering) get a turn even when every await resolves immediately. */
function yieldToEventLoop(): Promise<void> {
  if (typeof MessageChannel !== "undefined") {
    return new Promise((resolve) => {
      const ch = new MessageChannel();
      ch.port1.onmessage = () => {
        ch.port1.close();
        resolve();
      };
      ch.port2.postMessage(null);
    });
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

interface PendingStep {
  step: number;
  losses: Tensor[];
  gradNorm: Tensor | null;
  lr: number;
}

/** Drives training of a {@link GPT}. */
export class Trainer {
  readonly opts: Required<Pick<TrainerOptions, "gradAccumSteps" | "evalInterval" | "evalIters" | "clipNorm" | "sampleInterval" | "sampleTokens" | "checkpointInterval" | "lossReadbackInterval" | "device" | "seed">> & TrainerOptions;
  private step = 0;
  private status: TrainerStatus = "idle";
  private stopRequested = false;
  private pauseWaiters: (() => void)[] = [];
  private paused = false;
  private readonly rng: Random;
  private readonly log: TrainingLogEntry[] = [];
  private lastLoss: number | null = null;
  private lastValLoss: number | null = null;
  private lr = 0;
  private tokensPerSec = 0;
  private msPerStep = 0;
  private pending: PendingStep[] = [];
  private intervalStart = 0;

  constructor(options: TrainerOptions) {
    this.opts = {
      gradAccumSteps: 1, evalInterval: 0, evalIters: 10, clipNorm: 0, sampleInterval: 0, sampleTokens: 200, checkpointInterval: 0,
      lossReadbackInterval: 1, device: options.model.device, seed: 1337,
      ...options,
    };
    if (this.opts.batchSize <= 0) throw new Error("Trainer: batchSize must be positive");
    this.rng = new Random(this.opts.seed);
    this.lr = this.opts.optimizer.lr;
  }

  /** Current state (log is a copy). */
  get state(): TrainerState {
    return {
      step: this.step, status: this.status, lastLoss: this.lastLoss, lastValLoss: this.lastValLoss, lr: this.lr,
      tokensPerSec: this.tokensPerSec, msPerStep: this.msPerStep, trainingLog: this.log.map((e) => ({ ...e })),
    };
  }

  /** Suspend after the current step; `run()` keeps awaiting until `resume()`. */
  pause(): void {
    if (this.status === "running") {
      this.paused = true;
      this.status = "paused";
    }
  }

  /** Continue a paused run. */
  resume(): void {
    if (!this.paused) return;
    this.paused = false;
    if (this.status === "paused") this.status = "running";
    const waiters = this.pauseWaiters;
    this.pauseWaiters = [];
    for (const w of waiters) w();
  }

  /** Finish the current step and return from `run()`. */
  stop(): void {
    this.stopRequested = true;
    this.resume();
  }

  private async waitWhilePaused(): Promise<void> {
    while (this.paused && !this.stopRequested) {
      await new Promise<void>((resolve) => this.pauseWaiters.push(resolve));
    }
  }

  /** Upload a flat i32 batch as [B, T] on the training device. */
  private upload(ids: Int32Array, B: number, T: number): Tensor {
    const cpu = Tensor.fromArray(ids, [B, T], { dtype: "i32", device: "cpu" });
    const dev = cpu.to(this.opts.device);
    if (dev !== cpu) cpu.free();
    return dev;
  }

  /** Mean loss over `iters` random batches of `ds` (model in eval mode). One readback. */
  async evaluate(ds: TokenDataset, iters = this.opts.evalIters): Promise<number> {
    const { model, batchSize } = this.opts;
    const wasTraining = model.training;
    model.eval();
    let acc: Tensor | null = null;
    for (let i = 0; i < iters; i++) {
      const { x, y } = ds.getBatch(batchSize, this.rng);
      const xs = this.upload(x, batchSize, ds.blockSize);
      const ys = this.upload(y, batchSize, ds.blockSize);
      const prev: Tensor | null = acc;
      acc = tidy((): Tensor => noGrad((): Tensor => {
        const l = model.forward(xs, ys).loss as Tensor;
        return prev ? prev.add(l) : l;
      }));
      if (prev) prev.free();
      xs.free();
      ys.free();
    }
    model.train(wasTraining);
    if (!acc) return NaN;
    const total = await acc.item();
    acc.free();
    return total / iters;
  }

  /** Generate `sampleTokens` tokens from `samplePrompt` (needs a tokenizer). */
  async sample(): Promise<string | null> {
    const { model, tokenizer } = this.opts;
    if (!tokenizer) return null;
    const prompt = this.opts.samplePrompt ?? "\n";
    let ids = tokenizer.encode(prompt);
    if (ids.length === 0) ids = Int32Array.from([0]);
    const wasTraining = model.training;
    model.eval();
    const idx = this.upload(ids, 1, ids.length);
    const out = await model.generate(idx, this.opts.sampleTokens, {
      temperature: this.opts.sampleTemperature ?? 0.8, topK: this.opts.sampleTopK, seed: this.rng.nextSeed(),
    });
    const all = await out.data();
    out.free();
    idx.free();
    model.train(wasTraining);
    return tokenizer.decode(all.subarray(ids.length));
  }

  /** Serialise model + optimizer + log. */
  async checkpoint(): Promise<SavedGPT> {
    return saveGPT(this.opts.model, this.opts.tokenizer?.toJSON() ?? null, this.log, { optimizer: this.opts.optimizer, step: this.step });
  }

  /** Restore weights, optimizer state, step count and log from a checkpoint made by `checkpoint()`. */
  loadCheckpoint(bytes: Uint8Array | ArrayBuffer, meta: KilnCheckpointMeta): void {
    const { records } = parseSafetensors(bytes);
    const sd = recordsToStateDict(records);
    const tied = (meta.extra as GPTCheckpointExtra | undefined)?.tied ?? {};
    for (const [alias, canonical] of Object.entries(tied)) {
      const t = sd.get(canonical);
      if (t && !sd.has(alias)) sd.set(alias, t);
    }
    this.opts.model.loadStateDict(sd, true);
    for (const t of new Set(sd.values())) t.free();
    restoreOptimizer(this.opts.optimizer, this.opts.model, records, meta);
    const extra = meta.extra as GPTCheckpointExtra | undefined;
    this.step = extra?.step ?? this.opts.optimizer.stepCount;
    this.log.length = 0;
    this.log.push(...meta.trainingLog.map((e) => ({ ...e })));
    const last = this.log[this.log.length - 1];
    this.lastLoss = last?.loss ?? null;
  }

  /** Read back pending losses (one device sync), fire `onStep`, append to the log. */
  private async flush(): Promise<void> {
    if (this.pending.length === 0) return;
    const pending = this.pending;
    this.pending = [];
    const all: Tensor[] = [];
    for (const p of pending) {
      all.push(...p.losses);
      if (p.gradNorm) all.push(p.gradNorm);
    }
    const packed = tidy(() => noGrad(() => stack(all.map((t) => t.reshape(1)), 0)));
    const values = (await packed.data()) as Float32Array;
    packed.free();
    for (const t of all) t.free();
    const elapsed = performance.now() - this.intervalStart;
    this.intervalStart = performance.now();
    const tokensPerStep = this.opts.batchSize * this.opts.train.blockSize * this.opts.gradAccumSteps;
    this.msPerStep = elapsed / pending.length;
    this.tokensPerSec = this.msPerStep > 0 ? (tokensPerStep * 1000) / this.msPerStep : 0;
    let i = 0;
    for (const p of pending) {
      let sum = 0;
      for (let k = 0; k < p.losses.length; k++) sum += values[i++] as number;
      const loss = sum / p.losses.length;
      const gradNorm = p.gradNorm ? (values[i++] as number) : undefined;
      this.lastLoss = loss;
      const info: StepInfo = { step: p.step, loss, lr: p.lr, tokensPerSec: this.tokensPerSec, msPerStep: this.msPerStep };
      if (gradNorm !== undefined) info.gradNorm = gradNorm;
      this.log.push({ step: p.step, loss, lr: p.lr, tokensPerSec: this.tokensPerSec, msPerStep: this.msPerStep });
      this.opts.callbacks?.onStep?.(info);
    }
  }

  private async maybeEval(): Promise<void> {
    const { evalInterval, val, callbacks } = this.opts;
    if (!evalInterval || !val) return;
    if (this.step % evalInterval !== 0) return;
    if (this.step === 0 && !this.opts.evalAtStart) return;
    await this.flush();
    const valLoss = await this.evaluate(val);
    const trainLoss = await this.evaluate(this.opts.train);
    this.lastValLoss = valLoss;
    const entry = this.log[this.log.length - 1];
    if (entry && entry.step === this.step) entry.valLoss = valLoss;
    else this.log.push({ step: this.step, loss: trainLoss, valLoss });
    callbacks?.onEval?.({ step: this.step, trainLoss, valLoss });
  }

  private async maybeSample(): Promise<void> {
    const { sampleInterval, callbacks, tokenizer } = this.opts;
    if (!sampleInterval || !tokenizer || this.step === 0 || this.step % sampleInterval !== 0) return;
    await this.flush();
    const text = await this.sample();
    if (text !== null) callbacks?.onSample?.({ step: this.step, text });
  }

  private async maybeCheckpoint(): Promise<void> {
    const { checkpointInterval, callbacks } = this.opts;
    if (!checkpointInterval || !callbacks?.onCheckpoint || this.step === 0 || this.step % checkpointInterval !== 0) return;
    await this.flush();
    const saved = await this.checkpoint();
    await callbacks.onCheckpoint({ step: this.step, ...saved });
  }

  /** One optimizer step (gradAccumSteps micro-batches). Enqueues the loss readback. */
  private trainStep(): void {
    const { model, optimizer, train, batchSize, gradAccumSteps, clipNorm } = this.opts;
    const T = train.blockSize;
    const lr = this.opts.lrSchedule ? this.opts.lrSchedule(this.step) : optimizer.lr;
    optimizer.lr = lr;
    this.lr = lr;
    const losses: Tensor[] = [];
    for (let micro = 0; micro < gradAccumSteps; micro++) {
      const { x, y } = train.getBatch(batchSize, this.rng);
      const xs = this.upload(x, batchSize, T);
      const ys = this.upload(y, batchSize, T);
      const loss = tidy(() => {
        const out = model.forward(xs, ys);
        const l = out.loss as Tensor;
        const scaled = gradAccumSteps > 1 ? l.div(gradAccumSteps) : l;
        scaled.backward();
        return l.detach();
      });
      xs.free();
      ys.free();
      losses.push(loss);
    }
    const gradNorm = clipNorm > 0 ? clipGradNorm(model.parameters(), clipNorm) : null;
    optimizer.step();
    optimizer.zeroGrad();
    this.step++;
    this.pending.push({ step: this.step, losses, gradNorm, lr });
  }

  /** Run until `maxSteps` (or `stop()`); resolves when finished. */
  async run(): Promise<TrainerState> {
    if (this.status === "running" || this.status === "paused") throw new Error("Trainer.run: already running");
    this.status = "running";
    this.stopRequested = false;
    this.opts.model.train();
    this.intervalStart = performance.now();
    try {
      while (this.step < this.opts.maxSteps && !this.stopRequested) {
        await this.waitWhilePaused();
        if (this.stopRequested) break;
        await this.maybeEval();
        await this.maybeSample();
        await this.maybeCheckpoint();
        this.trainStep();
        if (this.pending.length >= this.opts.lossReadbackInterval) {
          await this.flush();
          await yieldToEventLoop();
        }
      }
      await this.flush();
      if (!this.stopRequested) {
        await this.maybeEval();
        await this.maybeSample();
        await this.maybeCheckpoint();
      }
    } finally {
      await this.flush();
      this.status = this.stopRequested ? "stopped" : "done";
    }
    return this.state;
  }
}
