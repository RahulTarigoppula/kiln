/**
 * Learning-rate schedules as pure functions of the step (0-based).
 */

/** A schedule maps a step to a learning rate. */
export type LRSchedule = (step: number) => number;

/** Options for {@link cosineWithWarmup}. */
export interface CosineWarmupOptions {
  /** Linear warm-up steps (lr grows from ~0 to lrMax). */
  warmup: number;
  /** Step at which the cosine decay reaches `lrMin` (stays there afterwards). */
  total: number;
  lrMax: number;
  lrMin?: number;
}

/** nanoGPT schedule: linear warm-up, cosine decay to `lrMin`, then constant. */
export function cosineWithWarmup(step: number, opts: CosineWarmupOptions): number {
  const lrMin = opts.lrMin ?? opts.lrMax / 10;
  if (step < opts.warmup) return opts.lrMax * (step + 1) / (opts.warmup + 1);
  if (step >= opts.total) return lrMin;
  const span = Math.max(1, opts.total - opts.warmup);
  const ratio = (step - opts.warmup) / span;
  const coeff = 0.5 * (1 + Math.cos(Math.PI * ratio));
  return lrMin + coeff * (opts.lrMax - lrMin);
}

/** Curry {@link cosineWithWarmup} into an {@link LRSchedule}. */
export function makeCosineWithWarmup(opts: CosineWarmupOptions): LRSchedule {
  return (step) => cosineWithWarmup(step, opts);
}

/** Options for {@link stepSchedule}. */
export interface StepScheduleOptions {
  lr: number;
  /** Multiply by `gamma` every `stepSize` steps. */
  stepSize: number;
  gamma: number;
}

/** Step decay: `lr · gamma^floor(step / stepSize)`. */
export function stepSchedule(step: number, opts: StepScheduleOptions): number {
  return opts.lr * Math.pow(opts.gamma, Math.floor(step / Math.max(1, opts.stepSize)));
}

/** Curry {@link stepSchedule}. */
export function makeStepSchedule(opts: StepScheduleOptions): LRSchedule {
  return (step) => stepSchedule(step, opts);
}

/** Constant learning rate. */
export function constantSchedule(lr: number): LRSchedule {
  return () => lr;
}
