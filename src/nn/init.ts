/**
 * Parameter initialisers. Each returns a fresh f32 tensor on `device` (default: the default device). Random
 * initialisers draw their kernel seed from the global generator (`manualSeed` makes them reproducible) and are
 * expressed through tensor ops so they run identically on every backend.
 */
import type { Device } from "../backend/types";
import { Tensor } from "../tensor/tensor";
import { noGrad } from "../autograd/graph";
import type { Shape } from "../tensor/shape";

/** Creation options shared by the initialisers. */
export interface InitOptions {
  device?: Device;
  /** Explicit kernel seed (otherwise drawn from the global generator). */
  seed?: number;
}

/** N(mean, std²). */
export function normal(shape: Shape, mean = 0, std = 1, opts: InitOptions = {}): Tensor {
  return noGrad(() => {
    const z = Tensor.randn(shape, { device: opts.device, seed: opts.seed });
    if (std === 1 && mean === 0) return z;
    const s = std === 1 ? z : z.mul(std);
    const out = mean === 0 ? s : s.add(mean);
    if (out !== z) z.free();
    if (s !== z && s !== out) s.free();
    return out;
  });
}

/** U[lo, hi). */
export function uniform(shape: Shape, lo = 0, hi = 1, opts: InitOptions = {}): Tensor {
  return noGrad(() => {
    const u = Tensor.rand(shape, { device: opts.device, seed: opts.seed });
    if (lo === 0 && hi === 1) return u;
    const s = u.mul(hi - lo);
    const out = lo === 0 ? s : s.add(lo);
    u.free();
    if (s !== out) s.free();
    return out;
  });
}

/** All zeros. */
export function zeros(shape: Shape, opts: InitOptions = {}): Tensor {
  return Tensor.zeros(shape, { device: opts.device });
}

/** All ones. */
export function ones(shape: Shape, opts: InitOptions = {}): Tensor {
  return Tensor.ones(shape, { device: opts.device });
}

/** Constant fill. */
export function constant(shape: Shape, value: number, opts: InitOptions = {}): Tensor {
  return Tensor.full(shape, value, { device: opts.device });
}

/** Fan-in/fan-out of a weight shaped [out, in, ...receptive]. */
function fans(shape: Shape): { fanIn: number; fanOut: number } {
  if (shape.length < 2) return { fanIn: shape[0] ?? 1, fanOut: shape[0] ?? 1 };
  let receptive = 1;
  for (let i = 2; i < shape.length; i++) receptive *= shape[i] as number;
  return { fanIn: (shape[1] as number) * receptive, fanOut: (shape[0] as number) * receptive };
}

/** Kaiming (He) uniform: U(−b, b) with b = gain·√(3/fan_in); gain √2 for ReLU. */
export function kaimingUniform(shape: Shape, opts: InitOptions & { gain?: number; mode?: "fanIn" | "fanOut" } = {}): Tensor {
  const { fanIn, fanOut } = fans(shape);
  const fan = (opts.mode ?? "fanIn") === "fanIn" ? fanIn : fanOut;
  const bound = (opts.gain ?? Math.SQRT2) * Math.sqrt(3 / Math.max(1, fan));
  return uniform(shape, -bound, bound, opts);
}

/** Kaiming (He) normal: N(0, gain²/fan_in). */
export function kaimingNormal(shape: Shape, opts: InitOptions & { gain?: number; mode?: "fanIn" | "fanOut" } = {}): Tensor {
  const { fanIn, fanOut } = fans(shape);
  const fan = (opts.mode ?? "fanIn") === "fanIn" ? fanIn : fanOut;
  return normal(shape, 0, (opts.gain ?? Math.SQRT2) / Math.sqrt(Math.max(1, fan)), opts);
}

/** Xavier/Glorot uniform: U(−b, b) with b = gain·√(6/(fan_in+fan_out)). */
export function xavierUniform(shape: Shape, opts: InitOptions & { gain?: number } = {}): Tensor {
  const { fanIn, fanOut } = fans(shape);
  const bound = (opts.gain ?? 1) * Math.sqrt(6 / Math.max(1, fanIn + fanOut));
  return uniform(shape, -bound, bound, opts);
}
