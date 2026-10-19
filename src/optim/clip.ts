/**
 * Global-norm gradient clipping without host readbacks: the squared norms are reduced on the device
 * (`backend.sumSquares` per gradient -> one small concatenation -> sum) and the clip coefficient
 * `min(1, maxNorm / (norm + eps))` is applied with broadcasting ops, so the whole thing is a handful of
 * enqueued kernels on WebGPU. The returned scalar tensor holds the pre-clip norm; `await norm.item()` only
 * when you need the number.
 */
import { Tensor } from "../tensor/tensor";
import { escape, tidy } from "../tensor/scope";
import { noGrad } from "../autograd/graph";
import { cat, contiguousRaw } from "../autograd/ops";

/** Clip the gradients of `params` in place so their global L2 norm is at most `maxNorm`. Returns the norm. */
export function clipGradNorm(params: readonly Tensor[], maxNorm: number, eps = 1e-6): Tensor {
  const withGrad = params.filter((p) => p.grad !== null);
  const first = withGrad[0];
  if (!first) return escape(Tensor.zeros([], { device: params[0]?.device ?? "cpu" }));
  return noGrad(() => tidy(() => {
    const sq = withGrad.map((p) => {
      const g = p.grad as Tensor;
      const gd = contiguousRaw(g);
      const out = Tensor.empty([1], { device: g.device });
      g.backend.sumSquares(gd.buf, g.numel, out.buf);
      return out;
    });
    const total = sq.length === 1 ? (sq[0] as Tensor) : cat(sq, 0).sum();
    const norm = total.reshape().sqrt(); // 0-d
    const scale = norm.add(eps).reciprocal().mul(maxNorm).minimum(1);
    for (const p of withGrad) {
      const old = p.grad as Tensor;
      p.grad = escape(old.mul(scale));
      old.free();
    }
    return norm;
  }));
}

/** Global L2 norm of the gradients (device-side; no clipping). */
export function gradNorm(params: readonly Tensor[]): Tensor {
  const withGrad = params.filter((p) => p.grad !== null);
  const first = withGrad[0];
  if (!first) return escape(Tensor.zeros([], { device: params[0]?.device ?? "cpu" }));
  return noGrad(() => tidy(() => {
    const sq = withGrad.map((p) => {
      const g = p.grad as Tensor;
      const gd = contiguousRaw(g);
      const out = Tensor.empty([1], { device: g.device });
      g.backend.sumSquares(gd.buf, g.numel, out.buf);
      return out;
    });
    const total = sq.length === 1 ? (sq[0] as Tensor) : cat(sq, 0).sum();
    return total.reshape().sqrt();
  }));
}
