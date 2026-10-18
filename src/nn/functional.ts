/**
 * Functional forms of the layers (re-exported from the autograd ops for convenience) plus a few helpers.
 */
import type { Tensor } from "../tensor/tensor";
import * as ops from "../autograd/ops";

export {
  gelu, relu, tanh, sigmoid, softmax, logSoftmax, crossEntropy, layerNorm, dropout, embedding, matmul,
  type CrossEntropyOptions,
} from "../autograd/ops";

/** `x·Wᵀ (+ b)` with weight `[out, in]`. */
export function linear(x: Tensor, weight: Tensor, bias?: Tensor | null): Tensor {
  const y = ops.matmul(x, ops.transpose(weight, 0, 1));
  return bias ? ops.add(y, bias) : y;
}

/** Mean squared error between `pred` and `target`. */
export function mseLoss(pred: Tensor, target: Tensor): Tensor {
  return ops.mean(ops.square(ops.sub(pred, target)));
}
