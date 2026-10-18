/**
 * Parameter-free activation modules.
 */
import type { Tensor } from "../tensor/tensor";
import { Module } from "./module";

/** GELU (tanh approximation, as in GPT-2). */
export class GELU extends Module {
  forward(x: Tensor): Tensor {
    return x.gelu();
  }
}

/** Rectified linear unit. */
export class ReLU extends Module {
  forward(x: Tensor): Tensor {
    return x.relu();
  }
}

/** Hyperbolic tangent. */
export class Tanh extends Module {
  forward(x: Tensor): Tensor {
    return x.tanh();
  }
}

/** Logistic sigmoid. */
export class Sigmoid extends Module {
  forward(x: Tensor): Tensor {
    return x.sigmoid();
  }
}

/** Softmax along `dim`. */
export class Softmax extends Module {
  constructor(readonly dim = -1) {
    super();
  }

  forward(x: Tensor): Tensor {
    return x.softmax(this.dim);
  }
}
