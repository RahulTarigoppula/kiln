/**
 * CPU backend: every `Backend` method implemented on Float32Array / Int32Array views over ArrayBuffers.
 * A `Buf.handle` on this backend is always an `ArrayBuffer`.
 */
import type { Backend, BinaryOp, Buf, ReduceOp, UnaryOp } from "../types";
import { hashNormal, hashUniform } from "../../tensor/random";
import { binaryKernel, copyStridedKernel, unaryKernel } from "./elementwise";
import { matmulBatched } from "./matmul";
import { argmaxKernel, reduceKernel, sumSquaresKernel } from "./reduce";
import { crossEntropyKernel, layerNormBackwardKernel, layerNormKernel, softmaxBackwardKernel, softmaxKernel } from "./rows";

export { gemmNN, gemmNT, gemmTN, gemmTT } from "./matmul";

/** CPU buffer: an ArrayBuffer handle. */
export interface CpuBuf extends Buf {
  readonly device: "cpu";
  readonly handle: ArrayBuffer;
}

function f32(b: Buf): Float32Array {
  return new Float32Array(b.handle as ArrayBuffer);
}
function i32(b: Buf): Int32Array {
  return new Int32Array(b.handle as ArrayBuffer);
}
function u32(b: Buf): Uint32Array {
  return new Uint32Array(b.handle as ArrayBuffer);
}

/** The CPU backend singleton class. */
export class CpuBackend implements Backend {
  readonly device = "cpu" as const;

  alloc(byteLength: number): CpuBuf {
    return { byteLength, device: "cpu", handle: new ArrayBuffer(byteLength) };
  }

  free(_buf: Buf): void {
    // Garbage-collected; nothing to do.
  }

  write(buf: Buf, data: ArrayBufferView): void {
    new Uint8Array(buf.handle as ArrayBuffer).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  }

  read(buf: Buf, byteLength: number): Promise<ArrayBuffer> {
    return Promise.resolve((buf.handle as ArrayBuffer).slice(0, byteLength));
  }

  copy(src: Buf, dst: Buf, byteLength: number, srcByteOffset = 0, dstByteOffset = 0): void {
    new Uint8Array(dst.handle as ArrayBuffer, dstByteOffset, byteLength).set(
      new Uint8Array(src.handle as ArrayBuffer, srcByteOffset, byteLength),
    );
  }

  unary(op: UnaryOp, a: Buf, n: number, out: Buf, alpha = 0, beta = 0): void {
    unaryKernel(op, f32(a), n, f32(out), alpha, beta);
  }

  binary(op: BinaryOp, a: Buf, b: Buf, out: Buf, outShape: number[], aStrides: number[], bStrides: number[]): void {
    binaryKernel(op, f32(a), f32(b), f32(out), outShape, aStrides, bStrides);
  }

  copyStrided(a: Buf, aShape: number[], aStrides: number[], aOffset: number, out: Buf): void {
    copyStridedKernel(i32(a), aShape, aStrides, aOffset, i32(out));
  }

  where(cond: Buf, a: Buf, b: Buf, n: number, out: Buf): void {
    const c = u32(cond);
    const A = f32(a);
    const B = f32(b);
    const O = f32(out);
    for (let i = 0; i < n; i++) O[i] = (c[i] as number) !== 0 ? (A[i] as number) : (B[i] as number);
  }

  maskedFill(a: Buf, mask: Buf, value: number, n: number, out: Buf): void {
    const m = u32(mask);
    const A = f32(a);
    const O = f32(out);
    for (let i = 0; i < n; i++) O[i] = (m[i] as number) !== 0 ? value : (A[i] as number);
  }

  fill(buf: Buf, n: number, value: number): void {
    f32(buf).fill(value, 0, n);
  }

  matmul(
    a: Buf, b: Buf, out: Buf, batch: number, M: number, K: number, N: number,
    transA: boolean, transB: boolean, aBatchStride: number, bBatchStride: number,
  ): void {
    matmulBatched(f32(a), f32(b), f32(out), batch, M, K, N, transA, transB, aBatchStride, bBatchStride);
  }

  reduce(op: ReduceOp, a: Buf, shape: number[], axes: number[], out: Buf): void {
    reduceKernel(op, f32(a), shape, axes, f32(out));
  }

  argmax(a: Buf, rows: number, cols: number, out: Buf): void {
    argmaxKernel(f32(a), rows, cols, i32(out));
  }

  softmax(a: Buf, rows: number, cols: number, out: Buf, log: boolean): void {
    softmaxKernel(f32(a), rows, cols, f32(out), log);
  }

  softmaxBackward(out: Buf, gradOut: Buf, rows: number, cols: number, gradIn: Buf, log: boolean): void {
    softmaxBackwardKernel(f32(out), f32(gradOut), rows, cols, f32(gradIn), log);
  }

  layerNorm(x: Buf, gamma: Buf, beta: Buf, rows: number, cols: number, eps: number, out: Buf, meanRstd: Buf): void {
    layerNormKernel(f32(x), f32(gamma), f32(beta), rows, cols, eps, f32(out), f32(meanRstd));
  }

  layerNormBackward(
    x: Buf, gamma: Buf, meanRstd: Buf, gradOut: Buf, rows: number, cols: number,
    gradX: Buf, gradGamma: Buf, gradBeta: Buf,
  ): void {
    layerNormBackwardKernel(f32(x), f32(gamma), f32(meanRstd), f32(gradOut), rows, cols, f32(gradX), f32(gradGamma), f32(gradBeta));
  }

  crossEntropy(
    logits: Buf, targets: Buf, rows: number, cols: number, lossOut: Buf, gradOut?: Buf,
    ignoreIndex?: number, validCount?: number,
  ): void {
    crossEntropyKernel(f32(logits), i32(targets), rows, cols, f32(lossOut), gradOut ? f32(gradOut) : undefined, ignoreIndex, validCount);
  }

  gather(table: Buf, ids: Buf, n: number, dim: number, out: Buf): void {
    const T = f32(table);
    const I = i32(ids);
    const O = f32(out);
    const rows = T.length / dim;
    for (let i = 0; i < n; i++) {
      const id = I[i] as number;
      const dst = i * dim;
      if (id < 0 || id >= rows) {
        O.fill(0, dst, dst + dim);
        continue;
      }
      O.set(T.subarray(id * dim, id * dim + dim), dst);
    }
  }

  scatterAdd(gradOut: Buf, ids: Buf, n: number, dim: number, gradTable: Buf): void {
    const G = f32(gradOut);
    const I = i32(ids);
    const T = f32(gradTable);
    const rows = T.length / dim;
    for (let i = 0; i < n; i++) {
      const id = I[i] as number;
      if (id < 0 || id >= rows) continue;
      const src = i * dim;
      const dst = id * dim;
      for (let c = 0; c < dim; c++) T[dst + c] = (T[dst + c] as number) + (G[src + c] as number);
    }
  }

  adamw(
    param: Buf, grad: Buf, m: Buf, v: Buf, n: number, lr: number, beta1: number, beta2: number,
    eps: number, weightDecay: number, step: number,
  ): void {
    const P = f32(param);
    const G = f32(grad);
    const M = f32(m);
    const V = f32(v);
    const bc1 = 1 - Math.pow(beta1, step);
    const bc2 = 1 - Math.pow(beta2, step);
    const decay = 1 - lr * weightDecay;
    const stepSize = lr / bc1;
    const invBc2 = 1 / Math.sqrt(bc2);
    for (let i = 0; i < n; i++) {
      const g = G[i] as number;
      const mi = beta1 * (M[i] as number) + (1 - beta1) * g;
      const vi = beta2 * (V[i] as number) + (1 - beta2) * g * g;
      M[i] = mi;
      V[i] = vi;
      P[i] = (P[i] as number) * decay - stepSize * mi / (Math.sqrt(vi) * invBc2 + eps);
    }
  }

  sumSquares(a: Buf, n: number, out: Buf): void {
    f32(out)[0] = sumSquaresKernel(f32(a), n);
  }

  randn(buf: Buf, n: number, seed: number): void {
    const O = f32(buf);
    const s = seed >>> 0;
    for (let i = 0; i < n; i++) O[i] = hashNormal(s, i);
  }

  rand(buf: Buf, n: number, seed: number): void {
    const O = f32(buf);
    const s = seed >>> 0;
    for (let i = 0; i < n; i++) O[i] = hashUniform(s, i);
  }

  dropoutMask(mask: Buf, n: number, p: number, seed: number): void {
    const O = f32(mask);
    const s = seed >>> 0;
    for (let i = 0; i < n; i++) O[i] = hashUniform(s, i) < p ? 0 : 1;
  }

  sync(): Promise<void> {
    return Promise.resolve();
  }
}

/** Shared CPU backend instance. */
export const cpuBackend: CpuBackend = new CpuBackend();
