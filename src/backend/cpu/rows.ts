/**
 * Fused row kernels: softmax / log-softmax (+ backward), layer norm (+ backward), cross-entropy.
 * All use numerically stable formulations (max subtraction, two-pass variance).
 */

/** Row-wise (log-)softmax. */
export function softmaxKernel(a: Float32Array, rows: number, cols: number, out: Float32Array, log: boolean): void {
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let m = -Infinity;
    for (let c = 0; c < cols; c++) { const v = a[base + c] as number; if (v > m) m = v; }
    let s = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp((a[base + c] as number) - m);
      out[base + c] = e;
      s += e;
    }
    if (log) {
      const lse = m + Math.log(s);
      for (let c = 0; c < cols; c++) out[base + c] = (a[base + c] as number) - lse;
    } else {
      const inv = 1 / s;
      for (let c = 0; c < cols; c++) out[base + c] = (out[base + c] as number) * inv;
    }
  }
}

/**
 * Backward of (log-)softmax given the forward output.
 *  softmax:     gradIn = out * (g - Σ g·out)
 *  log-softmax: gradIn = g - exp(out) · Σ g
 */
export function softmaxBackwardKernel(
  out: Float32Array, g: Float32Array, rows: number, cols: number, gradIn: Float32Array, log: boolean,
): void {
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    if (log) {
      let s = 0;
      for (let c = 0; c < cols; c++) s += g[base + c] as number;
      for (let c = 0; c < cols; c++) gradIn[base + c] = (g[base + c] as number) - Math.exp(out[base + c] as number) * s;
    } else {
      let s = 0;
      for (let c = 0; c < cols; c++) s += (g[base + c] as number) * (out[base + c] as number);
      for (let c = 0; c < cols; c++) gradIn[base + c] = (out[base + c] as number) * ((g[base + c] as number) - s);
    }
  }
}

/** Row-wise layer norm; meanRstd receives (mean, rstd) pairs. */
export function layerNormKernel(
  x: Float32Array, gamma: Float32Array, beta: Float32Array, rows: number, cols: number, eps: number,
  out: Float32Array, meanRstd: Float32Array,
): void {
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += x[base + c] as number;
    mean /= cols;
    let v = 0;
    for (let c = 0; c < cols; c++) { const d = (x[base + c] as number) - mean; v += d * d; }
    v /= cols;
    const rstd = 1 / Math.sqrt(v + eps);
    meanRstd[2 * r] = mean;
    meanRstd[2 * r + 1] = rstd;
    for (let c = 0; c < cols; c++) {
      out[base + c] = ((x[base + c] as number) - mean) * rstd * (gamma[c] as number) + (beta[c] as number);
    }
  }
}

/**
 * Layer norm backward. gradGamma/gradBeta are overwritten (not accumulated).
 *  xhat = (x-mean)·rstd ; dxhat = g·gamma
 *  dx = rstd · (dxhat − mean(dxhat) − xhat · mean(dxhat·xhat))
 */
export function layerNormBackwardKernel(
  x: Float32Array, gamma: Float32Array, meanRstd: Float32Array, g: Float32Array, rows: number, cols: number,
  gradX: Float32Array, gradGamma: Float32Array, gradBeta: Float32Array,
): void {
  gradGamma.fill(0, 0, cols);
  gradBeta.fill(0, 0, cols);
  const dgamma = new Float64Array(cols);
  const dbeta = new Float64Array(cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    const mean = meanRstd[2 * r] as number;
    const rstd = meanRstd[2 * r + 1] as number;
    let sumDx = 0;
    let sumDxX = 0;
    for (let c = 0; c < cols; c++) {
      const xhat = ((x[base + c] as number) - mean) * rstd;
      const gc = g[base + c] as number;
      const dxhat = gc * (gamma[c] as number);
      sumDx += dxhat;
      sumDxX += dxhat * xhat;
      dgamma[c] = (dgamma[c] as number) + gc * xhat;
      dbeta[c] = (dbeta[c] as number) + gc;
    }
    const mDx = sumDx / cols;
    const mDxX = sumDxX / cols;
    for (let c = 0; c < cols; c++) {
      const xhat = ((x[base + c] as number) - mean) * rstd;
      const dxhat = (g[base + c] as number) * (gamma[c] as number);
      gradX[base + c] = rstd * (dxhat - mDx - xhat * mDxX);
    }
  }
  for (let c = 0; c < cols; c++) {
    gradGamma[c] = dgamma[c] as number;
    gradBeta[c] = dbeta[c] as number;
  }
}

/**
 * Fused cross-entropy. lossOut[r] = logsumexp(row) − logit[target]; rows whose target equals ignoreIndex
 * get loss 0 and grad 0. gradOut (if given) = (softmax − onehot) / denom with denom = validCount ?? rows.
 */
export function crossEntropyKernel(
  logits: Float32Array, targets: Int32Array, rows: number, cols: number, lossOut: Float32Array,
  gradOut: Float32Array | undefined, ignoreIndex: number | undefined, validCount: number | undefined,
): void {
  const denom = Math.max(1, validCount ?? rows);
  const inv = 1 / denom;
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    const t = targets[r] as number;
    if ((ignoreIndex !== undefined && t === ignoreIndex) || t < 0 || t >= cols) {
      lossOut[r] = 0;
      if (gradOut) gradOut.fill(0, base, base + cols);
      continue;
    }
    let m = -Infinity;
    for (let c = 0; c < cols; c++) { const v = logits[base + c] as number; if (v > m) m = v; }
    let s = 0;
    for (let c = 0; c < cols; c++) s += Math.exp((logits[base + c] as number) - m);
    const lse = m + Math.log(s);
    lossOut[r] = lse - (logits[base + t] as number);
    if (gradOut) {
      const invS = 1 / s;
      for (let c = 0; c < cols; c++) {
        gradOut[base + c] = Math.exp((logits[base + c] as number) - m) * invS * inv;
      }
      gradOut[base + t] = (gradOut[base + t] as number) - inv;
    }
  }
}
