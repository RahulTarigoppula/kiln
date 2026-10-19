/**
 * Principal component analysis by power iteration with deflation, on a dense row-major matrix.
 * Suitable for token-embedding tables (vocab x d) in the browser.
 */

export interface PcaResult {
  /** k components, each of length `cols` (unit norm), row-major kxcols. */
  components: Float32Array;
  /** Projected rows, row-major rowsxk. */
  projected: Float32Array;
  /** Column means that were subtracted. */
  mean: Float32Array;
  /** Variance captured by each component (eigenvalues). */
  eigenvalues: number[];
  /** Fraction of total variance explained per component. */
  explained: number[];
}

export interface PcaOptions {
  components?: number;
  iterations?: number;
  /** Deterministic start vector seed. */
  seed?: number;
  tolerance?: number;
}

function lcg(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function pca(matrix: Float32Array, rows: number, cols: number, opts: PcaOptions = {}): PcaResult {
  const k = Math.max(1, Math.min(opts.components ?? 2, cols, rows));
  const iters = opts.iterations ?? 120;
  const tol = opts.tolerance ?? 1e-7;
  if (matrix.length < rows * cols) throw new Error("pca: matrix is smaller than rows*cols");

  // Centre.
  const mean = new Float32Array(cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    for (let c = 0; c < cols; c++) mean[c] = (mean[c] ?? 0) + (matrix[base + c] ?? 0);
  }
  for (let c = 0; c < cols; c++) mean[c] = (mean[c] ?? 0) / Math.max(1, rows);
  const X = new Float64Array(rows * cols);
  let total = 0;
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    for (let c = 0; c < cols; c++) {
      const v = (matrix[base + c] ?? 0) - (mean[c] ?? 0);
      X[base + c] = v;
      total += v * v;
    }
  }
  const denom = Math.max(1, rows - 1);
  total /= denom;

  const components = new Float32Array(k * cols);
  const eigenvalues: number[] = [];
  const rand = lcg(opts.seed ?? 7);
  const v = new Float64Array(cols);
  const xv = new Float64Array(rows);
  const next = new Float64Array(cols);

  for (let comp = 0; comp < k; comp++) {
    for (let c = 0; c < cols; c++) v[c] = rand() - 0.5;
    normalize(v);
    let lambda = 0;
    for (let it = 0; it < iters; it++) {
      // xv = X v
      for (let r = 0; r < rows; r++) {
        const base = r * cols;
        let s = 0;
        for (let c = 0; c < cols; c++) s += (X[base + c] ?? 0) * (v[c] ?? 0);
        xv[r] = s;
      }
      // next = X^T xv
      next.fill(0);
      for (let r = 0; r < rows; r++) {
        const base = r * cols;
        const w = xv[r] ?? 0;
        if (w === 0) continue;
        for (let c = 0; c < cols; c++) next[c] = (next[c] ?? 0) + (X[base + c] ?? 0) * w;
      }
      const newLambda = normalize(next) / denom;
      let diff = 0;
      for (let c = 0; c < cols; c++) {
        const d = (next[c] ?? 0) - (v[c] ?? 0);
        diff += d * d;
        v[c] = next[c] ?? 0;
      }
      lambda = newLambda;
      if (diff < tol) break;
    }
    eigenvalues.push(lambda);
    for (let c = 0; c < cols; c++) components[comp * cols + c] = v[c] ?? 0;
    // Deflate: X <- X − (X v) vᵀ
    for (let r = 0; r < rows; r++) {
      const base = r * cols;
      let s = 0;
      for (let c = 0; c < cols; c++) s += (X[base + c] ?? 0) * (v[c] ?? 0);
      if (s === 0) continue;
      for (let c = 0; c < cols; c++) X[base + c] = (X[base + c] ?? 0) - s * (v[c] ?? 0);
    }
  }

  // Project original centred data.
  const projected = new Float32Array(rows * k);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    for (let comp = 0; comp < k; comp++) {
      let s = 0;
      const cb = comp * cols;
      for (let c = 0; c < cols; c++) s += ((matrix[base + c] ?? 0) - (mean[c] ?? 0)) * (components[cb + c] ?? 0);
      projected[r * k + comp] = s;
    }
  }
  const explained = eigenvalues.map((e) => (total > 0 ? e / total : 0));
  return { components, projected, mean, eigenvalues, explained };
}

function normalize(v: Float64Array): number {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += (v[i] ?? 0) * (v[i] ?? 0);
  n = Math.sqrt(n);
  if (n > 0) for (let i = 0; i < v.length; i++) v[i] = (v[i] ?? 0) / n;
  return n;
}
