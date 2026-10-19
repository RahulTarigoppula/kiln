/**
 * CPU GEMM kernels on Float32Array.
 *
 *  - NN / TN share `gemmRows4`: 4 rows of C are accumulated together so every B element loaded in the
 *    inner j loop feeds four FMAs (4-way unroll over i). A is addressed through (rowStride, colStride) so a
 *    K×M-stored (transposed) A costs nothing extra - its four row reads become one contiguous quad.
 *    Tiles: K in blocks of 256, N in blocks of 512 so the 4 live C rows + the B block stay cache-resident.
 *  - NT (B stored N×K) uses a 2×2 register-blocked dot-product kernel: both operands stream along k.
 *  - TT is computed as (Bᵗ·Aᵗ)ᵗ through the NN kernel into a scratch tile, then transposed.
 */

const BK = 256;
const BN = 512;

/**
 * C[M×N] (row-major, overwritten) = A · B where A[i,k] = A[aOff + i*sAi + k*sAk] and B is K×N row-major.
 * NN: (sAi, sAk) = (K, 1); TN with A stored K×M: (sAi, sAk) = (1, M).
 */
function gemmRows4(
  A: Float32Array, aOff: number, sAi: number, sAk: number,
  B: Float32Array, bOff: number, C: Float32Array, cOff: number,
  M: number, K: number, N: number,
): void {
  C.fill(0, cOff, cOff + M * N);
  for (let k0 = 0; k0 < K; k0 += BK) {
    const k1 = Math.min(k0 + BK, K);
    for (let j0 = 0; j0 < N; j0 += BN) {
      const j1 = Math.min(j0 + BN, N);
      let i = 0;
      for (; i + 3 < M; i += 4) {
        const a0 = aOff + i * sAi, a1 = a0 + sAi, a2 = a1 + sAi, a3 = a2 + sAi;
        const c0 = cOff + i * N, c1 = c0 + N, c2 = c1 + N, c3 = c2 + N;
        for (let k = k0; k < k1; k++) {
          const kk = k * sAk;
          const v0 = A[a0 + kk] as number;
          const v1 = A[a1 + kk] as number;
          const v2 = A[a2 + kk] as number;
          const v3 = A[a3 + kk] as number;
          if (v0 === 0 && v1 === 0 && v2 === 0 && v3 === 0) continue;
          const bRow = bOff + k * N;
          for (let j = j0; j < j1; j++) {
            const b = B[bRow + j] as number;
            C[c0 + j] = (C[c0 + j] as number) + v0 * b;
            C[c1 + j] = (C[c1 + j] as number) + v1 * b;
            C[c2 + j] = (C[c2 + j] as number) + v2 * b;
            C[c3 + j] = (C[c3 + j] as number) + v3 * b;
          }
        }
      }
      for (; i < M; i++) {
        const aRow = aOff + i * sAi;
        const cRow = cOff + i * N;
        for (let k = k0; k < k1; k++) {
          const av = A[aRow + k * sAk] as number;
          if (av === 0) continue;
          const bRow = bOff + k * N;
          for (let j = j0; j < j1; j++) C[cRow + j] = (C[cRow + j] as number) + av * (B[bRow + j] as number);
        }
      }
    }
  }
}

/** C[M×N] = A[M×K] · B[K×N], all row-major; C is overwritten. */
export function gemmNN(
  A: Float32Array, aOff: number, B: Float32Array, bOff: number, C: Float32Array, cOff: number,
  M: number, K: number, N: number,
): void {
  gemmRows4(A, aOff, K, 1, B, bOff, C, cOff, M, K, N);
}

/** C[M×N] = Aᵗ · B where A is stored K×M row-major and B is K×N row-major. */
export function gemmTN(
  A: Float32Array, aOff: number, B: Float32Array, bOff: number, C: Float32Array, cOff: number,
  M: number, K: number, N: number,
): void {
  gemmRows4(A, aOff, 1, M, B, bOff, C, cOff, M, K, N);
}

/** C[M×N] = A[M×K] · Bᵗ where B is stored N×K row-major (2×2 register-blocked dot products). */
export function gemmNT(
  A: Float32Array, aOff: number, B: Float32Array, bOff: number, C: Float32Array, cOff: number,
  M: number, K: number, N: number,
): void {
  const JB = 64;
  for (let j0 = 0; j0 < N; j0 += JB) {
    const j1 = Math.min(j0 + JB, N);
    let i = 0;
    for (; i + 1 < M; i += 2) {
      const a0 = aOff + i * K;
      const a1 = a0 + K;
      const c0 = cOff + i * N;
      const c1 = c0 + N;
      let j = j0;
      for (; j + 1 < j1; j += 2) {
        const b0 = bOff + j * K;
        const b1 = b0 + K;
        let s00 = 0, s01 = 0, s10 = 0, s11 = 0;
        for (let k = 0; k < K; k++) {
          const x0 = A[a0 + k] as number;
          const x1 = A[a1 + k] as number;
          const y0 = B[b0 + k] as number;
          const y1 = B[b1 + k] as number;
          s00 += x0 * y0; s01 += x0 * y1; s10 += x1 * y0; s11 += x1 * y1;
        }
        C[c0 + j] = s00; C[c0 + j + 1] = s01; C[c1 + j] = s10; C[c1 + j + 1] = s11;
      }
      for (; j < j1; j++) {
        const b0 = bOff + j * K;
        let s0 = 0, s1 = 0;
        for (let k = 0; k < K; k++) {
          const y = B[b0 + k] as number;
          s0 += (A[a0 + k] as number) * y;
          s1 += (A[a1 + k] as number) * y;
        }
        C[c0 + j] = s0; C[c1 + j] = s1;
      }
    }
    for (; i < M; i++) {
      const a0 = aOff + i * K;
      const c0 = cOff + i * N;
      for (let j = j0; j < j1; j++) {
        const b0 = bOff + j * K;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        const K4 = K & ~3;
        let k = 0;
        for (; k < K4; k += 4) {
          s0 += (A[a0 + k] as number) * (B[b0 + k] as number);
          s1 += (A[a0 + k + 1] as number) * (B[b0 + k + 1] as number);
          s2 += (A[a0 + k + 2] as number) * (B[b0 + k + 2] as number);
          s3 += (A[a0 + k + 3] as number) * (B[b0 + k + 3] as number);
        }
        for (; k < K; k++) s0 += (A[a0 + k] as number) * (B[b0 + k] as number);
        C[c0 + j] = (s0 + s1) + (s2 + s3);
      }
    }
  }
}

/** C[M×N] = Aᵗ · Bᵗ with A stored K×M and B stored N×K: Cᵗ = B·A via gemmNN, then transposed. */
export function gemmTT(
  A: Float32Array, aOff: number, B: Float32Array, bOff: number, C: Float32Array, cOff: number,
  M: number, K: number, N: number,
): void {
  const tmp = new Float32Array(M * N);
  gemmNN(B, bOff, A, aOff, tmp, 0, N, K, M);
  for (let j = 0; j < N; j++) {
    const tRow = j * M;
    for (let i = 0; i < M; i++) C[cOff + i * N + j] = tmp[tRow + i] as number;
  }
}

/** Batched dispatcher used by the backend. */
export function matmulBatched(
  A: Float32Array, B: Float32Array, C: Float32Array,
  batch: number, M: number, K: number, N: number,
  transA: boolean, transB: boolean, aBatchStride: number, bBatchStride: number,
): void {
  const kernel = transA ? (transB ? gemmTT : gemmTN) : transB ? gemmNT : gemmNN;
  for (let b = 0; b < batch; b++) {
    kernel(A, b * aBatchStride, B, b * bBatchStride, C, b * M * N, M, K, N);
  }
}
