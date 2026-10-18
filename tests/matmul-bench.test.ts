import { describe, expect, it } from "vitest";
import { Tensor } from "../src/tensor/tensor";
import { cpuBackend } from "../src/backend/cpu/index";

/** Sanity bound, not a performance target: a 256×256×256 matmul must finish well under a second. */
describe("matmul performance sanity", () => {
  it("256x256 matmul runs under a generous bound", () => {
    const n = 256;
    const a = Tensor.randn([n, n], { seed: 1 });
    const b = Tensor.randn([n, n], { seed: 2 });
    const out = Tensor.empty([n, n]);
    // warm-up
    cpuBackend.matmul(a.buf, b.buf, out.buf, 1, n, n, n, false, false, 0, 0);
    const reps = 3;
    const t0 = performance.now();
    for (let i = 0; i < reps; i++) cpuBackend.matmul(a.buf, b.buf, out.buf, 1, n, n, n, false, false, 0, 0);
    const ms = (performance.now() - t0) / reps;
    const gflops = (2 * n * n * n) / (ms * 1e6);
    // eslint-disable-next-line no-console
    console.log(`256x256 matmul: ${ms.toFixed(2)} ms (${gflops.toFixed(2)} GFLOPS)`);
    expect(ms).toBeLessThan(1000);
  });

  it("transposed-B fast path is not dramatically slower than NN", () => {
    const n = 256;
    const a = Tensor.randn([n, n], { seed: 3 });
    const b = Tensor.randn([n, n], { seed: 4 });
    const out = Tensor.empty([n, n]);
    const time = (transB: boolean): number => {
      cpuBackend.matmul(a.buf, b.buf, out.buf, 1, n, n, n, false, transB, 0, 0);
      const t0 = performance.now();
      for (let i = 0; i < 3; i++) cpuBackend.matmul(a.buf, b.buf, out.buf, 1, n, n, n, false, transB, 0, 0);
      return (performance.now() - t0) / 3;
    };
    const nn = time(false);
    const nt = time(true);
    expect(nt).toBeLessThan(Math.max(1000, nn * 10));
  });
});
