/**
 * CPU micro-benchmarks: matmul GFLOPS for square sizes and elementwise bandwidth. Prints markdown tables.
 * Run: `npm run bench`
 */
import { Tensor } from "../src/tensor/tensor";
import { cpuBackend } from "../src/backend/cpu/index";

function timeIt(fn: () => void, minMs = 300, maxReps = 50): number {
  fn(); // warm-up
  let reps = 0;
  const t0 = performance.now();
  let elapsed = 0;
  while ((elapsed < minMs && reps < maxReps) || reps < 2) {
    fn();
    reps++;
    elapsed = performance.now() - t0;
  }
  return elapsed / reps;
}

function benchMatmul(): void {
  console.log("### Matmul (f32, square, single thread)\n");
  console.log("| size | layout | ms/op | GFLOPS |");
  console.log("|-----:|:------:|------:|-------:|");
  for (const n of [128, 256, 512, 1024]) {
    const a = Tensor.randn([n, n], { seed: 1 });
    const b = Tensor.randn([n, n], { seed: 2 });
    const out = Tensor.empty([n, n]);
    const flops = 2 * n * n * n;
    for (const [label, tA, tB] of [["NN", false, false], ["NT", false, true], ["TN", true, false]] as const) {
      if (n === 1024 && label !== "NN") continue;
      const ms = timeIt(() => cpuBackend.matmul(a.buf, b.buf, out.buf, 1, n, n, n, tA, tB, 0, 0), n >= 1024 ? 100 : 300, n >= 1024 ? 3 : 50);
      console.log(`| ${n} | ${label} | ${ms.toFixed(2)} | ${(flops / (ms * 1e6)).toFixed(2)} |`);
    }
  }
  console.log();
}

function benchElementwise(): void {
  console.log("### Elementwise (f32)\n");
  console.log("| op | elements | ms/op | GB/s (read+write) |");
  console.log("|:---|---------:|------:|------------------:|");
  const n = 1 << 22; // 4M elements = 16 MB per tensor
  const a = Tensor.randn([n], { seed: 3 });
  const b = Tensor.randn([n], { seed: 4 });
  const out = Tensor.empty([n]);
  const row = (name: string, bytes: number, fn: () => void): void => {
    const ms = timeIt(fn);
    console.log(`| ${name} | ${n} | ${ms.toFixed(2)} | ${(bytes / (ms * 1e6)).toFixed(2)} |`);
  };
  row("add (contiguous)", 12 * n, () => cpuBackend.binary("add", a.buf, b.buf, out.buf, [n], [1], [1]));
  row("mul (broadcast scalar)", 8 * n, () => cpuBackend.binary("mul", a.buf, b.buf, out.buf, [n], [1], [0]));
  row("add (broadcast rows [2048,2048]+[2048])", 12 * n, () =>
    cpuBackend.binary("add", a.buf, b.buf, out.buf, [2048, 2048], [2048, 1], [0, 1]));
  row("exp", 8 * n, () => cpuBackend.unary("exp", a.buf, n, out.buf));
  row("gelu", 8 * n, () => cpuBackend.unary("gelu", a.buf, n, out.buf));
  row("scale", 8 * n, () => cpuBackend.unary("scale", a.buf, n, out.buf, 0.5));
  row("copyStrided (transpose 2048x2048)", 8 * n, () => cpuBackend.copyStrided(a.buf, [2048, 2048], [1, 2048], 0, out.buf));
  row("reduce sum (rows of 2048)", 4 * n, () => cpuBackend.reduce("sum", a.buf, [2048, 2048], [1], out.buf));
  row("softmax (rows of 2048)", 8 * n, () => cpuBackend.softmax(a.buf, 2048, 2048, out.buf, false));
  row("adamw", 20 * n, () => cpuBackend.adamw(out.buf, a.buf, b.buf, b.buf, n, 1e-3, 0.9, 0.95, 1e-8, 0.1, 1));
  console.log();
}

console.log(`# Kiln CPU bench (${process.platform}, node ${process.version})\n`);
benchMatmul();
benchElementwise();
