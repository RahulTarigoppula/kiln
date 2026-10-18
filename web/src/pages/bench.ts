/** Bench — matmul GFLOPS CPU vs WebGPU, elementwise GB/s, softmax / layernorm rows/s. */
import { h } from "../lib/dom";
import { Scope, signal } from "../lib/signals";
import { BarChart, chartTheme } from "../lib/charts";
import { adapterLabel, ensureGpu, gpuState, markBusy } from "../lib/gpu";
import { fmtMs } from "../lib/format";
import { formatMetric, MATMUL_SIZES, runBench, toMarkdown, type BenchRow } from "../lib/bench";
import { button, card, chartBox, chip, copyText, notice, pageHead, progressBar, toast } from "../components/ui";

export async function mount(root: HTMLElement): Promise<() => void> {
  const scope = new Scope();
  const rows = signal<BenchRow[]>([]);
  const running = signal(false);
  const progress = signal<{ label: string; frac: number }>({ label: "", frac: 0 });
  let includeCpu = true;
  let cpuMax = 1024;
  let abort: AbortController | null = null;

  const backend = await ensureGpu();
  const gpuLabel = backend ? adapterLabel(gpuState.peek().info) : null;

  // ---- header
  const runBtn = button("Run benchmark", { icon: "zap", variant: "primary", onClick: () => void start() });
  const stopBtn = button("Stop", { icon: "stop", variant: "ghost", onClick: () => abort?.abort() });
  stopBtn.style.display = "none";
  const copyBtn = button("Copy as Markdown", {
    icon: "copy",
    disabled: true,
    onClick: async () => {
      const ok = await copyText(toMarkdown(rows.peek(), gpuLabel));
      toast(ok ? "Copied results as Markdown" : "Clipboard unavailable", ok ? "ok" : "bad");
    },
  });
  root.appendChild(
    pageHead(
      "Kernel throughput",
      "Bench",
      "Raw Backend-interface kernels timed in this tab: tiled WGSL matmul against the register-blocked CPU GEMM, plus bandwidth-bound elementwise and fused row kernels.",
      [copyBtn, stopBtn, runBtn],
    ),
  );

  // ---- options + progress
  const cpuCheck = h("input", { type: "checkbox", checked: true, onchange: () => (includeCpu = cpuCheck.checked) });
  const cpuMaxSel = h(
    "select",
    { class: "select", style: "width:auto;height:30px;padding-right:28px", onchange: () => (cpuMax = Number(cpuMaxSel.value)) },
    ...[256, 512, 1024].map((n) => h("option", { value: String(n) }, `CPU matmul up to ${n}`)),
  );
  cpuMaxSel.value = "1024";
  const prog = progressBar();
  const progLabel = h("span", { class: "mono small" }, "");
  const optsCard = card(null, [
    h(
      "div",
      { class: "bench__opts" },
      h("label", { class: "check" }, cpuCheck, "Include CPU backend"),
      cpuMaxSel,
      h("span", { class: "spacer" }),
      backend ? chip("device", gpuLabel ?? "WebGPU", "ember") : chip("device", "CPU only", "warn"),
      chip("matmul sizes", `${MATMUL_SIZES[0]}…${MATMUL_SIZES[MATMUL_SIZES.length - 1]}`),
    ),
    h("div", { class: "stack stack--sm", style: "margin-top:14px" }, prog.el, progLabel),
  ]);
  optsCard.classList.add("card--tight");
  root.appendChild(optsCard);
  if (!backend) {
    root.appendChild(
      notice("warn", [
        h("b", null, "WebGPU is unavailable in this browser"),
        " — only the CPU backend will be measured. See the Tests page for browser guidance.",
      ]),
    );
  }

  // ---- chart + table
  const chartB = chartBox("chart--tall");
  chartB.setEmpty("No results yet", "Run the benchmark to compare GFLOPS across sizes.");
  const legend = h(
    "div",
    { class: "legend" },
    h("span", { class: "legend__item" }, h("span", { class: "legend__swatch", style: `--sw:${chartTheme().cool}` }), "CPU"),
    h("span", { class: "legend__item" }, h("span", { class: "legend__swatch", style: `--sw:${chartTheme().ember}` }), "WebGPU"),
  );
  const chartCard = card("Matmul throughput", [chartB.wrap], { sub: "GFLOPS, log scale · square f32 matrices", actions: [legend] });
  const tbody = h("tbody");
  const table = h(
    "div",
    { class: "table-wrap" },
    h(
      "table",
      { class: "table" },
      h(
        "thead",
        null,
        h("tr", null, h("th", null, "Kernel"), h("th", null, "Size"), h("th", { class: "num" }, "CPU"), h("th", { class: "num" }, "WebGPU"), h("th", { class: "num" }, "Speedup"), h("th", { class: "num" }, "ms / call")),
      ),
      tbody,
    ),
  );
  const tableCard = card("All kernels", [table], { sub: "GFLOPS · GB/s · rows/s" });
  root.appendChild(h("div", { class: "bench__grid", style: "margin-top:16px" }, chartCard, tableCard));

  const chart = new BarChart(chartB.wrap, {
    categories: [],
    series: [],
    logY: true,
    yLabel: "GFLOPS",
    valueFormat: (v) => formatMetric(v, "GFLOPS"),
    showValues: true,
  });
  scope.add(() => chart.destroy());

  scope.effect(() => {
    const rs = rows();
    const mm = rs.filter((r) => r.group === "matmul");
    const th = chartTheme();
    chart.setData({
      categories: mm.map((r) => r.size.split("×")[0] ?? r.size),
      series: [
        { name: "CPU", color: th.cool, values: mm.map((r) => r.cpu) },
        { name: "WebGPU", color: th.ember, values: mm.map((r) => r.gpu) },
      ],
    });
    chartB.setEmpty(mm.length ? null : "No results yet", "Run the benchmark to compare GFLOPS across sizes.");
    tbody.replaceChildren(
      ...rs.map((r) => {
        const speed = r.cpu !== null && r.gpu !== null && r.cpu > 0 ? r.gpu / r.cpu : null;
        return h(
          "tr",
          null,
          h("td", { class: "op" }, r.kernel),
          h("td", { class: "dim" }, r.size),
          h("td", { class: "num" }, r.cpu === null ? "—" : `${formatMetric(r.cpu, r.unit)} ${r.unit}`),
          h("td", { class: "num", style: r.gpu !== null ? "color:var(--ember-bright)" : "" }, r.gpu === null ? "—" : `${formatMetric(r.gpu, r.unit)} ${r.unit}`),
          h("td", { class: "num" }, speed === null ? "—" : `${speed.toFixed(1)}×`),
          h("td", { class: "num dim" }, `${fmtMs(r.cpuMs)} / ${fmtMs(r.gpuMs)}`),
        );
      }),
    );
    copyBtn.disabled = rs.length === 0;
  });

  scope.effect(() => {
    const p = progress();
    const on = running();
    prog.set(on ? p.frac : p.frac >= 1 ? 1 : 0);
    progLabel.textContent = on ? `Running… ${p.label}` : p.frac >= 1 ? "Done" : "";
    runBtn.disabled = on;
    stopBtn.style.display = on ? "" : "none";
    cpuCheck.disabled = on;
    cpuMaxSel.disabled = on;
  });

  async function start(): Promise<void> {
    if (running.peek()) return;
    running.set(true);
    rows.set([]);
    abort = new AbortController();
    try {
      await markBusy(
        runBench({
          gpu: backend,
          includeCpu,
          cpuMaxMatmul: cpuMax,
          signal: abort.signal,
          onProgress: (label, frac) => progress.set({ label, frac }),
          onRow: (row) => rows.update((r) => [...r, row]),
        }),
      );
    } catch (err: unknown) {
      toast(`Benchmark failed: ${err instanceof Error ? err.message : String(err)}`, "bad");
      console.error(err);
    } finally {
      running.set(false);
    }
  }

  return () => {
    abort?.abort();
    scope.dispose();
  };
}
