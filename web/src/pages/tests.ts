/** Tests — in-browser GPU ↔ CPU parity suite (`runParityAndPublish` → window.__kilnParityDone). */
import { runParityAndPublish, type ParityResult } from "../../../src/backend/webgpu/parity";
import { h, svg, icons, KILN_MARK } from "../lib/dom";
import { Scope, signal } from "../lib/signals";
import { adapterLabel, BROWSER_GUIDANCE, ensureGpu, gpuState, markBusy } from "../lib/gpu";
import { fmtDuration } from "../lib/format";
import { button, card, chip, notice, pageHead, progressBar } from "../components/ui";

export async function mount(root: HTMLElement): Promise<() => void> {
  const scope = new Scope();
  const results = signal<ParityResult[]>([]);
  const running = signal(false);
  const failuresOnly = signal(false);
  const elapsed = signal(0);
  let runToken = 0;

  const backend = await ensureGpu();
  const state = gpuState.peek();

  const rerun = button("Re-run suite", { icon: "refresh", variant: "primary", onClick: () => void run() });
  const filterBtn = button("Failures only", {
    size: "sm",
    onClick: () => {
      failuresOnly.update((v) => !v);
      filterBtn.setAttribute("aria-pressed", String(failuresOnly.peek()));
      filterBtn.classList.toggle("btn--primary", failuresOnly.peek());
    },
  });
  root.appendChild(
    pageHead(
      "GPU ↔ CPU parity",
      "Tests",
      "Every Backend kernel runs on both devices with random inputs (odd sizes, broadcasting, batches, transposes, duplicate ids) and the outputs are compared. Threshold 1e-4; 1e-3 for matmul with large K.",
      backend ? [filterBtn, rerun] : [],
    ),
  );

  if (!backend) {
    root.appendChild(unavailableState(state.apiPresent, state.error));
    // Still publish so the headless runner (scripts/gpu-parity.mjs) sees `unavailable: true`.
    void runParityAndPublish();
    return () => scope.dispose();
  }

  // ---- summary + progress
  const summary = h("div", { class: "tests__summary" });
  const prog = progressBar();
  const statusLine = h("div", { class: "status-line" });
  const summaryCard = card(null, [summary, h("div", { style: "margin-top:12px" }, prog.el), h("div", { style: "margin-top:8px" }, statusLine)]);
  summaryCard.classList.add("card--tight");
  root.appendChild(summaryCard);

  // ---- table
  const tbody = h("tbody");
  const tableCard = card(
    "Results",
    [
      h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          { class: "table" },
          h(
            "thead",
            null,
            h(
              "tr",
              null,
              h("th", null, "Op"),
              h("th", null, "Case"),
              h("th", { class: "num" }, "max |Δ|"),
              h("th", { class: "num" }, "max rel"),
              h("th", { class: "num" }, "threshold"),
              h("th", { class: "num" }, "GPU ms"),
              h("th", null, "Result"),
            ),
          ),
          tbody,
        ),
      ),
    ],
    { sub: "one row per case; hover a row for details" },
  );
  tableCard.style.marginTop = "16px";
  root.appendChild(tableCard);

  let rafId = 0;
  const scheduleRender = (): void => {
    if (rafId) return;
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      renderTable();
    });
  };

  function renderTable(): void {
    const rs = results.peek();
    const only = failuresOnly.peek();
    const shown = only ? rs.filter((r) => !r.pass) : rs;
    tbody.replaceChildren(
      ...shown.map((r) =>
        h(
          "tr",
          { class: r.pass ? "" : "is-fail", title: r.error ?? `${r.op} · ${r.case}` },
          h("td", { class: "op" }, r.op),
          h("td", { class: "dim" }, r.error ? `${r.case} — ${r.error}` : r.case),
          h("td", { class: "num" }, fmtErr(r.maxAbs)),
          h("td", { class: "num" }, fmtErr(r.maxRel)),
          h("td", { class: "num dim" }, r.threshold ? r.threshold.toExponential(0) : "—"),
          h("td", { class: "num dim" }, r.ms.toFixed(1)),
          h("td", null, h("span", { class: `pass-pill ${r.pass ? "pass-pill--ok" : "pass-pill--bad"}` }, svg(r.pass ? icons.check : icons.warn), r.pass ? "PASS" : "FAIL")),
        ),
      ),
    );
    if (!shown.length) tbody.appendChild(h("tr", null, h("td", { class: "dim" }, only ? "No failures." : "Waiting for results…")));
  }

  scope.effect(() => {
    const rs = results();
    const on = running();
    const fails = rs.filter((r) => !r.pass).length;
    const passes = rs.length - fails;
    summary.replaceChildren(
      h("span", { class: `pass-pill ${on ? "" : fails ? "pass-pill--bad" : "pass-pill--ok"}` }, on ? h("span", { class: "spinner" }) : svg(fails ? icons.warn : icons.check), on ? "RUNNING" : fails ? "FAILURES" : "ALL PASS"),
      chip("cases", rs.length),
      chip("passed", passes, "ok"),
      chip("failed", fails, fails ? "bad" : ""),
      chip("time", fmtDuration(elapsed())),
      chip("adapter", adapterLabel(state.info), "ember"),
    );
    prog.set(on ? null : 1);
    rerun.disabled = on;
    scheduleRender();
  });
  scope.effect(() => {
    failuresOnly();
    scheduleRender();
  });

  async function run(): Promise<void> {
    if (running.peek()) return;
    const my = ++runToken;
    running.set(true);
    results.set([]);
    const t0 = performance.now();
    statusLine.replaceChildren(h("span", { class: "status-dot status-dot--run" }), "Running parity cases…");
    const timer = setInterval(() => elapsed.set(performance.now() - t0), 200);
    try {
      const published = await markBusy(
        runParityAndPublish(backend ?? undefined, {
          onResult: (r) => {
            if (my !== runToken) return;
            results.update((rs) => [...rs, r]);
          },
        }),
      );
      if (my !== runToken) return;
      if (published.unavailable) {
        results.set(published.results);
      }
      elapsed.set(performance.now() - t0);
      statusLine.replaceChildren(
        h("span", { class: `status-dot ${published.pass ? "status-dot--ok" : "status-dot--bad"}` }),
        published.pass ? `All ${published.results.length} cases match the CPU backend.` : `${published.failures} of ${published.results.length} cases exceed their threshold.`,
        h("span", { class: "mono" }, "window.__kilnParityDone"),
      );
    } catch (err: unknown) {
      statusLine.replaceChildren(h("span", { class: "status-dot status-dot--bad" }), `Suite crashed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      clearInterval(timer);
      if (my === runToken) running.set(false);
    }
  }

  void run();

  return () => {
    runToken++;
    if (rafId) cancelAnimationFrame(rafId);
    scope.dispose();
  };
}

function fmtErr(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  return v.toExponential(2);
}

function unavailableState(apiPresent: boolean, error: string | null): HTMLElement {
  return card(null, [
    h(
      "div",
      { class: "unavailable" },
      svg(KILN_MARK, "hero-glyph"),
      h("h2", null, "WebGPU unavailable"),
      h(
        "p",
        { class: "prose", style: "margin:0" },
        apiPresent
          ? "navigator.gpu exists but no adapter could be created — the GPU driver may be blocked, or hardware acceleration is disabled in this browser."
          : "This browser does not expose navigator.gpu. The parity suite compares the WebGPU kernels against the CPU backend, so it needs a WebGPU-capable browser.",
      ),
      h(
        "div",
        { class: "browser-list" },
        ...BROWSER_GUIDANCE.map((b) => h("div", null, h("b", null, `${b.name} ${b.version}`), h("span", null, b.note))),
      ),
      notice("info", [
        "Headless CI uses Chromium with ",
        h("code", null, "--enable-unsafe-webgpu --use-angle=swiftshader"),
        " via ",
        h("code", null, "npm run test:gpu"),
        ". Everything else in Kiln (Studio on CPU, Playground, Inspect) still works without a GPU.",
      ]),
      error ? h("div", { class: "small mono" }, error) : null,
    ),
  ]);
}
