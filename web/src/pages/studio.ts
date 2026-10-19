/** Studio - configure a dataset / tokenizer / GPT preset and train live in a Web Worker. */
import { h } from "../lib/dom";
import { batch, Scope, signal } from "../lib/signals";
import { LineChart, chartTheme } from "../lib/charts";
import { ensureGpu, gpuState, busyCount } from "../lib/gpu";
import { fmtBytes, fmtCompact, fmtDuration, fmtFixed, fmtInt, fmtMs, fmtSci } from "../lib/format";
import { estimateParams, estimateTrainingBytes, PRESET_NAMES, PRESETS, presetConfig, quickTextStats, type PresetName, type TokenizerKind, type TrainHyper } from "../lib/engine";
import { TrainSession } from "../lib/session";
import type { SessionState, StartJob, TrainEvent } from "../worker/protocol";
import type { GenerateOptions, SampleEvent } from "../lib/engine";
import { button, card, chartBox, chip, downloadBlob, field, fileDrop, notice, numberInput, pageHead, progressBar, segmented, slider, tile, toast } from "../components/ui";

type DataSource = "shakespeare" | "paste" | "upload";

const CONTEXT_OPTIONS = [32, 64, 128, 256, 512];

export async function mount(root: HTMLElement): Promise<() => void> {
  const scope = new Scope();
  await ensureGpu();
  const hasGpu = gpuState.peek().backend !== null;

  // ------------------------------------------------------------ state
  const source = signal<DataSource>("shakespeare");
  const text = signal("");
  const textName = signal("");
  const textLoading = signal(false);
  const tokKind = signal<TokenizerKind>("char");
  const bpeVocab = signal(1024);
  const preset = signal<PresetName>("micro");
  const context = signal(PRESETS.micro.blockSize);
  const batchSize = signal(PRESETS.micro.batchSize);
  const lr = signal(PRESETS.micro.lr);
  const maxSteps = signal(PRESETS.micro.maxSteps);
  const evalInterval = signal(100);
  const sampleInterval = signal(250);
  const preferGpu = signal<"webgpu" | "cpu">(hasGpu ? "webgpu" : "cpu");

  const sessionState = signal<SessionState>("idle");
  const sessionInfo = signal<{ device: string; adapter: string | null; inWorker: boolean } | null>(null);
  const status = signal<{ message: string; progress: number | null }>({ message: "Idle", progress: 0 });
  const lastError = signal<{ message: string; notWired: boolean } | null>(null);
  const datasetInfo = signal<{ tokens: number; vocabSize: number; trainTokens: number; valTokens: number; paramCount: number } | null>(null);
  const metrics = signal({ step: 0, loss: NaN, valLoss: NaN, lr: NaN, tps: NaN, msStep: NaN, eta: NaN, mem: NaN });
  const samples = signal<SampleEvent[]>([]);
  const checkpointReady = signal(false);
  const logScale = signal<"linear" | "log">("linear");
  const trainX: number[] = [];
  const trainY: number[] = [];
  const valX: number[] = [];
  const valY: number[] = [];

  // ------------------------------------------------------------ session
  let session: TrainSession | null = null;
  let genCounter = 0;
  let liveSample: { id: number; el: HTMLElement; text: string } | null = null;

  function ensureSession(): TrainSession {
    if (session) return session;
    session = new TrainSession(preferGpu.peek() === "webgpu");
    scope.add(session.on(onEvent));
    return session;
  }

  function onEvent(ev: TrainEvent): void {
    switch (ev.type) {
      case "ready":
        sessionInfo.set({ device: ev.device, adapter: ev.adapter, inWorker: ev.inWorker });
        break;
      case "status":
        status.set({ message: ev.message, progress: ev.progress ?? null });
        break;
      case "dataset":
        datasetInfo.set({ tokens: ev.tokens, vocabSize: ev.vocabSize, trainTokens: ev.trainTokens, valTokens: ev.valTokens, paramCount: ev.paramCount });
        break;
      case "state":
        sessionState.set(ev.state);
        if (ev.state === "running" || ev.state === "preparing") busyFlag(true);
        else busyFlag(false);
        if (ev.state === "done") status.set({ message: "Training complete", progress: 1 });
        if (ev.state === "stopped") status.set({ message: "Stopped", progress: 0 });
        if (ev.state === "error") status.set({ message: "Failed", progress: 0 });
        break;
      case "step": {
        const m = ev.metrics;
        trainX.push(m.step);
        trainY.push(m.loss);
        metrics.update((prev) => ({ ...prev, step: m.step, loss: m.loss, lr: m.lr, tps: m.tokensPerSec, msStep: m.msPerStep, eta: ev.etaMs ?? NaN, mem: ev.memBytes ?? NaN }));
        // A late step event (flushed after Stop) must not overwrite the "Stopped" status line.
        if (sessionState.peek() === "running" || sessionState.peek() === "paused") {
          status.set({ message: `Training · step ${m.step} / ${maxSteps.peek()}`, progress: m.step / Math.max(1, maxSteps.peek()) });
        }
        scheduleChart();
        break;
      }
      case "eval":
        valX.push(ev.metrics.step);
        valY.push(ev.metrics.valLoss);
        metrics.update((prev) => ({ ...prev, valLoss: ev.metrics.valLoss }));
        scheduleChart();
        break;
      case "sample":
        samples.update((s) => [ev.sample, ...s].slice(0, 50));
        break;
      case "token":
        if (liveSample && liveSample.id === ev.id) {
          liveSample.text += ev.token.text;
          const body = liveSample.el.querySelector(".sample__text");
          if (body) body.textContent = liveSample.text;
        }
        break;
      case "generated":
        if (liveSample && liveSample.id === ev.id) {
          liveSample.el.querySelector(".caret")?.remove();
          liveSample = null;
        }
        break;
      case "checkpoint": {
        const meta = ev.meta;
        const base = `kiln-${preset.peek()}-step${metrics.peek().step || 0}`;
        downloadBlob(`${base}.safetensors`, ev.safetensors);
        setTimeout(() => downloadBlob(`${base}.json`, JSON.stringify(meta, null, 2), "application/json"), 300);
        toast("Checkpoint downloaded (.safetensors + .json)", "ok");
        break;
      }
      case "error":
        lastError.set({ message: ev.message, notWired: ev.notWired ?? false });
        if (!ev.notWired) toast(ev.message, "bad");
        break;
    }
  }

  let busyOn = false;
  function busyFlag(on: boolean): void {
    if (on === busyOn) return;
    busyOn = on;
    busyCount.update((n) => Math.max(0, n + (on ? 1 : -1)));
  }
  scope.add(() => busyFlag(false));

  // ------------------------------------------------------------ dataset card
  async function loadShakespeare(): Promise<void> {
    if (text.peek() && textName.peek() === "tinyshakespeare.txt") return;
    textLoading.set(true);
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}data/tinyshakespeare.txt`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const t = await res.text();
      batch(() => {
        text.set(t);
        textName.set("tinyshakespeare.txt");
      });
    } catch (err: unknown) {
      toast(`Could not fetch TinyShakespeare: ${err instanceof Error ? err.message : String(err)}`, "bad");
    } finally {
      textLoading.set(false);
    }
  }

  const pasteArea = h("textarea", { class: "textarea", placeholder: "Paste any text - a few hundred KB works well for char-level models.", rows: 7 });
  pasteArea.addEventListener("input", () => {
    text.set(pasteArea.value);
    textName.set("pasted text");
  });
  const upload = fileDrop("Drop a .txt / .md file or click to browse", ".txt,.md,.csv,.json,text/plain", async (files) => {
    const f = files[0];
    if (!f) return;
    textLoading.set(true);
    try {
      const t = await f.text();
      batch(() => {
        text.set(t);
        textName.set(f.name);
      });
    } finally {
      textLoading.set(false);
    }
  });
  const sourceBody = h("div");
  const dataStats = h("div", { class: "data-stats" });
  const sourceSeg = segmented<DataSource>(
    [
      { value: "shakespeare", label: "TinyShakespeare" },
      { value: "paste", label: "Paste" },
      { value: "upload", label: "Upload" },
    ],
    source,
  );
  scope.effect(() => {
    const s = source();
    sourceBody.replaceChildren();
    if (s === "shakespeare") {
      sourceBody.appendChild(h("div", { class: "small" }, "1.1 MB of Shakespeare's plays (public domain), fetched from ", h("code", null, "/kiln/data/tinyshakespeare.txt"), "."));
      void loadShakespeare();
    } else if (s === "paste") {
      sourceBody.appendChild(pasteArea);
      if (textName.peek() !== "pasted text") {
        text.set(pasteArea.value);
        textName.set("pasted text");
      }
    } else {
      sourceBody.appendChild(upload);
      if (textName.peek() === "tinyshakespeare.txt" || textName.peek() === "pasted text") {
        text.set("");
        textName.set("");
      }
    }
  });
  scope.effect(() => {
    const t = text();
    const stats = quickTextStats(t, { kind: tokKind(), vocabSize: bpeVocab() });
    const loadingNow = textLoading();
    dataStats.replaceChildren(
      stat(loadingNow ? "..." : fmtCompact(stats.chars), "characters"),
      stat(loadingNow ? "..." : String(stats.charVocab), "distinct chars"),
      stat(loadingNow ? "..." : `~ ${fmtCompact(stats.estTokens)}`, "tokens"),
    );
  });
  const datasetCard = card("Dataset", [sourceSeg, sourceBody, dataStats], { class: "card--tight" });
  datasetCard.querySelector(".card__head")?.insertAdjacentElement("afterend", h("div", { class: "stack stack--sm" }));
  (datasetCard as HTMLElement).classList.add("stack");

  // ------------------------------------------------------------ tokenizer card
  const vocabSlider = slider("BPE vocabulary", bpeVocab.peek(), { min: 512, max: 4096, step: 256, onChange: (v) => bpeVocab.set(v) });
  const tokSeg = segmented<TokenizerKind>(
    [
      { value: "char", label: "Character" },
      { value: "bpe", label: "Byte-level BPE" },
    ],
    tokKind,
  );
  const tokHint = h("div", { class: "field__hint" });
  scope.effect(() => {
    const k = tokKind();
    vocabSlider.el.style.display = k === "bpe" ? "" : "none";
    tokHint.textContent = k === "char" ? "One token per distinct character - the classic nanoGPT Shakespeare setup. Vocab comes from the text." : "GPT-2-style merges trained on your text inside the worker before training starts (a few seconds for 1 MB).";
  });
  const tokenizerCard = card("Tokenizer", [h("div", { class: "stack stack--sm" }, tokSeg, vocabSlider.el, tokHint)], { class: "card--tight" });

  // ------------------------------------------------------------ model card
  const presetBtns = PRESET_NAMES.map((name) =>
    h(
      "button",
      {
        class: "preset",
        type: "button",
        "aria-pressed": String(preset.peek() === name),
        onclick: () => {
          const p = PRESETS[name];
          batch(() => {
            preset.set(name);
            context.set(p.blockSize);
            batchSize.set(p.batchSize);
            lr.set(p.lr);
            maxSteps.set(p.maxSteps);
          });
        },
      },
      h("div", { class: "preset__name" }, name),
      h("div", { class: "preset__desc" }, PRESETS[name].blurb),
    ),
  );
  const contextSel = h(
    "select",
    { class: "select", onchange: () => context.set(Number(contextSel.value)) },
    ...CONTEXT_OPTIONS.map((n) => h("option", { value: String(n) }, `${n} tokens`)),
  );
  const paramEl = h("span", { class: "param-count__n" }, "-");
  const paramSub = h("span", { class: "param-count__l" }, "parameters");
  const memHint = h("div", { class: "field__hint" });
  scope.effect(() => {
    const p = preset();
    for (const b of presetBtns) b.setAttribute("aria-pressed", String(b.querySelector(".preset__name")?.textContent === p));
    contextSel.value = String(context());
    const stats = quickTextStats(text(), { kind: tokKind(), vocabSize: bpeVocab() });
    const vocab = tokKind() === "char" ? Math.max(stats.charVocab, 1) : bpeVocab();
    const cfg = presetConfig(PRESETS[p], vocab, { blockSize: context() });
    const n = estimateParams(cfg);
    paramEl.textContent = fmtCompact(n, 2);
    paramSub.textContent = `params · vocab ${vocab} · 12·L·d² + (V+T)·d`;
    memHint.textContent = `~ ${fmtBytes(estimateTrainingBytes(cfg, batchSize()))} training memory at batch ${batchSize()} (f32 params, grads, Adam moments, activations).`;
  });
  const modelCard = card(
    "Model",
    [
      h("div", { class: "stack stack--sm" }, h("div", { class: "preset-grid" }, ...presetBtns), h("div", { class: "param-count" }, paramEl, paramSub), field("Context (block size)", contextSel), memHint),
    ],
    { class: "card--tight", sub: "GPT, weight-tied, nanoGPT naming" },
  );

  // ------------------------------------------------------------ training card
  const batchIn = numberInput(batchSize.peek(), { min: 1, max: 256, step: 1, onChange: (v) => batchSize.set(Math.max(1, Math.round(v))) });
  const lrIn = numberInput(lr.peek(), { min: 0, max: 1, step: 0.0001, onChange: (v) => lr.set(v) });
  const stepsIn = numberInput(maxSteps.peek(), { min: 1, max: 1_000_000, step: 100, onChange: (v) => maxSteps.set(Math.max(1, Math.round(v))) });
  const evalIn = numberInput(evalInterval.peek(), { min: 10, max: 10_000, step: 10, onChange: (v) => evalInterval.set(Math.max(1, Math.round(v))) });
  const sampleIn = numberInput(sampleInterval.peek(), { min: 10, max: 10_000, step: 10, onChange: (v) => sampleInterval.set(Math.max(1, Math.round(v))) });
  scope.effect(() => {
    batchIn.value = String(batchSize());
    lrIn.value = String(lr());
    stepsIn.value = String(maxSteps());
  });
  const deviceSeg = segmented<"webgpu" | "cpu">(
    [
      { value: "webgpu", label: "WebGPU" },
      { value: "cpu", label: "CPU" },
    ],
    preferGpu,
    () => {
      if (session && sessionState.peek() !== "running" && sessionState.peek() !== "preparing") {
        session.dispose();
        session = null;
        sessionInfo.set(null);
      }
    },
  );
  if (!hasGpu) (deviceSeg.querySelector("[data-v=webgpu]") as HTMLButtonElement | null)?.setAttribute("disabled", "");
  const trainingCard = card(
    "Training",
    [
      h(
        "div",
        { class: "stack stack--sm" },
        h("div", { class: "grid grid--2" }, field("Batch size", batchIn), field("Learning rate", lrIn)),
        h("div", { class: "grid grid--2" }, field("Max steps", stepsIn), field("Eval every", evalIn)),
        h("div", { class: "grid grid--2" }, field("Sample every", sampleIn), field("Device", deviceSeg)),
        h("div", { class: "field__hint" }, "AdamW · cosine LR with 5% warmup · weight decay 0.1 · grad-clip 1.0 · 90/10 train/val split."),
      ),
    ],
    { class: "card--tight" },
  );

  // ------------------------------------------------------------ controls
  const startBtn = button("Start", { icon: "play", variant: "primary", onClick: start });
  const pauseBtn = button("Pause", { icon: "pause", onClick: () => session?.send({ type: "pause" }) });
  const resumeBtn = button("Resume", { icon: "play", onClick: () => session?.send({ type: "resume" }) });
  const stopBtn = button("Stop", { icon: "stop", variant: "danger", onClick: () => session?.send({ type: "stop" }) });
  const dlBtn = button("Download checkpoint", { icon: "download", onClick: () => session?.send({ type: "export" }) });
  const prog = progressBar();
  const statusText = h("span");
  const statusMono = h("span", { class: "mono" });
  const errorHost = h("div");
  const controlsCard = card(
    null,
    [
      h("div", { class: "studio__controls" }, startBtn, pauseBtn, resumeBtn, stopBtn, dlBtn),
      h("div", { style: "margin-top:14px" }, prog.el),
      h("div", { class: "status-line", style: "margin-top:8px" }, h("span", { class: "status-dot", id: "studio-dot" }), statusText, h("span", { class: "spacer" }), statusMono),
      errorHost,
    ],
    { class: "card--tight" },
  );
  const dot = controlsCard.querySelector("#studio-dot") as HTMLElement;

  scope.effect(() => {
    const st = sessionState();
    const s = status();
    const info = sessionInfo();
    const running = st === "running";
    const paused = st === "paused";
    const preparing = st === "preparing";
    const haveText = text().length > 0;
    startBtn.disabled = running || paused || preparing || !haveText;
    startBtn.replaceChildren(startBtn.firstChild as Node, st === "stopped" || st === "done" || st === "error" ? "Restart" : "Start");
    pauseBtn.style.display = running ? "" : "none";
    resumeBtn.style.display = paused ? "" : "none";
    stopBtn.disabled = !(running || paused || preparing);
    dlBtn.disabled = !(checkpointReady() || running || paused || st === "done" || st === "stopped");
    prog.set(preparing && s.progress === null ? null : (s.progress ?? 0));
    statusText.textContent = s.message;
    dot.className = `status-dot ${running || preparing ? "status-dot--run" : st === "error" ? "status-dot--bad" : st === "done" ? "status-dot--ok" : ""}`;
    statusMono.textContent = info ? `${info.inWorker ? "worker" : "main thread"} · ${info.device === "webgpu" ? info.adapter ?? "WebGPU" : "CPU"}` : "";
  });
  scope.effect(() => {
    const err = lastError();
    errorHost.replaceChildren();
    if (!err) return;
    errorHost.style.marginTop = "12px";
    errorHost.appendChild(notice("bad", [err.message]));
  });

  // ------------------------------------------------------------ metric tiles
  const tStep = tile("Step");
  const tLoss = tile("Train loss", "-", { ember: true });
  const tVal = tile("Val loss");
  const tLr = tile("Learning rate");
  const tTps = tile("Tokens / s");
  const tMs = tile("ms / step");
  const tEta = tile("ETA");
  const tMem = tile("GPU memory");
  scope.effect(() => {
    const m = metrics();
    tStep.set(fmtInt(m.step), `of ${fmtInt(maxSteps())}`);
    tLoss.set(fmtFixed(m.loss, 4));
    tVal.set(fmtFixed(m.valLoss, 4));
    tLr.set(fmtSci(m.lr, 2));
    tTps.set(Number.isFinite(m.tps) ? fmtCompact(m.tps, 1) : "-");
    tMs.set(fmtMs(m.msStep));
    tEta.set(Number.isFinite(m.eta) ? fmtDuration(m.eta) : "-");
    tMem.set(Number.isFinite(m.mem) ? fmtBytes(m.mem) : "-", sessionInfo()?.device === "cpu" ? "cpu backend" : "live buffers");
  });
  const tiles = h("div", { class: "tiles" }, tStep.el, tLoss.el, tVal.el, tLr.el, tTps.el, tMs.el, tEta.el, tMem.el);

  // ------------------------------------------------------------ loss chart
  const chartB = chartBox("chart--tall");
  chartB.setEmpty("Loss curve", "Train loss streams in per step; val loss every eval interval.");
  const scaleSeg = segmented<"linear" | "log">(
    [
      { value: "linear", label: "linear" },
      { value: "log", label: "log" },
    ],
    logScale,
  );
  const th = chartTheme();
  const legend = h(
    "div",
    { class: "legend" },
    h("span", { class: "legend__item" }, h("span", { class: "legend__swatch", style: `--sw:${th.ember}` }), "train"),
    h("span", { class: "legend__item" }, h("span", { class: "legend__swatch", style: `--sw:${th.cool}` }), "val"),
  );
  const chartCard = card("Loss", [chartB.wrap], { sub: "cross-entropy, nats", actions: [legend, scaleSeg] });
  const chart = new LineChart(chartB.wrap, { xLabel: "step", yLabel: "loss", yFormat: (v) => v.toFixed(3), xMin: 0 });
  scope.add(() => chart.destroy());
  let chartRaf = 0;
  function scheduleChart(): void {
    if (chartRaf) return;
    chartRaf = requestAnimationFrame(() => {
      chartRaf = 0;
      chart.setSeries([
        { name: "train", color: th.ember, xs: trainX, ys: trainY, glow: true },
        { name: "val", color: th.cool, xs: valX, ys: valY, points: true, glow: true },
      ]);
      chartB.setEmpty(trainX.length ? null : "Loss curve");
    });
  }
  scope.effect(() => {
    chart.setOptions({ logY: logScale() === "log" });
  });

  // ------------------------------------------------------------ samples panel
  const samplePrompt = h("input", { class: "input input--mono", placeholder: "Prompt (optional) - e.g. ROMEO:", value: "" });
  const sampleBtn = button("Sample now", {
    icon: "flame",
    size: "sm",
    onClick: () => {
      const s = ensureSession();
      const id = ++genCounter;
      const opts: GenerateOptions = { maxNewTokens: 200, temperature: 0.9, topK: 40, topP: 1 };
      const el = sampleCard({ step: metrics.peek().step, text: samplePrompt.value });
      el.querySelector(".sample__text")?.appendChild(h("span", { class: "caret" }));
      liveSample = { id, el, text: samplePrompt.value };
      samplesHost.prepend(el);
      s.send({ type: "generate", id, prompt: samplePrompt.value, opts });
    },
  });
  const samplesHost = h("div", { class: "samples" });
  scope.effect(() => {
    const list = samples();
    const st = sessionState();
    sampleBtn.disabled = !(st === "running" || st === "paused" || st === "done" || st === "stopped") || !!liveSample;
    const live = liveSample?.el;
    samplesHost.replaceChildren(...(live ? [live] : []), ...list.map(sampleCard));
    if (!list.length && !live) samplesHost.appendChild(h("div", { class: "empty" }, h("b", null, "No samples yet"), "Generated text appears here every sample interval."));
  });
  const samplesCard = card("Samples", [h("div", { class: "row", style: "margin-bottom:12px" }, samplePrompt, sampleBtn), samplesHost], { sub: "streams from the in-training model" });
  samplePrompt.style.flex = "1";

  // ------------------------------------------------------------ run card
  const runInfo = h("div");
  scope.effect(() => {
    const ds = datasetInfo();
    const info = sessionInfo();
    runInfo.replaceChildren(
      h(
        "div",
        { class: "row" },
        chip("device", info ? (info.device === "webgpu" ? "WebGPU" : "CPU") : preferGpu() === "webgpu" ? "WebGPU" : "CPU", "ember"),
        chip("thread", info ? (info.inWorker ? "worker" : "main") : "-"),
        chip("preset", preset()),
        ds ? chip("tokens", fmtCompact(ds.tokens)) : null,
        ds ? chip("vocab", ds.vocabSize) : null,
        ds ? chip("params", fmtCompact(ds.paramCount)) : null,
      ),
    );
  });
  const runCard = card("Run", [runInfo, h("div", { class: "field__hint", style: "margin-top:12px" }, "Training runs in a Web Worker so the page never stalls; it falls back to the main thread when the worker has no WebGPU. Checkpoints download as safetensors + a JSON sidecar (config, tokenizer, training log).")], { class: "card--tight" });

  // ------------------------------------------------------------ layout
  root.appendChild(pageHead("Train live", "Studio", "Pick a corpus, a tokenizer and a GPT preset, then press Start and watch the loss fall on your GPU. Everything here - kernels, autograd, optimiser - is Kiln's own TypeScript, and it all runs inside this tab: nothing you paste or upload leaves your device.", []));
  if (!hasGpu) {
    root.appendChild(
      notice("warn", [
        h("b", null, "WebGPU is not available in this browser, so training will run on a single CPU thread (slow). "),
        "Training needs WebGPU: Chrome or Edge 113+, Safari 26+, or Firefox 141+ on Windows. The nano preset still trains on the CPU if you want to try it; the Playground and Inspect pages work fully without a GPU.",
      ]),
    );
  }
  root.appendChild(
    h(
      "div",
      { class: "studio" },
      h("aside", { class: "studio__config" }, datasetCard, tokenizerCard, modelCard, trainingCard),
      h("div", { class: "studio__main" }, controlsCard, tiles, chartCard, h("div", { class: "studio__lower" }, samplesCard, runCard)),
    ),
  );

  // ------------------------------------------------------------ actions
  function start(): void {
    const t = text.peek();
    if (!t) {
      toast("Load or paste some text first", "bad");
      return;
    }
    const s = ensureSession();
    const p = PRESETS[preset.peek()];
    const hyper: TrainHyper = {
      batchSize: batchSize.peek(),
      lr: lr.peek(),
      maxSteps: maxSteps.peek(),
      warmupSteps: Math.max(10, Math.round(maxSteps.peek() * 0.05)),
      weightDecay: 0.1,
      gradClip: 1.0,
      evalInterval: evalInterval.peek(),
      evalIters: 10,
      sampleInterval: sampleInterval.peek(),
      sampleTokens: 200,
      seed: 1337,
    };
    const job: StartJob = {
      text: t,
      tokenizer: { kind: tokKind.peek(), vocabSize: bpeVocab.peek() },
      model: { blockSize: context.peek(), nLayer: p.nLayer, nHead: p.nHead, nEmbd: p.nEmbd, dropout: 0.1, bias: true },
      hyper,
      preferGpu: preferGpu.peek() === "webgpu",
    };
    trainX.length = 0;
    trainY.length = 0;
    valX.length = 0;
    valY.length = 0;
    batch(() => {
      lastError.set(null);
      datasetInfo.set(null);
      samples.set([]);
      checkpointReady.set(false);
      metrics.set({ step: 0, loss: NaN, valLoss: NaN, lr: NaN, tps: NaN, msStep: NaN, eta: NaN, mem: NaN });
      status.set({ message: "Preparing...", progress: null });
      sessionState.set("preparing");
    });
    scheduleChart();
    s.send({ type: "start", job });
  }

  return () => {
    if (chartRaf) cancelAnimationFrame(chartRaf);
    session?.dispose();
    session = null;
    scope.dispose();
  };
}

function stat(v: string, l: string): HTMLElement {
  return h("div", { class: "data-stat" }, h("div", { class: "data-stat__v" }, v), h("div", { class: "data-stat__l" }, l));
}

function sampleCard(s: SampleEvent): HTMLElement {
  return h("div", { class: "sample" }, h("div", { class: "sample__meta" }, `step ${s.step}`), h("div", { class: "sample__text" }, s.text));
}
