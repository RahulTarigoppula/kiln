/** Playground — generate from a checkpoint with per-token probability shading. */
import { h } from "../lib/dom";
import { Scope, signal } from "../lib/signals";
import { gpuState } from "../lib/gpu";
import { fmtCompact, fmtDuration, fmtPercent } from "../lib/format";
import { generateTokens, isEngineNotWired, type GenerateOptions, type TokenEvent } from "../lib/engine";
import { checkpoint, model, modelUnavailable } from "../lib/model-store";
import { checkpointPicker } from "../components/checkpoint-picker";
import { button, card, chip, field, notice, numberInput, pageHead, slider, toast } from "../components/ui";

export function mount(root: HTMLElement): () => void {
  const scope = new Scope();
  const temperature = signal(0.8);
  const topK = signal(40);
  const topP = signal(0.95);
  const maxTokens = signal(200);
  const seed = signal<number | null>(null);
  const generating = signal(false);
  const stats = signal<{ tokens: number; ms: number } | null>(null);
  let abort: AbortController | null = null;

  // ---- left column
  const ckCard = card("Checkpoint", [checkpointPicker(scope)], { class: "card--tight" });
  const tempS = slider("Temperature", temperature.peek(), { min: 0.1, max: 2, step: 0.05, format: (v) => v.toFixed(2), onChange: (v) => temperature.set(v) });
  const topKS = slider("Top-k", topK.peek(), { min: 0, max: 200, step: 1, format: (v) => (v === 0 ? "off" : String(v)), onChange: (v) => topK.set(v) });
  const topPS = slider("Top-p", topP.peek(), { min: 0.05, max: 1, step: 0.01, format: (v) => (v >= 1 ? "off" : v.toFixed(2)), onChange: (v) => topP.set(v) });
  const maxS = slider("Max new tokens", maxTokens.peek(), { min: 1, max: 1000, step: 1, onChange: (v) => maxTokens.set(v) });
  const seedIn = numberInput(0, { min: 0, step: 1, onChange: (v) => seed.set(v) });
  seedIn.placeholder = "random";
  seedIn.value = "";
  const samplingCard = card("Sampling", [h("div", { class: "stack stack--sm" }, tempS.el, topKS.el, topPS.el, maxS.el, field("Seed", seedIn, { hint: "Leave empty for a fresh seed each run." }))], { class: "card--tight" });

  // ---- right column
  const prompt = h("textarea", { class: "textarea input--mono", rows: 4, placeholder: "ROMEO:\nBut soft, what light through yonder window breaks?" }, "ROMEO:\n");
  const genBtn = button("Generate", { icon: "flame", variant: "primary", onClick: () => void run() });
  const stopBtn = button("Stop", { icon: "stop", variant: "danger", onClick: () => abort?.abort() });
  stopBtn.style.display = "none";
  const clearBtn = button("Clear", { variant: "ghost", onClick: () => reset() });
  const promptEl = h("span", { class: "tokens__prompt" });
  const outEl = h("span");
  const caret = h("span", { class: "caret" });
  caret.style.display = "none";
  const tokensBox = h("div", { class: "tokens", "aria-live": "polite" }, promptEl, outEl, caret);
  const statsRow = h("div", { class: "row" });
  const noticeHost = h("div");
  const outputCard = card(
    "Output",
    [
      noticeHost,
      tokensBox,
      h(
        "div",
        { class: "row row--between", style: "margin-top:12px" },
        h("div", { class: "prob-scale" }, "p(token)", h("span", { class: "prob-scale__bar" }), "low → high · hover a token for the top-5 alternatives"),
        statsRow,
      ),
    ],
    { sub: "each token shaded by its sampling probability" },
  );
  const promptCard = card("Prompt", [field("Prompt", prompt), h("div", { class: "row", style: "margin-top:12px" }, genBtn, stopBtn, clearBtn)], { class: "card--tight" });

  root.appendChild(pageHead("Generate", "Playground", "Sample from the shipped Shakespeare model or any Kiln checkpoint. Tokens stream in one at a time, shaded by how confident the model was.", []));
  root.appendChild(h("div", { class: "playground" }, h("aside", { class: "stack" }, ckCard, samplingCard), h("div", { class: "stack" }, promptCard, outputCard)));

  // ---- token tooltip
  const tip = h("div", { class: "tooltip" });
  document.body.appendChild(tip);
  scope.add(() => tip.remove());
  const tokenData = new WeakMap<HTMLElement, TokenEvent>();
  scope.listen(tokensBox, "mouseover", (ev) => {
    const target = (ev.target as HTMLElement).closest(".tok") as HTMLElement | null;
    const data = target ? tokenData.get(target) : undefined;
    if (!target || !data) {
      tip.classList.remove("is-on");
      return;
    }
    tip.replaceChildren(
      h("div", { style: "color:var(--ash-3);margin-bottom:4px" }, `sampled ${JSON.stringify(data.text)} · p = ${fmtPercent(data.prob)}`),
      ...data.top.slice(0, 5).flatMap((alt) => [
        h("div", { class: "tooltip__row" }, h("span", { style: alt.id === data.id ? "color:var(--ember-bright)" : "" }, JSON.stringify(alt.text)), h("b", null, fmtPercent(alt.prob))),
        h("div", { class: "tooltip__bar", style: `width:${Math.max(2, alt.prob * 100)}%` }),
      ]),
    );
    const r = target.getBoundingClientRect();
    tip.style.left = `${Math.min(window.innerWidth - 220, r.left)}px`;
    tip.style.top = `${r.bottom + 8}px`;
    tip.classList.add("is-on");
  });
  scope.listen(tokensBox, "mouseleave", () => tip.classList.remove("is-on"));

  scope.effect(() => {
    const on = generating();
    genBtn.disabled = on;
    stopBtn.style.display = on ? "" : "none";
    caret.style.display = on ? "" : "none";
    prompt.disabled = on;
  });
  scope.effect(() => {
    const s = stats();
    const ck = checkpoint();
    const chips: HTMLElement[] = [];
    if (ck) chips.push(chip("device", gpuState().backend ? "WebGPU" : "CPU", "ember"));
    if (s) chips.push(chip("tokens", s.tokens), chip("time", fmtDuration(s.ms)));
    if (s && s.ms > 0) chips.push(chip("tok/s", fmtCompact((s.tokens / s.ms) * 1000, 1)));
    statsRow.replaceChildren(...chips);
  });
  scope.effect(() => {
    const ck = checkpoint();
    const m = model();
    const un = modelUnavailable();
    noticeHost.replaceChildren();
    noticeHost.style.marginBottom = "12px";
    if (!ck) noticeHost.appendChild(notice("info", ["Load a checkpoint to start generating. The shipped ", h("code", null, "shakespeare-tiny"), " model is a 4-layer char-level GPT trained on TinyShakespeare."]));
    else if (!m && un) noticeHost.appendChild(notice("ember", [h("b", null, "Generation needs the phase-2 model engine. "), "The checkpoint is parsed (", String(ck.records.length), " tensors); wire ", h("code", null, "instantiateModel()"), " / ", h("code", null, "generateTokens()"), " in ", h("code", null, "web/src/lib/engine.ts"), " and this page streams tokens."]));
    else noticeHost.style.marginBottom = "0";
  });

  function reset(): void {
    outEl.replaceChildren();
    promptEl.textContent = "";
    stats.set(null);
  }

  async function run(): Promise<void> {
    const m = model.peek();
    const ck = checkpoint.peek();
    if (!ck) {
      toast("Load a checkpoint first", "bad");
      return;
    }
    if (!m) {
      toast(modelUnavailable.peek() ? "Model engine not wired yet (phase 2)" : "Model not ready", "bad");
      return;
    }
    reset();
    promptEl.textContent = prompt.value;
    generating.set(true);
    abort = new AbortController();
    const opts: GenerateOptions = { maxNewTokens: maxTokens.peek(), temperature: temperature.peek(), topK: topK.peek(), topP: topP.peek() };
    const sd = seed.peek();
    if (sd !== null && seedIn.value !== "") opts.seed = sd;
    const t0 = performance.now();
    let n = 0;
    try {
      const ids = m.tokenizer.encode(prompt.value);
      for await (const tok of generateTokens(m, ids, opts, abort.signal)) {
        if (abort.signal.aborted) break;
        const span = h("span", { class: "tok tok--new", style: `--p:${Math.min(1, Math.max(0, tok.prob)).toFixed(3)}` }, tok.text);
        tokenData.set(span, tok);
        outEl.appendChild(span);
        n++;
        stats.set({ tokens: n, ms: performance.now() - t0 });
      }
    } catch (err: unknown) {
      if (!abort.signal.aborted) toast(isEngineNotWired(err) ? "Model engine not wired yet (phase 2)" : `Generation failed: ${err instanceof Error ? err.message : String(err)}`, "bad");
    } finally {
      stats.set({ tokens: n, ms: performance.now() - t0 });
      generating.set(false);
    }
  }

  return () => {
    abort?.abort();
    scope.dispose();
  };
}
