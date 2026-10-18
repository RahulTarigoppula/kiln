/** Inspect — attention heatmaps, token-embedding PCA, weight histograms for the loaded checkpoint. */
import { h } from "../lib/dom";
import { Scope, signal } from "../lib/signals";
import { BarChart, Heatmap, Scatter, chartTheme, histogram } from "../lib/charts";
import { fmtCompact, fmtPercent, fmtTick, visibleToken } from "../lib/format";
import { attentionMaps, findEmbeddingRecord, isEngineNotWired, type AttentionMaps, type TensorRecord } from "../lib/engine";
import { pca } from "../lib/pca";
import { checkpoint, model, modelUnavailable } from "../lib/model-store";
import { checkpointPicker } from "../components/checkpoint-picker";
import { button, card, chartBox, chip, field, notice, pageHead, select, tile } from "../components/ui";

export function mount(root: HTMLElement): () => void {
  const scope = new Scope();
  const layer = signal(0);
  const head = signal(0);
  const paramName = signal<string>("");
  const attn = signal<AttentionMaps | null>(null);
  const attnError = signal<string | null>(null);
  const tokens = signal<string[]>([]);

  // ---- left column
  const ckCard = card("Checkpoint", [checkpointPicker(scope)], { class: "card--tight" });
  const promptIn = h("input", { class: "input input--mono", value: "The king is dead, long live the", placeholder: "Prompt tokens for the attention map" });
  const layerSel = h("select", { class: "select", onchange: () => layer.set(Number(layerSel.value)) });
  const headSel = h("select", { class: "select", onchange: () => head.set(Number(headSel.value)) });
  const computeBtn = button("Compute attention", { icon: "zap", variant: "primary", size: "sm", onClick: () => void computeAttention() });
  const attnCard = card("Attention", [h("div", { class: "stack stack--sm" }, field("Prompt", promptIn), h("div", { class: "grid grid--2" }, field("Layer", layerSel), field("Head", headSel)), computeBtn)], { class: "card--tight" });
  const paramSel = h("select", { class: "select", onchange: () => paramName.set(paramSel.value) });
  const paramCard = card("Weights", [field("Parameter", paramSel, { hint: "Histogram of the selected tensor (64 bins)." })], { class: "card--tight" });

  // ---- charts
  const heatBox = chartBox("chart--square");
  const heatCard = card("Attention heatmap", [heatBox.wrap], { sub: "rows = query token, columns = key token" });
  const heat = new Heatmap(heatBox.wrap);
  scope.add(() => heat.destroy());

  const pcaBox = chartBox("chart--square");
  const pcaChips = h("div", { class: "row" });
  const pcaCard = card("Token embeddings · PCA", [pcaBox.wrap, h("div", { style: "margin-top:10px" }, pcaChips)], { sub: "power-iteration PCA of the embedding table, in-browser" });
  const scatter = new Scatter(pcaBox.wrap, { xLabel: "pc1", yLabel: "pc2", labelLimit: 140 });
  scope.add(() => scatter.destroy());

  const histBox = chartBox();
  const histStats = h("div", { class: "hist-stats" });
  const histCard = card("Weight histogram", [histBox.wrap, histStats], { sub: "distribution of parameter values" });
  const hist = new BarChart(histBox.wrap, { categories: [], series: [], yLabel: "count", valueFormat: (v) => fmtCompact(v, 1) });
  scope.add(() => hist.destroy());

  root.appendChild(pageHead("Look inside", "Inspect", "Attention patterns per layer and head for a prompt, a 2-D projection of the token embeddings, and the shape of every weight tensor.", []));
  root.appendChild(
    h(
      "div",
      { class: "inspect" },
      h("aside", { class: "stack" }, ckCard, attnCard, paramCard),
      h("div", { class: "stack" }, h("div", { class: "inspect__charts" }, heatCard, pcaCard), histCard),
    ),
  );

  // ---- react to checkpoint
  scope.effect(() => {
    const ck = checkpoint();
    const c = ck?.config ?? null;
    layerSel.replaceChildren(...Array.from({ length: c?.nLayer ?? 1 }, (_, i) => h("option", { value: String(i) }, `layer ${i}`)));
    headSel.replaceChildren(...Array.from({ length: c?.nHead ?? 1 }, (_, i) => h("option", { value: String(i) }, `head ${i}`)));
    layer.set(0);
    head.set(0);
    layerSel.disabled = !c;
    headSel.disabled = !c;
    computeBtn.disabled = !ck;
    paramSel.replaceChildren(...(ck?.records ?? []).map((r) => h("option", { value: r.name }, `${r.name}  [${r.shape.join("×")}]`)));
    paramSel.disabled = !ck;
    const emb = ck ? findEmbeddingRecord(ck) : null;
    paramName.set(emb?.name ?? ck?.records[0]?.name ?? "");
    if (paramName.peek()) paramSel.value = paramName.peek();
    attn.set(null);
    attnError.set(null);
    updateTokens();
    renderPca(ck ? emb : null);
  });
  scope.listen(promptIn, "input", () => {
    updateTokens();
    attn.set(null);
  });

  function updateTokens(): void {
    const ck = checkpoint.peek();
    const tk = ck?.tokenizer;
    const T = ck?.config?.blockSize ?? 64;
    if (!tk) {
      tokens.set([]);
      return;
    }
    const ids = Array.from(tk.encode(promptIn.value)).slice(-T);
    tokens.set(ids.map((id) => visibleToken(tk.decode([id]))));
  }

  scope.effect(() => {
    const t = tokens();
    const a = attn();
    const err = attnError();
    const ck = checkpoint();
    const un = modelUnavailable();
    if (!ck) {
      heat.setData(null);
      heatBox.setEmpty("No checkpoint", "Load one to visualise attention.");
      return;
    }
    if (!t.length) {
      heat.setData(null);
      heatBox.setEmpty("Empty prompt", "Type a prompt to tokenise.");
      return;
    }
    if (a) {
      const L = Math.min(layer(), a.nLayer - 1);
      const H = Math.min(head(), a.nHead - 1);
      const w = a.weights[L];
      if (!w) return;
      const T = a.T;
      const slice = w.subarray(H * T * T, (H + 1) * T * T);
      heat.setData({ values: slice, rows: T, cols: T, xLabels: t, yLabels: t, vmin: 0, vmax: 1, xTitle: "key", yTitle: "query" });
      heatBox.setEmpty(null);
      return;
    }
    // Axis preview without weights.
    heat.setData({ values: new Float32Array(t.length * t.length), rows: t.length, cols: t.length, xLabels: t, yLabels: t, vmin: 0, vmax: 1, xTitle: "key", yTitle: "query" });
    heatBox.setEmpty(err ? "Attention unavailable" : `${t.length} tokens`, err ? (un ? "Needs the phase-2 model engine (attentionMaps in web/src/lib/engine.ts)." : err) : "Press “Compute attention” to run a forward pass with attention capture.");
  });

  async function computeAttention(): Promise<void> {
    const m = model.peek();
    const ck = checkpoint.peek();
    if (!ck?.tokenizer) return;
    if (!m) {
      attnError.set(modelUnavailable.peek() ?? "Model not instantiated");
      attn.set(null);
      return;
    }
    computeBtn.disabled = true;
    try {
      const ids = ck.tokenizer.encode(promptIn.value);
      const T = ck.config?.blockSize ?? ids.length;
      const trimmed = ids.length > T ? ids.slice(ids.length - T) : ids;
      attn.set(await attentionMaps(m, trimmed));
      attnError.set(null);
    } catch (err: unknown) {
      attnError.set(isEngineNotWired(err) ? err.message : err instanceof Error ? err.message : String(err));
      attn.set(null);
    } finally {
      computeBtn.disabled = false;
    }
  }

  // ---- PCA
  function renderPca(rec: TensorRecord | null): void {
    const ck = checkpoint.peek();
    pcaChips.replaceChildren();
    if (!ck || !rec || rec.shape.length !== 2 || !(rec.data instanceof Float32Array)) {
      scatter.setPoints([]);
      pcaBox.setEmpty("No embedding table", ck ? "This checkpoint has no recognisable 2-D embedding tensor." : "Load a checkpoint to project its token embeddings.");
      return;
    }
    const rows = rec.shape[0] ?? 0;
    const cols = rec.shape[1] ?? 0;
    const maxRows = 2048;
    const useRows = Math.min(rows, maxRows);
    const res = pca(rec.data.subarray(0, useRows * cols), useRows, cols, { components: 2 });
    const tk = ck.tokenizer;
    const norms = new Float32Array(useRows);
    let maxNorm = 0;
    for (let r = 0; r < useRows; r++) {
      let s = 0;
      for (let c = 0; c < cols; c++) {
        const v = rec.data[r * cols + c] ?? 0;
        s += v * v;
      }
      norms[r] = Math.sqrt(s);
      if ((norms[r] ?? 0) > maxNorm) maxNorm = norms[r] ?? 0;
    }
    scatter.setPoints(
      Array.from({ length: useRows }, (_, i) => ({
        x: res.projected[i * 2] ?? 0,
        y: res.projected[i * 2 + 1] ?? 0,
        label: tk ? visibleToken(tk.decode([i])) : String(i),
        weight: maxNorm > 0 ? (norms[i] ?? 0) / maxNorm : 0.5,
      })),
    );
    pcaBox.setEmpty(null);
    pcaChips.append(chip("tensor", rec.name), chip("shape", rec.shape.join("×")), chip("pc1", fmtPercent(res.explained[0] ?? 0), "ember"), chip("pc2", fmtPercent(res.explained[1] ?? 0), "ember"));
    if (useRows < rows) pcaChips.append(chip("shown", `${useRows}/${rows} rows`, "warn"));
  }

  // ---- histogram
  scope.effect(() => {
    const ck = checkpoint();
    const name = paramName();
    const rec = ck?.records.find((r) => r.name === name) ?? null;
    histStats.replaceChildren();
    if (!rec) {
      hist.setData({ categories: [], series: [] });
      histBox.setEmpty("No tensor selected", ck ? "Pick a parameter on the left." : "Load a checkpoint to inspect its weights.");
      return;
    }
    const bins = 64;
    const hg = histogram(rec.data, bins);
    const labels = Array.from({ length: bins }, (_, i) => (i % 8 === 0 ? fmtTick(hg.edges[i] ?? 0) : ""));
    hist.setData({ categories: labels, series: [{ name: rec.name, color: chartTheme().ember, values: Array.from(hg.counts) }] });
    histBox.setEmpty(null);
    const n = rec.data.length;
    const tiles = [tile("count", fmtCompact(n)), tile("mean", hg.mean.toExponential(2)), tile("std", hg.std.toExponential(2)), tile("min", hg.min.toFixed(4)), tile("max", hg.max.toFixed(4))];
    histStats.append(...tiles.map((t) => t.el));
  });

  return () => scope.dispose();
}
