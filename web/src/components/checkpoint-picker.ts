/** Checkpoint picker shared by Playground and Inspect (shipped checkpoints or local files). */
import { h } from "../lib/dom";
import type { Scope } from "../lib/signals";
import { fmtBytes, fmtCompact } from "../lib/format";
import { estimateParams } from "../lib/engine";
import { checkpoint, DEFAULT_CHECKPOINT, loadError, loadFromFiles, loadShipped, loading, model, modelUnavailable, SHIPPED_CHECKPOINTS } from "../lib/model-store";
import { button, fileDrop, kv, notice } from "./ui";

export function checkpointPicker(scope: Scope): HTMLElement {
  let selected = DEFAULT_CHECKPOINT;
  const blurb = h("div", { class: "small", style: "font-family:var(--font-mono);font-size:11px" }, selected.blurb);
  const sel = h(
    "select",
    {
      class: "select",
      "aria-label": "Shipped checkpoint",
      onchange: () => {
        selected = SHIPPED_CHECKPOINTS.find((c) => c.id === sel.value) ?? DEFAULT_CHECKPOINT;
        blurb.textContent = selected.blurb;
      },
    },
    ...SHIPPED_CHECKPOINTS.map((c) => h("option", { value: c.id }, c.label)),
  );
  sel.value = selected.id;
  const loadBtn = button("Load", { icon: "download", variant: "primary", size: "sm", onClick: () => void loadShipped(selected) });
  const status = h("div", { class: "stack stack--sm" });
  const info = h("div");

  const el = h(
    "div",
    { class: "stack" },
    h("div", { class: "row", style: "gap:8px;align-items:center" }, sel, loadBtn),
    blurb,
    fileDrop("Drop a .safetensors + .json sidecar, or click to browse", ".safetensors,.json", (files) => void loadFromFiles(files), true),
    status,
    info,
  );
  sel.style.flex = "1";

  scope.effect(() => {
    const busy = loading();
    const err = loadError();
    const ck = checkpoint();
    const unavailable = modelUnavailable();
    const m = model();
    loadBtn.disabled = busy !== null;
    status.replaceChildren();
    if (busy) status.appendChild(h("div", { class: "status-line" }, h("span", { class: "spinner" }), busy));
    if (err) status.appendChild(notice("bad", [err]));
    if (ck && unavailable) status.appendChild(notice("warn", [h("b", null, "Checkpoint parsed, model not built. "), unavailable]));
    if (ck && m) status.appendChild(notice("ok", [`Model ready on ${m.device === "webgpu" ? "WebGPU" : "CPU"} · ${fmtCompact(m.paramCount)} params.`]));
    info.replaceChildren();
    if (ck) {
      const c = ck.config;
      const rows: [string, string][] = [
        ["source", ck.source.split("/").pop() ?? ck.source],
        ["tensors", `${ck.records.length} · ${fmtBytes(ck.byteLength)}`],
      ];
      if (c) {
        rows.push(["config", `L${c.nLayer} · H${c.nHead} · d${c.nEmbd} · T${c.blockSize}`]);
        rows.push(["vocab", String(c.vocabSize)]);
        rows.push(["params", `≈ ${fmtCompact(estimateParams(c))}`]);
      }
      if (ck.tokenizer) rows.push(["tokenizer", `${(ck.meta?.tokenizer as { type?: string } | undefined)?.type ?? "?"} · ${ck.tokenizer.vocabSize} tokens`]);
      const log = ck.meta?.trainingLog ?? [];
      const last = log[log.length - 1];
      const lastVal = [...log].reverse().find((e) => e.valLoss !== undefined)?.valLoss;
      if (last) rows.push(["trained", `${last.step} steps · loss ${last.loss.toFixed(3)}${lastVal !== undefined ? ` · val ${lastVal.toFixed(3)}` : ""}`]);
      if (ck.meta?.createdAt) rows.push(["created", ck.meta.createdAt.slice(0, 10)]);
      info.appendChild(kv(rows));
    }
  });
  return el;
}
