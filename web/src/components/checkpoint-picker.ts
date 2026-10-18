/** Checkpoint picker shared by Playground and Inspect (shipped checkpoint or local files). */
import { h } from "../lib/dom";
import type { Scope } from "../lib/signals";
import { fmtBytes, fmtCompact } from "../lib/format";
import { estimateParams } from "../lib/engine";
import { checkpoint, loadError, loadFromFiles, loadShipped, loading, model, modelUnavailable, SHIPPED_CHECKPOINT } from "../lib/model-store";
import { button, fileDrop, kv, notice } from "./ui";

export function checkpointPicker(scope: Scope): HTMLElement {
  const loadBtn = button("Load", { icon: "download", variant: "primary", size: "sm", onClick: () => void loadShipped() });
  const status = h("div", { class: "stack stack--sm" });
  const info = h("div");

  const el = h(
    "div",
    { class: "stack" },
    h(
      "div",
      { class: "row row--between", style: "gap:12px;align-items:flex-start" },
      h("div", { style: "min-width:0" }, h("div", { style: "font-weight:600;color:var(--ash-0)" }, SHIPPED_CHECKPOINT.label), h("div", { class: "small", style: "font-family:var(--font-mono);font-size:11px" }, SHIPPED_CHECKPOINT.blurb)),
      loadBtn,
    ),
    fileDrop("Drop a .safetensors + .json sidecar, or click to browse", ".safetensors,.json", (files) => void loadFromFiles(files), true),
    status,
    info,
  );

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
    if (ck && unavailable) status.appendChild(notice("ember", [h("b", null, "Checkpoint parsed; model engine pending. "), "Weights, config and tokenizer are loaded — instantiating the GPT needs the phase-2 modules (see web/src/lib/engine.ts)."]));
    if (ck && m) status.appendChild(notice("ok", [`Model ready on ${m.device}.`]));
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
      if (last) rows.push(["trained", `${last.step} steps · loss ${last.loss.toFixed(3)}${last.valLoss !== undefined ? ` · val ${last.valLoss.toFixed(3)}` : ""}`]);
      if (ck.meta?.createdAt) rows.push(["created", ck.meta.createdAt.slice(0, 10)]);
      info.appendChild(kv(rows));
    }
  });
  return el;
}
