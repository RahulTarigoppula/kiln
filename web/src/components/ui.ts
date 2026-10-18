/** Shared UI building blocks (vanilla DOM). */
import { h, svg, icons, type Child } from "../lib/dom";
import type { Signal } from "../lib/signals";

export function pageHead(eyebrow: string, title: string, lede?: string, actions?: Child[]): HTMLElement {
  return h(
    "header",
    { class: "page-head" },
    h("div", null, h("div", { class: "page-head__eyebrow" }, eyebrow), h("h1", null, title), lede ? h("p", { class: "page-head__lede" }, lede) : null),
    actions && actions.length ? h("div", { class: "page-head__actions" }, ...actions) : null,
  );
}

export function card(title: string | null, body: Child[], opts: { sub?: string; actions?: Child[]; class?: string } = {}): HTMLElement {
  return h(
    "section",
    { class: `card ${opts.class ?? ""}`.trim() },
    title !== null
      ? h(
          "div",
          { class: "card__head" },
          h("div", null, h("h3", null, title), opts.sub ? h("div", { class: "card__sub" }, opts.sub) : null),
          opts.actions?.length ? h("div", { class: "row" }, ...opts.actions) : null,
        )
      : null,
    ...body,
  );
}

export interface TileHandle {
  el: HTMLElement;
  set(value: string, sub?: string): void;
}

export function tile(label: string, value = "—", opts: { unit?: string; ember?: boolean; sub?: string } = {}): TileHandle {
  const valueEl = h("span", null, value);
  const subEl = h("div", { class: "tile__sub" }, opts.sub ?? "");
  if (!opts.sub) subEl.style.display = "none";
  const el = h(
    "div",
    { class: "tile" },
    h("div", { class: "tile__label" }, label),
    h("div", { class: `tile__value ${opts.ember ? "tile__value--ember" : ""}` }, valueEl, opts.unit ? h("span", { class: "tile__unit" }, opts.unit) : null),
    subEl,
  );
  return {
    el,
    set(v, sub) {
      valueEl.textContent = v;
      if (sub !== undefined) {
        subEl.textContent = sub;
        subEl.style.display = sub ? "" : "none";
      }
    },
  };
}

export function button(label: string, opts: { icon?: keyof typeof icons; variant?: "primary" | "ghost" | "danger" | ""; size?: "sm" | ""; onClick?: (ev: MouseEvent) => void; title?: string; disabled?: boolean } = {}): HTMLButtonElement {
  const cls = ["btn", opts.variant ? `btn--${opts.variant}` : "", opts.size ? `btn--${opts.size}` : ""].filter(Boolean).join(" ");
  const b = h("button", { class: cls, type: "button", title: opts.title, disabled: opts.disabled, onclick: opts.onClick }, opts.icon ? svg(icons[opts.icon]) : null, label);
  return b;
}

export function iconEl(name: keyof typeof icons): SVGElement {
  return svg(icons[name]);
}

export interface FieldOpts {
  hint?: string;
  value?: HTMLElement;
}

export function field(label: string, control: HTMLElement, opts: FieldOpts = {}): HTMLElement {
  const id = control.id || `f-${Math.random().toString(36).slice(2, 8)}`;
  control.id = id;
  return h(
    "div",
    { class: "field" },
    h("label", { class: "field__label", for: id }, h("span", null, label), opts.value ?? null),
    control,
    opts.hint ? h("div", { class: "field__hint" }, opts.hint) : null,
  );
}

export function select(options: { value: string; label: string }[], value: string, onChange: (v: string) => void): HTMLSelectElement {
  const sel = h("select", { class: "select", onchange: () => onChange(sel.value) }, ...options.map((o) => h("option", { value: o.value }, o.label)));
  sel.value = value;
  return sel;
}

export function numberInput(value: number, opts: { min?: number; max?: number; step?: number; onChange: (v: number) => void; mono?: boolean }): HTMLInputElement {
  const inp = h("input", {
    class: `input ${opts.mono === false ? "" : "input--mono"}`,
    type: "number",
    value: String(value),
    min: opts.min,
    max: opts.max,
    step: opts.step,
    onchange: () => {
      const v = Number(inp.value);
      if (Number.isFinite(v)) opts.onChange(v);
    },
  });
  return inp;
}

export interface SliderOpts {
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}

/** Range slider with live value readout; returns the field wrapper and a setter. */
export function slider(label: string, value: number, opts: SliderOpts): { el: HTMLElement; set(v: number): void; input: HTMLInputElement } {
  const fmt = opts.format ?? ((v: number) => String(v));
  const readout = h("span", { class: "field__value" }, fmt(value));
  const input = h("input", { class: "range", type: "range", min: opts.min, max: opts.max, step: opts.step, value: String(value) });
  const paint = () => {
    const v = Number(input.value);
    const pct = ((v - opts.min) / (opts.max - opts.min)) * 100;
    input.style.setProperty("--pct", `${pct}%`);
    readout.textContent = fmt(v);
  };
  input.addEventListener("input", () => {
    paint();
    opts.onChange(Number(input.value));
  });
  paint();
  const el = field(label, input, { value: readout });
  return {
    el,
    input,
    set(v) {
      input.value = String(v);
      paint();
    },
  };
}

export function segmented<T extends string>(options: { value: T; label: string }[], current: Signal<T>, onChange?: (v: T) => void): HTMLElement {
  const btns = options.map((o) =>
    h(
      "button",
      {
        class: "seg__btn",
        type: "button",
        "aria-pressed": String(current.peek() === o.value),
        onclick: () => {
          current.set(o.value);
          onChange?.(o.value);
          for (const b of btns) b.setAttribute("aria-pressed", String(b.dataset["v"] === o.value));
        },
        dataset: { v: o.value },
      },
      o.label,
    ),
  );
  return h("div", { class: "seg", role: "group" }, ...btns);
}

export function notice(kind: "info" | "warn" | "bad" | "ember" | "ok", content: Child[]): HTMLElement {
  const icon = kind === "warn" || kind === "bad" ? "warn" : kind === "ok" ? "check" : "info";
  const cls = kind === "info" ? "notice" : `notice notice--${kind}`;
  return h("div", { class: cls, role: kind === "bad" ? "alert" : "status" }, svg(icons[icon]), h("div", null, ...content));
}

export function emptyState(title: string, text: string): HTMLElement {
  return h("div", { class: "empty" }, h("b", null, title), text);
}

export function chip(label: string, value: string | number, kind: "" | "ok" | "bad" | "warn" | "ember" = ""): HTMLElement {
  return h("span", { class: `chip ${kind ? `chip--${kind}` : ""}` }, label, " ", h("b", null, String(value)));
}

export function kv(rows: [string, string][]): HTMLElement {
  return h("dl", { class: "kv" }, ...rows.flatMap(([k, v]) => [h("dt", null, k), h("dd", { title: v }, v)]));
}

export function chartBox(cls = ""): { wrap: HTMLElement; setEmpty(title: string | null, text?: string): void } {
  const empty = h("div", { class: "chart__empty" });
  empty.style.display = "none";
  const wrap = h("div", { class: `chart ${cls}`.trim() }, empty);
  return {
    wrap,
    setEmpty(title, text = "") {
      if (!title) {
        empty.style.display = "none";
        return;
      }
      empty.replaceChildren(h("div", null, h("b", null, title), text));
      empty.style.display = "";
    },
  };
}

export function progressBar(): { el: HTMLElement; set(frac: number | null): void } {
  const bar = h("div", { class: "progress__bar" });
  const el = h("div", { class: "progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100" }, bar);
  return {
    el,
    set(frac) {
      if (frac === null) {
        el.classList.add("progress--indeterminate");
        el.removeAttribute("aria-valuenow");
        bar.style.width = "";
      } else {
        el.classList.remove("progress--indeterminate");
        const pct = Math.max(0, Math.min(100, frac * 100));
        bar.style.width = `${pct}%`;
        el.setAttribute("aria-valuenow", String(Math.round(pct)));
      }
    },
  };
}

let toastHost: HTMLElement | null = null;
export function toast(message: string, kind: "" | "ok" | "bad" = "", ms = 2800): void {
  if (!toastHost) {
    toastHost = h("div", { class: "toast-host", "aria-live": "polite" });
    document.body.appendChild(toastHost);
  }
  const t = h("div", { class: `toast ${kind ? `toast--${kind}` : ""}` }, message);
  toastHost.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

export function fileDrop(label: string, accept: string, onFiles: (files: FileList) => void, multiple = false): HTMLElement {
  const input = h("input", { type: "file", accept, multiple, onchange: () => input.files && input.files.length && onFiles(input.files) });
  const el = h(
    "label",
    {
      class: "file-drop",
      ondragover: (e) => {
        e.preventDefault();
        el.classList.add("is-over");
      },
      ondragleave: () => el.classList.remove("is-over"),
      ondrop: (e) => {
        e.preventDefault();
        el.classList.remove("is-over");
        if (e.dataTransfer?.files.length) onFiles(e.dataTransfer.files);
      },
    },
    svg(icons.upload, "file-drop__icon"),
    " ",
    label,
    input,
  );
  (el.querySelector("svg") as SVGElement).style.cssText = "width:14px;height:14px;vertical-align:-2px;margin-right:4px";
  return el;
}

export function downloadBlob(name: string, data: BlobPart, type = "application/octet-stream"): void {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = h("a", { href: url });
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
