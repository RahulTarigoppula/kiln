/** Minimal hyperscript helpers. */

export type Child = Node | string | number | null | undefined | false | Child[];

type EventHandlers = {
  [K in keyof HTMLElementEventMap as `on${K}`]?: (ev: HTMLElementEventMap[K]) => void;
};

export interface Props extends EventHandlers {
  class?: string;
  id?: string;
  style?: string;
  title?: string;
  html?: string;
  text?: string;
  for?: string;
  type?: string;
  value?: string | number;
  placeholder?: string;
  disabled?: boolean;
  checked?: boolean;
  href?: string;
  target?: string;
  rel?: string;
  min?: string | number;
  max?: string | number;
  step?: string | number;
  rows?: number;
  accept?: string;
  multiple?: boolean;
  tabindex?: number;
  role?: string;
  dataset?: Record<string, string>;
  [attr: `aria-${string}`]: string | boolean | undefined;
  [attr: `data-${string}`]: string | undefined;
}

function append(parent: Node, child: Child): void {
  if (child === null || child === undefined || child === false) return;
  if (Array.isArray(child)) {
    for (const c of child) append(parent, c);
    return;
  }
  if (child instanceof Node) {
    parent.appendChild(child);
    return;
  }
  parent.appendChild(document.createTextNode(String(child)));
}

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, props?: Props | null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (props) {
    for (const [key, raw] of Object.entries(props)) {
      if (raw === undefined || raw === null || raw === false) continue;
      if (key.startsWith("on") && typeof raw === "function") {
        el.addEventListener(key.slice(2), raw as EventListener);
      } else if (key === "class") {
        el.className = String(raw);
      } else if (key === "html") {
        el.innerHTML = String(raw);
      } else if (key === "text") {
        el.textContent = String(raw);
      } else if (key === "dataset") {
        for (const [k, v] of Object.entries(raw as Record<string, string>)) el.dataset[k] = v;
      } else if (key === "value") {
        (el as unknown as { value: string }).value = String(raw);
      } else if (key === "checked" || key === "disabled" || key === "multiple") {
        (el as unknown as Record<string, boolean>)[key] = Boolean(raw);
      } else if (raw === true) {
        el.setAttribute(key, "");
      } else {
        el.setAttribute(key, String(raw));
      }
    }
  }
  for (const c of children) append(el, c);
  return el;
}

/** Create an element from an SVG markup string. */
export function svg(markup: string, className?: string): SVGElement {
  const tpl = document.createElement("template");
  tpl.innerHTML = markup.trim();
  const el = tpl.content.firstElementChild as SVGElement;
  if (className) el.setAttribute("class", className);
  return el;
}

export function clear(el: Element): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function replaceChildren(el: Element, ...children: Child[]): void {
  clear(el);
  for (const c of children) append(el, c);
}

/** Inline icons (Lucide-style strokes, 24-unit viewBox). */
export const icons = {
  play: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l12-7.5z"/></svg>`,
  pause: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="4.5" width="4" height="15" rx="1"/><rect x="14" y="4.5" width="4" height="15" rx="1"/></svg>`,
  stop: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="5.5" y="5.5" width="13" height="13" rx="2"/></svg>`,
  download: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 4v11m0 0 4-4m-4 4-4-4M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"/></svg>`,
  copy: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h10"/></svg>`,
  refresh: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 11A8 8 0 1 0 18.3 17"/><path d="M20 4v7h-7"/></svg>`,
  zap: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>`,
  info: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-8h.01"/></svg>`,
  warn: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4m0 4h.01"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 12 5 5L20 7"/></svg>`,
  upload: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V5m0 0 4 4m-4-4-4 4M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"/></svg>`,
  github: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.7.5.5 5.7.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.3.8-.6v-2c-3.2.7-3.9-1.4-3.9-1.4-.5-1.3-1.3-1.7-1.3-1.7-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.7 1.3 3.4 1 .1-.8.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.7 0-1.3.4-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.2 1.2a11 11 0 0 1 5.8 0c2.2-1.5 3.2-1.2 3.2-1.2.6 1.6.2 2.8.1 3.1.8.8 1.2 1.8 1.2 3.1 0 4.4-2.7 5.4-5.3 5.7.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6 4.6-1.5 7.9-5.8 7.9-10.9C23.5 5.7 18.3.5 12 .5z"/></svg>`,
  flame: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 22c-4 0-6.5-2.7-6.5-6.2 0-3.3 2.4-4.9 3-7.6 1.3 1.6 1.9 3.2 1.6 4.8 1.3-1.5 2.1-3.8 2.1-6.5 3.6 2.3 5.3 5.4 5.3 8.9 0 3.7-2.6 6.6-5.5 6.6z"/></svg>`,
};

/** The Kiln mark: a kiln arch with an ember inside. */
export const KILN_MARK = `<svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
  <path d="M7 25V15.5a9 9 0 0 1 18 0V25" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  <path d="M5 25.5h22" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  <path d="M16 22.5c-2.6 0-4.2-1.7-4.2-3.9 0-2 1.5-3 1.9-4.7.9 1.1 1.3 2.1 1.1 3.1.9-1 1.5-2.5 1.5-4.2 2.4 1.5 3.6 3.5 3.6 5.8 0 2.3-1.7 3.9-3.9 3.9z" fill="#ff7a1a"/>
</svg>`;
