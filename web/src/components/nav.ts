/** Top navigation: Kiln mark, page links, device badge, GitHub link. */
import { h, svg, icons, KILN_MARK } from "../lib/dom";
import { effect } from "../lib/signals";
import { adapterLabel, busyCount, gpuState } from "../lib/gpu";
import { currentPath, routes } from "../router";

export const REPO_URL = "https://github.com/RahulTarigoppula/kiln";
export const ARCH_URL = `${REPO_URL}/blob/main/docs/ARCHITECTURE.md`;

export function deviceBadge(): HTMLElement {
  const dot = h("span", { class: "badge__dot" });
  const label = h("span", { class: "badge__label" }, "Initialising…");
  const kind = h("span", { class: "badge__kind" }, "");
  const el = h("div", { class: "badge badge--init", role: "status", "aria-live": "polite" }, dot, label, kind);
  effect(() => {
    const s = gpuState();
    const busy = busyCount() > 0;
    el.classList.toggle("badge--busy", busy);
    el.classList.remove("badge--gpu", "badge--cpu", "badge--init");
    if (s.status === "ready") {
      el.classList.add("badge--gpu");
      label.textContent = adapterLabel(s.info);
      kind.textContent = "WebGPU";
      el.title = `${s.info?.vendor ?? ""} ${s.info?.architecture ?? ""}\n${s.info?.description ?? ""}\ntimestamp-query: ${s.info?.timestampQuery ? "yes" : "no"}${busy ? "\nrunning a job" : ""}`.trim();
    } else if (s.status === "unavailable") {
      el.classList.add("badge--cpu");
      label.textContent = "CPU fallback";
      kind.textContent = s.apiPresent ? "no adapter" : "no WebGPU";
      el.title = s.error ?? "WebGPU unavailable";
    } else {
      el.classList.add("badge--init");
      label.textContent = s.status === "initializing" ? "Requesting GPU…" : "Device";
      kind.textContent = "";
    }
  });
  return el;
}

export function topNav(): HTMLElement {
  const links = routes
    .filter((r) => r.nav)
    .map((r) => h("a", { class: "nav__link", href: `#${r.path}`, dataset: { path: r.path } }, r.title));
  effect(() => {
    const p = currentPath();
    for (const a of links) {
      if (a.dataset["path"] === p) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    }
  });
  return h(
    "nav",
    { class: "nav", "aria-label": "Primary" },
    h(
      "div",
      { class: "container nav__inner" },
      h("a", { class: "brand", href: "#/studio" }, svg(KILN_MARK, "brand__mark"), "Kiln", h("span", { class: "brand__version" }, "v0.1")),
      h("div", { class: "nav__links" }, ...links),
      h(
        "div",
        { class: "nav__right" },
        deviceBadge(),
        h("a", { class: "icon-link", href: REPO_URL, target: "_blank", rel: "noopener", title: "Source on GitHub", "aria-label": "Source on GitHub" }, svg(icons.github)),
      ),
    ),
  );
}

export function footer(): HTMLElement {
  return h(
    "footer",
    { class: "footer" },
    h(
      "div",
      { class: "container footer__inner" },
      h("div", null, "Kiln - a WebGPU deep-learning framework written from scratch in TypeScript. Zero runtime dependencies."),
      h(
        "div",
        { class: "row" },
        h("a", { href: "#/about" }, "About"),
        h("a", { href: ARCH_URL, target: "_blank", rel: "noopener" }, "Architecture"),
        h("a", { href: REPO_URL, target: "_blank", rel: "noopener" }, "GitHub"),
      ),
    ),
  );
}
