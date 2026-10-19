/** Hash router: `#/studio`, `#/playground`, … Each page module exports `mount(root) -> dispose`. */
import { signal } from "./lib/signals";

export type PageMount = (root: HTMLElement) => void | (() => void) | Promise<void | (() => void)>;

export interface Route {
  path: string;
  title: string;
  nav: boolean;
  load: () => Promise<{ mount: PageMount }>;
}

export const routes: Route[] = [
  { path: "/studio", title: "Studio", nav: true, load: () => import("./pages/studio") },
  { path: "/playground", title: "Playground", nav: true, load: () => import("./pages/playground") },
  { path: "/inspect", title: "Inspect", nav: true, load: () => import("./pages/inspect") },
  { path: "/bench", title: "Bench", nav: true, load: () => import("./pages/bench") },
  { path: "/tests", title: "Tests", nav: true, load: () => import("./pages/tests") },
  { path: "/about", title: "About", nav: false, load: () => import("./pages/about") },
];

export const DEFAULT_ROUTE = "/studio";

export const currentPath = signal<string>(parseHash());

function parseHash(): string {
  const raw = location.hash.replace(/^#/, "");
  const path = raw.split("?")[0] ?? "";
  return path.startsWith("/") ? path.replace(/\/+$/, "") || "/" : DEFAULT_ROUTE;
}

export function navigate(path: string): void {
  location.hash = `#${path}`;
}

export function startRouter(outlet: HTMLElement): void {
  let dispose: (() => void) | void;
  let token = 0;

  const render = async (): Promise<void> => {
    const path = parseHash();
    if (!location.hash) {
      location.replace(`#${DEFAULT_ROUTE}`);
      return;
    }
    currentPath.set(path);
    const route = routes.find((r) => r.path === path);
    const my = ++token;
    if (dispose) {
      dispose();
      dispose = undefined;
    }
    outlet.replaceChildren();
    outlet.scrollTop = 0;
    window.scrollTo({ top: 0 });
    document.title = route ? `${route.title} - Kiln` : "Kiln";
    if (!route) {
      outlet.appendChild(notFound(path));
      return;
    }
    try {
      const mod = await route.load();
      if (my !== token) return;
      const d = await mod.mount(outlet);
      if (my !== token) {
        if (typeof d === "function") d();
        return;
      }
      dispose = d ?? undefined;
    } catch (err: unknown) {
      console.error(err);
      if (my !== token) return;
      const pre = document.createElement("pre");
      pre.className = "notice notice--bad";
      pre.textContent = `Failed to load page: ${err instanceof Error ? err.message : String(err)}`;
      outlet.appendChild(pre);
    }
  };

  window.addEventListener("hashchange", () => void render());
  void render();
}

function notFound(path: string): HTMLElement {
  const el = document.createElement("div");
  el.className = "notfound";
  const h2 = document.createElement("h2");
  h2.textContent = "Nothing fired here";
  const p = document.createElement("p");
  p.textContent = `No page at ${path}.`;
  const a = document.createElement("a");
  a.href = `#${DEFAULT_ROUTE}`;
  a.textContent = "Back to the Studio";
  el.append(h2, p, a);
  return el;
}
