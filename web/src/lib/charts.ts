/**
 * Canvas charts for Kiln: line chart (log-scale, multi-series, hover crosshair), bar chart,
 * attention heatmap and 2-D scatter. HiDPI aware, resize-observed, theme colours from CSS variables.
 */
import { fmtTick } from "./format";

export interface ChartTheme {
  bg: string;
  grid: string;
  axis: string;
  text: string;
  textDim: string;
  ember: string;
  cool: string;
  font: string;
  display: string;
}

let themeCache: ChartTheme | null = null;

export function chartTheme(): ChartTheme {
  if (themeCache) return themeCache;
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  themeCache = {
    bg: v("--bg", "#0e0d0b"),
    grid: v("--line", "rgba(255,244,230,0.08)"),
    axis: v("--line-strong", "rgba(255,244,230,0.16)"),
    text: v("--ash-2", "#b3ada3"),
    textDim: v("--ash-3", "#7f7a71"),
    ember: v("--ember", "#ff7a1a"),
    cool: v("--cool", "#8db7e6"),
    font: `11px ${v("--font-mono", "monospace")}`,
    display: `12px ${v("--font-ui", "sans-serif")}`,
  };
  return themeCache;
}

export interface TooltipRow {
  label: string;
  value: string;
  color?: string;
}

export interface TooltipContent {
  title?: string;
  rows: TooltipRow[];
}

/** Shared canvas plumbing: HiDPI sizing, resize, pointer tracking and a DOM tooltip. */
export abstract class CanvasChart {
  readonly canvas: HTMLCanvasElement;
  protected readonly ctx: CanvasRenderingContext2D;
  protected w = 0;
  protected h = 0;
  protected dpr = 1;
  protected pointer: { x: number; y: number } | null = null;
  private ro: ResizeObserver | null = null;
  private raf = 0;
  private tooltip: HTMLDivElement | null = null;
  private destroyed = false;

  constructor(readonly container: HTMLElement) {
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("role", "img");
    container.appendChild(this.canvas);
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.canvas.addEventListener("pointermove", this.onMove);
    this.canvas.addEventListener("pointerleave", this.onLeave);
    this.resize();
  }

  private onMove = (ev: PointerEvent): void => {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer = { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
    const tip = this.hover(this.pointer.x, this.pointer.y);
    this.showTooltip(tip, ev.clientX, ev.clientY);
    this.requestDraw();
  };

  private onLeave = (): void => {
    this.pointer = null;
    this.showTooltip(null, 0, 0);
    this.requestDraw();
  };

  protected showTooltip(content: TooltipContent | null, clientX: number, clientY: number): void {
    if (!content || content.rows.length === 0) {
      this.tooltip?.classList.remove("is-on");
      return;
    }
    if (!this.tooltip) {
      this.tooltip = document.createElement("div");
      this.tooltip.className = "tooltip";
      document.body.appendChild(this.tooltip);
    }
    const t = this.tooltip;
    t.replaceChildren();
    if (content.title) {
      const title = document.createElement("div");
      title.style.color = "var(--ash-3)";
      title.style.marginBottom = "4px";
      title.textContent = content.title;
      t.appendChild(title);
    }
    for (const row of content.rows) {
      const r = document.createElement("div");
      r.className = "tooltip__row";
      const l = document.createElement("span");
      l.textContent = row.label;
      if (row.color) l.style.color = row.color;
      const v = document.createElement("b");
      v.textContent = row.value;
      if (row.color) v.style.color = row.color;
      r.append(l, v);
      t.appendChild(r);
    }
    t.classList.add("is-on");
    const pad = 14;
    const tw = t.offsetWidth;
    const th = t.offsetHeight;
    let x = clientX + pad;
    let y = clientY + pad;
    if (x + tw > window.innerWidth - 8) x = clientX - tw - pad;
    if (y + th > window.innerHeight - 8) y = clientY - th - pad;
    t.style.left = `${Math.max(4, x)}px`;
    t.style.top = `${Math.max(4, y)}px`;
  }

  protected resize(): void {
    const rect = this.container.getBoundingClientRect();
    this.dpr = Math.min(3, window.devicePixelRatio || 1);
    this.w = Math.max(10, Math.floor(rect.width));
    this.h = Math.max(10, Math.floor(rect.height));
    this.canvas.width = Math.floor(this.w * this.dpr);
    this.canvas.height = Math.floor(this.h * this.dpr);
    this.canvas.style.width = `${this.w}px`;
    this.canvas.style.height = `${this.h}px`;
    this.requestDraw();
  }

  requestDraw(): void {
    if (this.raf || this.destroyed) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      if (this.destroyed) return;
      const ctx = this.ctx;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.w, this.h);
      this.render();
    });
  }

  protected abstract render(): void;
  protected abstract hover(x: number, y: number): TooltipContent | null;

  destroy(): void {
    this.destroyed = true;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.ro?.disconnect();
    this.ro = null;
    this.canvas.removeEventListener("pointermove", this.onMove);
    this.canvas.removeEventListener("pointerleave", this.onLeave);
    this.canvas.remove();
    this.tooltip?.remove();
    this.tooltip = null;
  }
}

// ---------------------------------------------------------------- scales & ticks

interface Scale {
  min: number;
  max: number;
  log: boolean;
  toPx(v: number): number;
  ticks(): number[];
}

function makeScale(min: number, max: number, pxStart: number, pxEnd: number, log: boolean, approxTicks = 5): Scale {
  if (log) {
    const lo = Math.log10(Math.max(min, 1e-12));
    const hi = Math.log10(Math.max(max, min * 1.0001, 1e-12));
    const span = hi - lo || 1;
    return {
      min,
      max,
      log: true,
      toPx: (v) => pxStart + ((Math.log10(Math.max(v, 1e-12)) - lo) / span) * (pxEnd - pxStart),
      ticks: () => {
        const out: number[] = [];
        const d0 = Math.floor(lo);
        const d1 = Math.ceil(hi);
        const decades = d1 - d0;
        const mults = decades <= 2 ? [1, 2, 5] : decades <= 4 ? [1, 3] : [1];
        for (let d = d0; d <= d1; d++) {
          for (const m of mults) {
            const v = m * 10 ** d;
            if (v >= min / 1.0001 && v <= max * 1.0001) out.push(v);
          }
        }
        return out;
      },
    };
  }
  const span = max - min || 1;
  return {
    min,
    max,
    log: false,
    toPx: (v) => pxStart + ((v - min) / span) * (pxEnd - pxStart),
    ticks: () => {
      const step = niceStep(span / approxTicks);
      const out: number[] = [];
      const start = Math.ceil(min / step) * step;
      for (let v = start; v <= max + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
      return out;
    },
  };
}

function niceStep(raw: number): number {
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const f = raw / pow;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return nice * pow;
}

// ---------------------------------------------------------------- line chart

export interface LineSeries {
  name: string;
  color: string;
  xs: ArrayLike<number>;
  ys: ArrayLike<number>;
  glow?: boolean;
  dashed?: boolean;
  /** Draw points instead of a line (sparse series such as eval loss). */
  points?: boolean;
}

export interface LineChartOptions {
  logY?: boolean;
  xLabel?: string;
  yLabel?: string;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  /** Fixed y domain; otherwise derived from data. */
  yMin?: number;
  yMax?: number;
  /** Lower bound for the x axis (e.g. 0 for steps). */
  xMin?: number;
}

export class LineChart extends CanvasChart {
  private series: LineSeries[] = [];
  private opts: LineChartOptions;
  private pad = { l: 54, r: 18, t: 16, b: 34 };

  constructor(container: HTMLElement, opts: LineChartOptions = {}) {
    super(container);
    this.opts = opts;
  }

  setOptions(opts: Partial<LineChartOptions>): void {
    this.opts = { ...this.opts, ...opts };
    this.requestDraw();
  }

  setSeries(series: LineSeries[]): void {
    this.series = series;
    this.requestDraw();
  }

  private domain(): { x: Scale; y: Scale } | null {
    const log = this.opts.logY ?? false;
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    let count = 0;
    for (const s of this.series) {
      const n = Math.min(s.xs.length, s.ys.length);
      for (let i = 0; i < n; i++) {
        const x = s.xs[i] ?? NaN;
        const y = s.ys[i] ?? NaN;
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        if (log && y <= 0) continue;
        count++;
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
      }
    }
    if (count === 0) return null;
    if (this.opts.xMin !== undefined) xmin = Math.min(xmin, this.opts.xMin);
    if (xmax === xmin) xmax = xmin + 1;
    if (this.opts.yMin !== undefined) ymin = this.opts.yMin;
    if (this.opts.yMax !== undefined) ymax = this.opts.yMax;
    if (ymax === ymin) {
      ymax = log ? ymin * 2 : ymin + 1;
      ymin = log ? ymin / 2 : ymin - 1;
    }
    if (!log) {
      const padY = (ymax - ymin) * 0.08;
      ymin -= padY;
      ymax += padY;
    } else {
      ymin /= 1.15;
      ymax *= 1.15;
    }
    const { l, r, t, b } = this.pad;
    return {
      x: makeScale(xmin, xmax, l, this.w - r, false, Math.max(3, Math.floor((this.w - l - r) / 90))),
      y: makeScale(ymin, ymax, this.h - b, t, log, Math.max(3, Math.floor((this.h - t - b) / 48))),
    };
  }

  protected render(): void {
    const ctx = this.ctx;
    const th = chartTheme();
    const dom = this.domain();
    const { l, r, t, b } = this.pad;
    const plotW = this.w - l - r;
    const plotH = this.h - t - b;

    // frame
    ctx.font = th.font;
    ctx.fillStyle = th.textDim;
    if (!dom) {
      return;
    }
    const { x, y } = dom;
    const yfmt = this.opts.yFormat ?? fmtTick;
    const xfmt = this.opts.xFormat ?? fmtTick;

    // grid
    ctx.lineWidth = 1;
    ctx.strokeStyle = th.grid;
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const tv of y.ticks()) {
      const py = Math.round(y.toPx(tv)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(l, py);
      ctx.lineTo(l + plotW, py);
      ctx.stroke();
      ctx.fillStyle = th.textDim;
      ctx.fillText(yfmt(tv), l - 8, py);
    }
    ctx.textBaseline = "top";
    ctx.textAlign = "center";
    for (const tv of x.ticks()) {
      const px = Math.round(x.toPx(tv)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, t);
      ctx.lineTo(px, t + plotH);
      ctx.stroke();
      ctx.fillText(xfmt(tv), px, t + plotH + 8);
    }
    // axes
    ctx.strokeStyle = th.axis;
    ctx.beginPath();
    ctx.moveTo(l + 0.5, t);
    ctx.lineTo(l + 0.5, t + plotH + 0.5);
    ctx.lineTo(l + plotW, t + plotH + 0.5);
    ctx.stroke();
    if (this.opts.yLabel) {
      ctx.save();
      ctx.translate(12, t + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = th.textDim;
      ctx.fillText(this.opts.yLabel, 0, 0);
      ctx.restore();
    }
    if (this.opts.xLabel) {
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      ctx.fillStyle = th.textDim;
      ctx.fillText(this.opts.xLabel, l + plotW, this.h - 2);
    }

    // series
    ctx.save();
    ctx.beginPath();
    ctx.rect(l, t - 2, plotW + 1, plotH + 3);
    ctx.clip();
    for (const s of this.series) {
      const n = Math.min(s.xs.length, s.ys.length);
      if (n === 0) continue;
      const stride = Math.max(1, Math.floor(n / (plotW * 2)));
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      if (s.points) {
        ctx.fillStyle = s.color;
        ctx.shadowColor = s.color;
        ctx.shadowBlur = s.glow ? 10 : 0;
        for (let i = 0; i < n; i++) {
          const yv = s.ys[i] ?? NaN;
          if (!Number.isFinite(yv) || (y.log && yv <= 0)) continue;
          ctx.beginPath();
          ctx.arc(x.toPx(s.xs[i] ?? 0), y.toPx(yv), 3, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.shadowBlur = 0;
        if (n > 1) {
          ctx.strokeStyle = s.color;
          ctx.globalAlpha = 0.55;
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          let started = false;
          for (let i = 0; i < n; i++) {
            const yv = s.ys[i] ?? NaN;
            if (!Number.isFinite(yv) || (y.log && yv <= 0)) continue;
            const px = x.toPx(s.xs[i] ?? 0);
            const py = y.toPx(yv);
            if (!started) {
              ctx.moveTo(px, py);
              started = true;
            } else ctx.lineTo(px, py);
          }
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        }
        continue;
      }
      const path = new Path2D();
      let started = false;
      let lastX = 0;
      let lastY = 0;
      for (let i = 0; i < n; i += stride) {
        const yv = s.ys[i] ?? NaN;
        if (!Number.isFinite(yv) || (y.log && yv <= 0)) continue;
        const px = x.toPx(s.xs[i] ?? 0);
        const py = y.toPx(yv);
        if (!started) {
          path.moveTo(px, py);
          started = true;
        } else path.lineTo(px, py);
        lastX = px;
        lastY = py;
      }
      if (!started) continue;
      if (s.dashed) ctx.setLineDash([5, 4]);
      if (s.glow) {
        ctx.strokeStyle = s.color;
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = 6;
        ctx.shadowColor = s.color;
        ctx.shadowBlur = 16;
        ctx.stroke(path);
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.8;
      ctx.stroke(path);
      ctx.setLineDash([]);
      // end dot
      ctx.fillStyle = s.color;
      ctx.shadowColor = s.color;
      ctx.shadowBlur = s.glow ? 12 : 0;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 2.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
    ctx.restore();

    // crosshair
    if (this.pointer && this.pointer.x >= l && this.pointer.x <= l + plotW) {
      const near = this.nearest(this.pointer.x);
      if (near) {
        const px = x.toPx(near.xv);
        ctx.strokeStyle = th.axis;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(Math.round(px) + 0.5, t);
        ctx.lineTo(Math.round(px) + 0.5, t + plotH);
        ctx.stroke();
        ctx.setLineDash([]);
        for (const hit of near.hits) {
          ctx.fillStyle = hit.series.color;
          ctx.beginPath();
          ctx.arc(x.toPx(hit.x), y.toPx(hit.y), 3.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = th.bg;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
    }
  }

  private nearest(px: number): { xv: number; hits: { series: LineSeries; x: number; y: number }[] } | null {
    const dom = this.domain();
    if (!dom) return null;
    const { l, r } = this.pad;
    const frac = (px - l) / (this.w - l - r);
    const xv = dom.x.min + frac * (dom.x.max - dom.x.min);
    const hits: { series: LineSeries; x: number; y: number }[] = [];
    let bestX = xv;
    let bestD = Infinity;
    for (const s of this.series) {
      const n = Math.min(s.xs.length, s.ys.length);
      let bi = -1;
      let bd = Infinity;
      for (let i = 0; i < n; i++) {
        const yv = s.ys[i] ?? NaN;
        if (!Number.isFinite(yv) || (dom.y.log && yv <= 0)) continue;
        const d = Math.abs((s.xs[i] ?? 0) - xv);
        if (d < bd) {
          bd = d;
          bi = i;
        }
      }
      if (bi >= 0) {
        const hx = s.xs[bi] ?? 0;
        hits.push({ series: s, x: hx, y: s.ys[bi] ?? 0 });
        if (bd < bestD) {
          bestD = bd;
          bestX = hx;
        }
      }
    }
    if (!hits.length) return null;
    return { xv: bestX, hits };
  }

  protected hover(px: number): TooltipContent | null {
    const near = this.nearest(px);
    if (!near) return null;
    const xfmt = this.opts.xFormat ?? fmtTick;
    const yfmt = this.opts.yFormat ?? ((v: number) => v.toFixed(4));
    return {
      title: `${this.opts.xLabel ?? "x"} ${xfmt(near.xv)}`,
      rows: near.hits.map((hit) => ({ label: hit.series.name, value: yfmt(hit.y), color: hit.series.color })),
    };
  }
}

// ---------------------------------------------------------------- bar chart

export interface BarSeries {
  name: string;
  color: string;
  values: (number | null)[];
}

export interface BarChartOptions {
  categories: string[];
  series: BarSeries[];
  logY?: boolean;
  yLabel?: string;
  valueFormat?: (v: number) => string;
  /** Show the value above each bar. */
  showValues?: boolean;
}

export class BarChart extends CanvasChart {
  private opts: BarChartOptions;
  private pad = { l: 56, r: 12, t: 22, b: 36 };
  private bars: { x: number; y: number; w: number; h: number; ci: number; si: number; v: number }[] = [];

  constructor(container: HTMLElement, opts: BarChartOptions) {
    super(container);
    this.opts = opts;
  }

  setData(opts: Partial<BarChartOptions>): void {
    this.opts = { ...this.opts, ...opts };
    this.requestDraw();
  }

  protected render(): void {
    const ctx = this.ctx;
    const th = chartTheme();
    const { categories, series } = this.opts;
    const { l, r, t, b } = this.pad;
    const plotW = this.w - l - r;
    const plotH = this.h - t - b;
    this.bars = [];
    ctx.font = th.font;
    const vals: number[] = [];
    for (const s of series) for (const v of s.values) if (v !== null && Number.isFinite(v) && (!this.opts.logY || v > 0)) vals.push(v);
    if (!categories.length || !vals.length) return;
    const log = this.opts.logY ?? false;
    let vmin = log ? Math.min(...vals) / 2 : 0;
    let vmax = Math.max(...vals);
    if (log) {
      vmin = 10 ** Math.floor(Math.log10(vmin));
      vmax *= 1.6;
    } else vmax *= 1.12;
    if (vmax <= vmin) vmax = vmin + 1;
    const y = makeScale(vmin, vmax, t + plotH, t, log, 5);
    const fmt = this.opts.valueFormat ?? fmtTick;

    // grid
    ctx.strokeStyle = th.grid;
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of y.ticks()) {
      const py = Math.round(y.toPx(tv)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(l, py);
      ctx.lineTo(l + plotW, py);
      ctx.stroke();
      ctx.fillStyle = th.textDim;
      ctx.fillText(fmtTick(tv), l - 8, py);
    }
    ctx.strokeStyle = th.axis;
    ctx.beginPath();
    ctx.moveTo(l, t + plotH + 0.5);
    ctx.lineTo(l + plotW, t + plotH + 0.5);
    ctx.stroke();
    if (this.opts.yLabel) {
      ctx.save();
      ctx.translate(12, t + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillStyle = th.textDim;
      ctx.fillText(this.opts.yLabel, 0, 0);
      ctx.restore();
    }

    const groupW = plotW / categories.length;
    const inner = groupW * 0.68;
    const barW = inner / Math.max(1, series.length);
    const hovered = this.hitBar(this.pointer?.x ?? -1, this.pointer?.y ?? -1);
    categories.forEach((cat, ci) => {
      const gx = l + ci * groupW + (groupW - inner) / 2;
      series.forEach((s, si) => {
        const v = s.values[ci] ?? null;
        if (v === null || !Number.isFinite(v) || (log && v <= 0)) return;
        const bx = gx + si * barW + 1;
        const bw = Math.max(1, barW - 2);
        const by = y.toPx(v);
        const bh = t + plotH - by;
        const bar = { x: bx, y: by, w: bw, h: bh, ci, si, v };
        this.bars.push(bar);
        const isHover = hovered && hovered.ci === ci && hovered.si === si;
        ctx.fillStyle = s.color;
        ctx.globalAlpha = isHover ? 1 : 0.85;
        ctx.shadowColor = s.color;
        ctx.shadowBlur = isHover ? 18 : 8;
        roundRect(ctx, bx, by, bw, bh, 3);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
        if (this.opts.showValues && bw > 22) {
          ctx.fillStyle = th.text;
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText(fmt(v), bx + bw / 2, by - 3);
        }
      });
      ctx.fillStyle = th.text;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(cat, l + ci * groupW + groupW / 2, t + plotH + 8);
    });
  }

  private hitBar(px: number, py: number) {
    for (const b of this.bars) if (px >= b.x && px <= b.x + b.w && py >= b.y - 4 && py <= b.y + b.h) return b;
    return null;
  }

  protected hover(px: number, py: number): TooltipContent | null {
    const b = this.hitBar(px, py);
    if (!b) return null;
    const s = this.opts.series[b.si];
    const fmt = this.opts.valueFormat ?? fmtTick;
    return { title: this.opts.categories[b.ci] ?? "", rows: [{ label: s?.name ?? "", value: fmt(b.v), color: s?.color }] };
  }
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const rr = Math.min(r, w / 2, Math.max(0, h));
  ctx.beginPath();
  ctx.moveTo(x, y + h);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h);
  ctx.closePath();
}

// ---------------------------------------------------------------- heatmap

export interface HeatmapData {
  values: ArrayLike<number>;
  rows: number;
  cols: number;
  xLabels?: string[];
  yLabels?: string[];
  vmin?: number;
  vmax?: number;
  /** Axis titles. */
  xTitle?: string;
  yTitle?: string;
}

/** Ember colormap: near-black -> deep ember -> ember -> pale gold. */
export function emberColor(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  const stops: [number, [number, number, number]][] = [
    [0, [18, 16, 14]],
    [0.25, [92, 34, 10]],
    [0.55, [201, 86, 10]],
    [0.8, [255, 122, 26]],
    [1, [255, 226, 160]],
  ];
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i]!;
    const [t0, c0] = stops[i - 1]!;
    if (x <= t1) {
      const f = (x - t0) / (t1 - t0);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(255,226,160)";
}

export class Heatmap extends CanvasChart {
  private data: HeatmapData | null = null;
  private geom = { x0: 0, y0: 0, cw: 0, ch: 0 };

  setData(data: HeatmapData | null): void {
    this.data = data;
    this.requestDraw();
  }

  protected render(): void {
    const d = this.data;
    const ctx = this.ctx;
    const th = chartTheme();
    if (!d || d.rows === 0 || d.cols === 0) return;
    ctx.font = th.font;
    const maxLabel = Math.max(0, ...(d.yLabels ?? []).map((s) => ctx.measureText(s).width));
    const left = Math.min(110, Math.max(28, maxLabel + 12)) + (d.yTitle ? 14 : 0);
    const bottom = d.xLabels ? Math.min(90, Math.max(28, Math.max(0, ...d.xLabels.map((s) => ctx.measureText(s).width)) + 12)) + (d.xTitle ? 14 : 0) : 10;
    const top = 8;
    const right = 8;
    const size = Math.min(this.w - left - right, this.h - top - bottom);
    const cw = size / d.cols;
    const ch = size / d.rows;
    const x0 = left;
    const y0 = top;
    this.geom = { x0, y0, cw, ch };
    let vmin = d.vmin ?? Infinity;
    let vmax = d.vmax ?? -Infinity;
    if (d.vmin === undefined || d.vmax === undefined) {
      for (let i = 0; i < d.rows * d.cols; i++) {
        const v = d.values[i] ?? 0;
        if (d.vmin === undefined && v < vmin) vmin = v;
        if (d.vmax === undefined && v > vmax) vmax = v;
      }
    }
    if (!(vmax > vmin)) vmax = vmin + 1;
    const hovered = this.cellAt(this.pointer?.x ?? -1, this.pointer?.y ?? -1);
    for (let r = 0; r < d.rows; r++) {
      for (let c = 0; c < d.cols; c++) {
        const v = d.values[r * d.cols + c] ?? 0;
        ctx.fillStyle = emberColor((v - vmin) / (vmax - vmin));
        ctx.fillRect(x0 + c * cw, y0 + r * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    // grid hairlines when cells are big enough
    if (cw > 14) {
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.lineWidth = 1;
      for (let c = 0; c <= d.cols; c++) {
        const px = Math.round(x0 + c * cw) + 0.5;
        ctx.beginPath();
        ctx.moveTo(px, y0);
        ctx.lineTo(px, y0 + size);
        ctx.stroke();
      }
      for (let r = 0; r <= d.rows; r++) {
        const py = Math.round(y0 + r * ch) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x0, py);
        ctx.lineTo(x0 + size, py);
        ctx.stroke();
      }
    }
    if (hovered) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0 + hovered.c * cw + 0.75, y0 + hovered.r * ch + 0.75, cw - 1.5, ch - 1.5);
    }
    // labels
    const step = Math.max(1, Math.ceil(12 / Math.min(cw, ch)));
    ctx.fillStyle = th.text;
    if (d.yLabels) {
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let r = 0; r < d.rows; r += step) {
        ctx.fillStyle = hovered?.r === r ? th.ember : th.text;
        ctx.fillText(d.yLabels[r] ?? "", x0 - 6, y0 + r * ch + ch / 2);
      }
    }
    if (d.xLabels) {
      for (let c = 0; c < d.cols; c += step) {
        ctx.save();
        ctx.translate(x0 + c * cw + cw / 2, y0 + size + 6);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillStyle = hovered?.c === c ? th.ember : th.text;
        ctx.fillText(d.xLabels[c] ?? "", 0, 0);
        ctx.restore();
      }
    }
    ctx.fillStyle = th.textDim;
    if (d.yTitle) {
      ctx.save();
      ctx.translate(10, y0 + size / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(d.yTitle, 0, 0);
      ctx.restore();
    }
    if (d.xTitle) {
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(d.xTitle, x0 + size / 2, this.h - 2);
    }
  }

  private cellAt(px: number, py: number): { r: number; c: number } | null {
    const d = this.data;
    if (!d) return null;
    const { x0, y0, cw, ch } = this.geom;
    const c = Math.floor((px - x0) / cw);
    const r = Math.floor((py - y0) / ch);
    if (c < 0 || r < 0 || c >= d.cols || r >= d.rows) return null;
    return { r, c };
  }

  protected hover(px: number, py: number): TooltipContent | null {
    const d = this.data;
    const cell = this.cellAt(px, py);
    if (!d || !cell) return null;
    const v = d.values[cell.r * d.cols + cell.c] ?? 0;
    const q = d.yLabels?.[cell.r] ?? String(cell.r);
    const k = d.xLabels?.[cell.c] ?? String(cell.c);
    return { title: `query ${q} -> key ${k}`, rows: [{ label: "weight", value: v.toFixed(4) }] };
  }
}

// ---------------------------------------------------------------- scatter

export interface ScatterPoint {
  x: number;
  y: number;
  label: string;
  /** 0..1 intensity for colouring (e.g. token frequency). */
  weight?: number;
}

export interface ScatterOptions {
  xLabel?: string;
  yLabel?: string;
  /** Draw labels for every point when there are at most this many. */
  labelLimit?: number;
}

export class Scatter extends CanvasChart {
  private points: ScatterPoint[] = [];
  private opts: ScatterOptions;
  private pad = { l: 40, r: 16, t: 14, b: 30 };
  private px: Float32Array = new Float32Array(0);
  private py: Float32Array = new Float32Array(0);

  constructor(container: HTMLElement, opts: ScatterOptions = {}) {
    super(container);
    this.opts = opts;
  }

  setPoints(points: ScatterPoint[]): void {
    this.points = points;
    this.requestDraw();
  }

  protected render(): void {
    const ctx = this.ctx;
    const th = chartTheme();
    const pts = this.points;
    const { l, r, t, b } = this.pad;
    const plotW = this.w - l - r;
    const plotH = this.h - t - b;
    ctx.font = th.font;
    if (!pts.length) return;
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const p of pts) {
      if (p.x < xmin) xmin = p.x;
      if (p.x > xmax) xmax = p.x;
      if (p.y < ymin) ymin = p.y;
      if (p.y > ymax) ymax = p.y;
    }
    const padX = (xmax - xmin || 1) * 0.08;
    const padY = (ymax - ymin || 1) * 0.08;
    const x = makeScale(xmin - padX, xmax + padX, l, l + plotW, false, 5);
    const y = makeScale(ymin - padY, ymax + padY, t + plotH, t, false, 5);
    ctx.strokeStyle = th.grid;
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of y.ticks()) {
      const py = Math.round(y.toPx(tv)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(l, py);
      ctx.lineTo(l + plotW, py);
      ctx.stroke();
      ctx.fillStyle = th.textDim;
      ctx.fillText(fmtTick(tv), l - 6, py);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tv of x.ticks()) {
      const px = Math.round(x.toPx(tv)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, t);
      ctx.lineTo(px, t + plotH);
      ctx.stroke();
      ctx.fillStyle = th.textDim;
      ctx.fillText(fmtTick(tv), px, t + plotH + 6);
    }
    ctx.fillStyle = th.textDim;
    if (this.opts.xLabel) {
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      ctx.fillText(this.opts.xLabel, l + plotW, this.h - 2);
    }
    if (this.opts.yLabel) {
      ctx.save();
      ctx.translate(10, t + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(this.opts.yLabel, 0, 0);
      ctx.restore();
    }
    this.px = new Float32Array(pts.length);
    this.py = new Float32Array(pts.length);
    const hovered = this.nearestIndex(this.pointer?.x ?? -1, this.pointer?.y ?? -1);
    const drawLabels = pts.length <= (this.opts.labelLimit ?? 160);
    pts.forEach((p, i) => {
      const cx = x.toPx(p.x);
      const cy = y.toPx(p.y);
      this.px[i] = cx;
      this.py[i] = cy;
      const w = p.weight ?? 0.5;
      ctx.fillStyle = emberColor(0.45 + w * 0.55);
      ctx.globalAlpha = i === hovered ? 1 : 0.9;
      ctx.shadowColor = th.ember;
      ctx.shadowBlur = i === hovered ? 16 : 6;
      ctx.beginPath();
      ctx.arc(cx, cy, i === hovered ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
      if (drawLabels || i === hovered) {
        ctx.fillStyle = i === hovered ? "#fff" : th.text;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(p.label, cx + 6, cy);
      }
    });
  }

  private nearestIndex(px: number, py: number): number {
    let best = -1;
    let bd = 14 * 14;
    for (let i = 0; i < this.px.length; i++) {
      const dx = (this.px[i] ?? 0) - px;
      const dy = (this.py[i] ?? 0) - py;
      const d = dx * dx + dy * dy;
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    return best;
  }

  protected hover(px: number, py: number): TooltipContent | null {
    const i = this.nearestIndex(px, py);
    const p = this.points[i];
    if (i < 0 || !p) return null;
    return { title: `token ${p.label}`, rows: [{ label: "pc1", value: p.x.toFixed(3) }, { label: "pc2", value: p.y.toFixed(3) }] };
  }
}

/** Histogram helper for weight distributions. */
export function histogram(values: ArrayLike<number>, bins = 64): { edges: Float64Array; counts: Uint32Array; min: number; max: number; mean: number; std: number } {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sum2 = 0;
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const v = values[i] ?? 0;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sum2 += v * v;
  }
  if (n === 0) {
    min = 0;
    max = 0;
  }
  const mean = n ? sum / n : 0;
  const variance = n ? Math.max(0, sum2 / n - mean * mean) : 0;
  const span = max - min || 1;
  const counts = new Uint32Array(bins);
  const edges = new Float64Array(bins + 1);
  for (let i = 0; i <= bins; i++) edges[i] = min + (span * i) / bins;
  for (let i = 0; i < n; i++) {
    const v = values[i] ?? 0;
    let bi = Math.floor(((v - min) / span) * bins);
    if (bi >= bins) bi = bins - 1;
    if (bi < 0) bi = 0;
    counts[bi] = (counts[bi] ?? 0) + 1;
  }
  return { edges, counts, min, max, mean, std: Math.sqrt(variance) };
}
