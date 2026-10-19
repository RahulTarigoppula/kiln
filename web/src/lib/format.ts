/** Number / duration / byte formatting for the UI (tabular, compact). */

export function fmtNumber(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return n > 0 ? "∞" : n < 0 ? "-∞" : "-";
  return n.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

/** 1234 -> "1.23k", 1.2e6 -> "1.2M". */
export function fmtCompact(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return "-";
  const abs = Math.abs(n);
  const units: [number, string][] = [
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "k"],
  ];
  for (const [v, s] of units) {
    if (abs >= v) return trimZeros((n / v).toFixed(digits)) + s;
  }
  return trimZeros(n.toFixed(abs < 10 ? digits : 0));
}

/** Loss-style fixed decimals. */
export function fmtFixed(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "-";
  return n.toFixed(digits);
}

/** Scientific-ish for learning rates: 3e-4 -> "3.0e-4". */
export function fmtSci(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "-";
  if (n === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(n)));
  if (exp >= -2 && exp <= 3) return trimZeros(n.toFixed(Math.max(0, 3 - exp)));
  const mant = n / 10 ** exp;
  return `${mant.toFixed(digits)}e${exp}`;
}

export function fmtPercent(p: number, digits = 1): string {
  if (!Number.isFinite(p)) return "-";
  return `${(p * 100).toFixed(digits)}%`;
}

export function fmtBytes(bytes: number, digits = 1): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${i === 0 ? v.toFixed(0) : v.toFixed(digits)} ${units[i] ?? "B"}`;
}

/** 1234 ms -> "1.23 s", 95000 -> "1m 35s", 3.7e6 -> "1h 02m". */
export function fmtDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "-";
  if (ms < 1000) return `${ms < 10 ? ms.toFixed(2) : ms < 100 ? ms.toFixed(1) : ms.toFixed(0)} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  const m = Math.floor(s / 60);
  const rs = Math.round(s - m * 60);
  if (m < 60) return `${m}m ${String(rs).padStart(2, "0")}s`;
  const hs = Math.floor(m / 60);
  const rm = m - hs * 60;
  return `${hs}h ${String(rm).padStart(2, "0")}m`;
}

/** Milliseconds with adaptive precision for step times. */
export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "-";
  if (ms < 1) return `${ms.toFixed(3)}`;
  if (ms < 10) return `${ms.toFixed(2)}`;
  if (ms < 100) return `${ms.toFixed(1)}`;
  return `${ms.toFixed(0)}`;
}

/** Throughput: 1.23e9 FLOP/s -> "1.23 GFLOPS". */
export function fmtRate(perSec: number, unit: string, digits = 2): string {
  if (!Number.isFinite(perSec)) return "-";
  return `${fmtCompact(perSec, digits)} ${unit}`;
}

export function fmtInt(n: number): string {
  if (!Number.isFinite(n)) return "-";
  return Math.round(n).toLocaleString("en-US");
}

/** Short axis tick label. */
export function fmtTick(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1e4) return fmtCompact(v, 1);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return trimZeros(v.toFixed(2));
  if (abs >= 0.01) return trimZeros(v.toFixed(3));
  return v.toExponential(0);
}

function trimZeros(s: string): string {
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}

/** Render control characters / whitespace visibly for token labels. */
export function visibleToken(s: string): string {
  return s
    .replace(/\n/g, "⏎")
    .replace(/\t/g, "⇥")
    .replace(/ /g, "␣")
    .replace(/\r/g, "␍");
}
