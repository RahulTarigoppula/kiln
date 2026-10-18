#!/usr/bin/env node
/**
 * Headless GPU ↔ CPU parity run.
 *
 * Serves `web/dist` (build it first: `npm run build`), opens `/kiln/#/tests` in Chromium with SwiftShader
 * WebGPU flags, waits for `window.__kilnParityDone` (set by `runParityAndPublish` in
 * `src/backend/webgpu/parity.ts`) and exits non-zero on any failing case.
 *
 * Best effort: when WebGPU cannot be initialised in headless Chromium the script prints a notice and exits 0.
 *
 *   npm run test:gpu
 *   node scripts/gpu-parity.mjs --timeout 300000 --port 4173 --headed
 *
 * Requires the `playwright` devDependency and `npx playwright install chromium`.
 */

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const distDir = join(root, "web", "dist");
const BASE = "/kiln/";

const args = process.argv.slice(2);
const argValue = (flag, dflt) => {
  const i = args.indexOf(flag);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : dflt;
};
const timeoutMs = Number(argValue("--timeout", "240000"));
const port = Number(argValue("--port", "0"));
const headed = args.includes("--headed");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".map": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".txt": "text/plain; charset=utf-8",
  ".wasm": "application/wasm",
  ".safetensors": "application/octet-stream",
  ".bin": "application/octet-stream",
};

async function exists(p) {
  try {
    return (await stat(p)).isFile();
  } catch {
    return false;
  }
}

function serveDist() {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://localhost");
      let pathname = decodeURIComponent(url.pathname);
      if (!pathname.startsWith(BASE)) {
        res.writeHead(302, { Location: BASE });
        res.end();
        return;
      }
      pathname = pathname.slice(BASE.length);
      let file = normalize(join(distDir, pathname));
      if (!file.startsWith(distDir)) {
        res.writeHead(403).end();
        return;
      }
      if (!(await exists(file))) file = join(distDir, "index.html"); // SPA fallback
      const body = await readFile(file);
      res.writeHead(200, {
        "Content-Type": MIME[extname(file)] ?? "application/octet-stream",
        "Cache-Control": "no-store",
      });
      res.end(body);
    } catch (err) {
      res.writeHead(500).end(String(err));
    }
  });
  return new Promise((resolveServer) => {
    server.listen(port, "127.0.0.1", () => resolveServer(server));
  });
}

function formatTable(results) {
  const rows = results.map((r) => [
    r.pass ? "PASS" : "FAIL",
    r.op,
    r.case,
    Number.isFinite(r.maxAbs) ? r.maxAbs.toExponential(2) : String(r.maxAbs),
    Number.isFinite(r.maxRel) ? r.maxRel.toExponential(2) : String(r.maxRel),
    r.threshold.toExponential(1),
    `${r.ms.toFixed(1)}ms`,
    r.error ?? "",
  ]);
  const widths = rows.reduce((w, row) => row.map((c, i) => Math.max(w[i] ?? 0, c.length)), []);
  return rows.map((row) => row.map((c, i) => c.padEnd(widths[i])).join("  ")).join("\n");
}

async function main() {
  if (!(await exists(join(distDir, "index.html")))) {
    console.error(`gpu-parity: ${distDir}/index.html not found. Run "npm run build" first.`);
    process.exit(1);
  }
  let chromium;
  try {
    ({ chromium } = await import("playwright"));
  } catch {
    console.error("gpu-parity: playwright is not installed. Run: npm i -D playwright && npx playwright install chromium");
    process.exit(1);
  }

  const server = await serveDist();
  const actualPort = server.address().port;
  const url = `http://127.0.0.1:${actualPort}${BASE}#/tests`;
  console.log(`gpu-parity: serving ${distDir} at ${url}`);

  let browser;
  let exitCode = 0;
  try {
    browser = await chromium.launch({
      headless: !headed,
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,WebGPU",
        "--use-angle=swiftshader",
        "--use-vulkan=swiftshader",
        "--ignore-gpu-blocklist",
        "--enable-gpu-rasterization",
        "--no-sandbox",
      ],
    });
    const page = await browser.newPage();
    page.on("console", (msg) => {
      const type = msg.type();
      if (type === "error" || type === "warning") console.log(`[browser:${type}] ${msg.text()}`);
    });
    page.on("pageerror", (err) => console.log(`[browser:pageerror] ${err.message}`));

    await page.goto(url, { waitUntil: "load" });

    const gpuProbe = await page.evaluate(async () => {
      if (!("gpu" in navigator)) return { available: false, reason: "navigator.gpu missing" };
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { available: false, reason: "requestAdapter() returned null" };
        return { available: true };
      } catch (e) {
        return { available: false, reason: String(e) };
      }
    });
    if (!gpuProbe.available) {
      console.log(`gpu-parity: WebGPU unavailable in headless Chromium (${gpuProbe.reason}). Skipping (exit 0).`);
      return;
    }

    let done;
    try {
      done = await page.waitForFunction(() => globalThis.__kilnParityDone, null, { timeout: timeoutMs, polling: 500 });
    } catch {
      console.error(`gpu-parity: timed out after ${timeoutMs} ms waiting for window.__kilnParityDone`);
      exitCode = 1;
      return;
    }
    const result = await done.jsonValue();
    if (result.unavailable) {
      console.log("gpu-parity: the page reported WebGPU as unavailable:");
      console.log(formatTable(result.results));
      console.log("Skipping (exit 0).");
      return;
    }
    if (result.info) console.log(`gpu-parity: adapter ${result.info.vendor} / ${result.info.architecture} ${result.info.description}`);
    console.log(formatTable(result.results));
    const failures = result.results.filter((r) => !r.pass);
    console.log(`\ngpu-parity: ${result.results.length - failures.length}/${result.results.length} cases passed`);
    if (failures.length) {
      console.error(`gpu-parity: ${failures.length} failing case(s)`);
      exitCode = 1;
    }
  } finally {
    await browser?.close().catch(() => {});
    server.close();
    process.exit(exitCode);
  }
}

main().catch((err) => {
  console.error("gpu-parity: unexpected error", err);
  process.exit(1);
});
