// Headless walk of the Basalt web app. Usage: node smoke-basalt.mjs <baseUrl> [width]
import { chromium } from "playwright";
const base = process.argv[2] ?? "http://localhost:5173/basalt-db/";
const width = Number(process.argv[3] ?? 1280);
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width, height: 800 } });
const p = await ctx.newPage();
const errs = [];
p.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errs.push(`[${m.type()}] ${m.text().slice(0, 300)}`); });
p.on("pageerror", (e) => errs.push("PAGEERROR " + String(e).slice(0, 300)));

async function snap(label) {
  const t = (await p.innerText("body")).replace(/\s+/g, " ").slice(0, 700);
  const over = await p.evaluate(() => [document.scrollingElement.scrollWidth, window.innerWidth]);
  console.log(`\n== ${label} overflow=${over[0] > over[1] ? `YES ${over[0]}>${over[1]}` : "no"}\n${t}`);
}

// 1. first visit (clean storage)
await p.goto(base + "#/notebook", { waitUntil: "networkidle" });
await p.waitForTimeout(300);
await snap("notebook t+300ms");
await p.waitForTimeout(2500);
await snap("notebook t+2.8s");
// 2. reload (second visit)
await p.reload({ waitUntil: "networkidle" });
await p.waitForTimeout(2500);
await snap("notebook reload");
// 3. error surfacing in the notebook
const cases = ["SELEC 1", "INSERT INTO notes (id, title) VALUES (1, 'dup')", "SELECT 1/0, abs(-9223372036854775808)", "SELECT * FROM notes WHERE id = ?"];
for (const sql of cases) {
  await p.click("text=+ Cell");
  const tas = await p.$$("textarea");
  const ta = tas[tas.length - 1];
  await ta.fill(sql);
  await ta.press("Control+Enter");
  await p.waitForTimeout(800);
  const cards = await p.$$(".cell");
  const meta = await cards[cards.length - 1].innerText();
  console.log(`\n== cell [${sql}] -> ${meta.replace(/\s+/g, " ").slice(0, 300)}`);
}
// still alive?
await p.click("text=+ Cell");
{ const tas = await p.$$("textarea"); const ta = tas[tas.length - 1]; await ta.fill("SELECT count(*) AS n FROM notes"); await ta.press("Control+Enter"); await p.waitForTimeout(800); const cards = await p.$$(".cell"); console.log("\n== alive check ->", (await cards[cards.length - 1].innerText()).replace(/\s+/g, " ").slice(0, 200)); }

for (const r of ["plan", "storage", "datasets", "bench", "about"]) {
  await p.goto(base + "#/" + r, { waitUntil: "networkidle" });
  await p.waitForTimeout(1500);
  await snap(r);
}
// datasets: load each and run samples
await p.goto(base + "#/datasets", { waitUntil: "networkidle" });
await p.waitForTimeout(500);
const loads = await p.$$("text=Load");
for (const btn of loads) { await btn.click(); await p.waitForFunction(() => ![...document.querySelectorAll("button")].some((b) => b.disabled), null, { timeout: 120000 }); }
await snap("datasets after load");
const nb = await p.$$("button:has-text('Notebook')");
console.log("sample buttons:", nb.length);
for (let i = 0; i < nb.length; i++) {
  await p.goto(base + "#/datasets", { waitUntil: "networkidle" }); await p.waitForTimeout(300);
  const btns = await p.$$("button:has-text('Notebook')"); await btns[i].click(); await p.waitForTimeout(2500);
  const cards = await p.$$(".cell"); const txt = (await cards[cards.length - 1].innerText()).replace(/\s+/g, " ");
  console.log(`sample ${i}: ${txt.slice(0, 160)}`);
}
// plan: explain a join
await p.goto(base + "#/plan", { waitUntil: "networkidle" }); await p.waitForTimeout(500);
await p.fill("textarea", "SELECT c.c_name, sum(o.o_totalprice) FROM customer c JOIN orders o ON o.o_custkey = c.c_custkey GROUP BY c.c_custkey ORDER BY 2 DESC LIMIT 5");
await p.keyboard.press("Control+Enter"); await p.waitForTimeout(500);
const planBtn = await p.$("button:has-text('Explain')"); if (planBtn) { await planBtn.click(); await p.waitForTimeout(1500); }
await snap("plan join");
// bench
await p.goto(base + "#/bench", { waitUntil: "networkidle" }); await p.waitForTimeout(500);
await p.selectOption("select", "3");
await p.click("button:has-text('Run benchmark')");
await p.waitForFunction(() => /done:/.test(document.body.innerText), null, { timeout: 300000 });
await snap("bench done");
console.log(await p.innerText("table.bench-table"));
await p.goto(base + "#/storage", { waitUntil: "networkidle" }); await p.waitForTimeout(1500);
await snap("storage after data");
console.log("\nERRORS", errs.length); errs.slice(0, 30).forEach((e) => console.log("  ", e));
await b.close();
