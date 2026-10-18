import "./styles/index.css";
import { h } from "./lib/dom";
import { ensureGpu } from "./lib/gpu";
import { footer, topNav } from "./components/nav";
import { startRouter } from "./router";

const app = document.getElementById("app");
if (!app) throw new Error("#app missing");

const outlet = h("main", { class: "main container", id: "main" });
app.replaceChildren(topNav(), outlet, footer());

// Kick off WebGPU right away so the device badge resolves before the user clicks anything.
void ensureGpu();

startRouter(outlet);
