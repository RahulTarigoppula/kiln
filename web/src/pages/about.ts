/** About - architecture overview and links. */
import { h, svg, KILN_MARK } from "../lib/dom";
import { ARCH_URL, REPO_URL } from "../components/nav";
import { card } from "../components/ui";

const LAYERS: [string, string][] = [
  ["src/tensor", "dtype, shape utilities (broadcast, strides, index math), the device-agnostic Tensor handle, seeded xoshiro128** RNG"],
  ["src/backend", "Backend interface + CPU backend (typed-array kernels) + WebGPU backend (WGSL kernels, buffer pool, dispatch batching)"],
  ["src/autograd", "Function nodes, tape, backward(); forward + backward for every differentiable op"],
  ["src/nn", "Module base, Linear, Embedding, LayerNorm, Dropout, GELU, CausalSelfAttention, TransformerBlock"],
  ["src/optim", "SGD, AdamW (fused GPU kernel), gradient clipping, LR schedulers"],
  ["src/data", "Char tokenizer, byte-level BPE (train + encode/decode), token dataset -> random (x, y) batches"],
  ["src/models", "GPT (config, forward, loss, generate with temperature / top-k / top-p), safetensors checkpoints"],
  ["src/train", "Training loop with callbacks: onStep, onEval, onSample; tokens/s; checkpoint cadence"],
  ["web/", "This app: Studio, Playground, Inspect, Bench, Tests - vanilla TS, a tiny signal store, canvas charts"],
];

export function mount(root: HTMLElement): () => void {
  root.appendChild(
    h(
      "div",
      { class: "about-hero" },
      svg(KILN_MARK, "hero-glyph"),
      h("div", null, h("div", { class: "page-head__eyebrow" }, "About"), h("h1", null, "Kiln"), h("p", { class: "page-head__lede" }, "A deep-learning framework written from scratch in TypeScript that trains a GPT live in a browser tab - on your GPU, through WebGPU.")),
    ),
  );

  root.appendChild(
    h(
      "div",
      { class: "grid grid--side-r" },
      card(null, [
        h(
          "div",
          { class: "prose" },
          h("h2", null, "What it is"),
          h(
            "p",
            null,
            "Kiln is a tensor library with a CPU backend and a WebGPU backend (hand-written WGSL kernels), reverse-mode autograd, neural-network modules, optimisers, a byte-level BPE tokenizer, a decoder-only transformer and a training loop. Zero runtime dependencies: the only things in the bundle are the code in this repository and your browser.",
          ),
          h("h2", null, "How it executes"),
          h(
            "p",
            null,
            "Execution is eager. Every op dispatches immediately to the backend: on the CPU results are available synchronously; on WebGPU ops enqueue work and return a tensor whose storage is a GPUBuffer. Only ",
            h("code", null, "await tensor.data()"),
            " forces a readback. The backend batches dispatches into command encoders and submits on readback or after 64 queued dispatches - PyTorch-like ergonomics without a graph compiler.",
          ),
          h("h2", null, "Why it is trustworthy"),
          h(
            "p",
            null,
            "Both backends implement one Backend interface, so the autograd layer is written once. The Tests page runs every kernel on both devices with random inputs and compares them (1e-4; 1e-3 for matmul with large K). CPU ops are checked against golden constants and central-difference gradient checks in vitest.",
          ),
          h("h2", null, "Design identity"),
          h(
            "p",
            null,
            "'The firing room': a near-black background, a warm ember accent and soft ash greys. Space Grotesk for display, Inter for UI, JetBrains Mono for numerals. The device badge in the top bar pulses while a job runs on the GPU.",
          ),
          h(
            "p",
            null,
            h("a", { href: ARCH_URL, target: "_blank", rel: "noopener" }, "Read docs/ARCHITECTURE.md on GitHub ->"),
            " · ",
            h("a", { href: REPO_URL, target: "_blank", rel: "noopener" }, "Source"),
          ),
        ),
      ]),
      card("Layering", [h("div", { class: "layer-list" }, ...LAYERS.map(([k, v]) => h("div", null, h("b", null, k), h("span", null, v))))], { sub: "bottom -> top" }),
    ),
  );
  return () => undefined;
}
