# Contributing to Kiln

Kiln is a from-scratch framework with no runtime dependencies, so the bar for a change is that it keeps the two
backends in agreement and the tests green. Read `docs/ARCHITECTURE.md` first; it is the contract every module is
built against, and a change that disagrees with it must update the document in the same commit.

## Setup

```sh
npm ci
npm run typecheck      # strict TypeScript, no emit
npm test               # vitest: tensor ops, gradchecks, nn, optim, tokenizers, checkpoints, trainer, memory
npm run lint:text      # bans typographic dashes/quotes/arrows in tracked text files
npm run dev            # Vite dev server for the web app at http://localhost:5173/kiln/
npm run build          # typecheck + production build to web/dist
npm run test:gpu       # headless GPU parity via Playwright (needs `npx playwright install chromium`)
```

Node 22 or newer. WebGPU work needs a Chromium-based browser; the Tests page runs the parity suite on your GPU.

## Rules of the house

- Strict TypeScript. `noUncheckedIndexedAccess` is on, so index typed arrays with `as number` or a check.
- Zero runtime dependencies in `src/` and `web/src/`. Dev dependencies are fine.
- Eager execution and refcounted buffers: anything that allocates tensors in a loop must run inside `tidy()`, and
  long-lived tensors (parameters, optimizer state, caches) are `escape()`d by their owner. `tests/memory.test.ts`
  asserts the live buffer count stays flat across training steps; keep it that way.
- Both backends implement `src/backend/types.ts`. A new op needs: the CPU kernel, the WGSL kernel, a parity case in
  `src/backend/webgpu/parity.ts`, an autograd definition in `src/autograd/ops.ts`, a golden-value test in
  `tests/ops.test.ts` and a gradcheck in `tests/gradcheck.test.ts`.
- Kernel semantics that both backends must share (tie-breaking, identities, RNG, rounding of fused kernels) are
  listed in `docs/ARCHITECTURE.md`. Match the CPU backend exactly; it is the reference.
- No UI frameworks in `web/`. Pages are vanilla TypeScript with the small signal store in `web/src/lib/signals.ts`.
- Plain ASCII punctuation in prose and strings (hyphens, straight quotes, "->" or words instead of arrows).
  `npm run lint:text` enforces it and runs in CI.

## Pull requests

Keep PRs focused. Describe what changed and how you verified it (which tests, which page in the web app, which
GPU). If a change affects numerics, include the parity numbers from the Tests page.
