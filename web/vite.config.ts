import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
// GitHub Pages serves the site at https://rahultarigoppula.github.io/kiln/
export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  base: process.env.BASE_PATH ?? "/kiln/",
  build: { outDir: "dist", emptyOutDir: true, target: "es2022", sourcemap: true },
  worker: { format: "es" },
  assetsInclude: ["**/*.wgsl", "**/*.bin", "**/*.safetensors"],
});
