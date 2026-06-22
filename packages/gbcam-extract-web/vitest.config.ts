import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    // Default node environment is fine: detectFrames itself only orchestrates
    // pure-JS sheet/individual loaders. Tests that need a DOM/canvas can
    // override per-file with `// @vitest-environment jsdom`.
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
});
