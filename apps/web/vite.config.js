import { defineConfig } from "vite";

export default defineConfig({
  server: {
    headers: {
      "Cache-Control": "no-store",
    },
  },
  build: {
    rollupOptions: {
      output: {
        // Peel third-party code (React et al.) into a separate, rarely-changing
        // chunk so the app bundle stays under Vite's size warning and the vendor
        // chunk caches across app-only deploys.
        manualChunks(id) {
          if (id.includes("node_modules")) {
            return "vendor";
          }
        },
      },
    },
  },
});
