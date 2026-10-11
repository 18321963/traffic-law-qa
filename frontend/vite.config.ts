import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const target = env.VITE_API_TARGET || "http://127.0.0.1:8001";
  const ragTarget = env.VITE_RAG_TARGET || "http://127.0.0.1:8000";
  const proxy = () => ({ target, changeOrigin: true });
  const ragProxy = () => ({
    target: ragTarget,
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/rag/, ""),
  });
  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: false,
      proxy: {
        "/health": proxy(),
        "/qa": proxy(),
        "/laws": proxy(),
        "/documents": proxy(),
        "/sessions": proxy(),
        "/rag": ragProxy(),
      },
    },
    preview: { port: 4173 },
  };
});
