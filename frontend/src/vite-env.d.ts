/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_MODE?: string;
  readonly VITE_API_BASE?: string;
  readonly VITE_API_TARGET?: string;
  readonly VITE_AGENT_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
