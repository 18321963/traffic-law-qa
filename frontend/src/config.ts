export const USAGE =
  "运行期设置：初值取自 VITE_* 环境变量，改动落 localStorage（键 tlq.frontend.settings），刷新后仍在；联调时不必重新构建。";

export type ApiMode = "mock" | "live";
export type AnswerMode = "stream" | "once";

export type Settings = {
  mode: ApiMode;
  base: string;
  apiKey: string;
  answerMode: AnswerMode;
  scenario: string;
  healthScenario: string;
  topK: number | null;
};

export const STORAGE_KEY = "tlq.frontend.settings";
export const SESSION_KEY = "tlq.frontend.sessions";

export const DEFAULT_SCENARIO = "auto";

function envString(key: string, fallback: string): string {
  const raw = import.meta.env[key as keyof ImportMetaEnv] as string | undefined;
  return typeof raw === "string" && raw.trim() !== "" ? raw.trim() : fallback;
}

export function defaultSettings(): Settings {
  return {
    mode: envString("VITE_API_MODE", "mock") === "live" ? "live" : "mock",
    base: envString("VITE_API_BASE", ""),
    apiKey: envString("VITE_AGENT_API_KEY", ""),
    answerMode: "stream",
    scenario: DEFAULT_SCENARIO,
    healthScenario: "ok",
    topK: null,
  };
}

export function loadSettings(): Settings {
  const base = defaultSettings();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return base;
    const saved = JSON.parse(raw) as Partial<Settings>;
    return {
      mode: saved.mode === "live" || saved.mode === "mock" ? saved.mode : base.mode,
      base: typeof saved.base === "string" ? saved.base : base.base,
      apiKey: typeof saved.apiKey === "string" ? saved.apiKey : base.apiKey,
      answerMode: saved.answerMode === "once" ? "once" : "stream",
      scenario: typeof saved.scenario === "string" && saved.scenario ? saved.scenario : base.scenario,
      healthScenario:
        typeof saved.healthScenario === "string" && saved.healthScenario ? saved.healthScenario : base.healthScenario,
      topK: typeof saved.topK === "number" && saved.topK > 0 ? saved.topK : null,
    };
  } catch {
    return base;
  }
}

export function saveSettings(settings: Settings): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    return;
  }
}
