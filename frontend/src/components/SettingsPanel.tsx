import type { ApiMode, AnswerMode, Settings } from "../config";
import { HEALTH_SCENARIOS, SCENARIOS } from "../api/mock";

type Props = {
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
  onClose: () => void;
};

export function SettingsPanel({ settings, onChange, onClose }: Props) {
  const live = settings.mode === "live";
  return (
    <aside className="settings">
      <header className="settings__head">
        <h2>设置</h2>
        <button type="button" className="btn btn--quiet" onClick={onClose}>
          收起
        </button>
      </header>

      <section className="settings__group">
        <h3>数据来源</h3>
        <div className="segmented" role="group" aria-label="数据来源">
          <button type="button" className={!live ? "is-active" : ""} onClick={() => onChange({ mode: "mock" as ApiMode })}>
            内置示例数据
          </button>
          <button type="button" className={live ? "is-active" : ""} onClick={() => onChange({ mode: "live" as ApiMode })}>
            直连服务
          </button>
        </div>
        <label className="field">
          <span>服务基地址</span>
          <input
            type="text"
            value={settings.base}
            placeholder="留空＝同源（开发时走 vite 代理）"
            onChange={(event) => onChange({ base: event.target.value })}
          />
        </label>
        <label className="field">
          <span>X-API-Key</span>
          <input
            type="password"
            value={settings.apiKey}
            placeholder="留空＝不发鉴权头"
            onChange={(event) => onChange({ apiKey: event.target.value })}
          />
        </label>
        <p className="settings__note">
          留空时前端不发 X-API-Key，也不敢把 key 写进构建产物以外的任何地方；这里的值只存在本机浏览器。
        </p>
      </section>

      {!live ? (
        <section className="settings__group">
          <h3>问答演练场景</h3>
          <select value={settings.scenario} onChange={(event) => onChange({ scenario: event.target.value })}>
            {SCENARIOS.map((row) => (
              <option key={row.id} value={row.id}>
                {row.label}
              </option>
            ))}
          </select>
          <p className="settings__note">{SCENARIOS.find((row) => row.id === settings.scenario)?.hint ?? ""}</p>
          <h3>状态条演练场景</h3>
          <select value={settings.healthScenario} onChange={(event) => onChange({ healthScenario: event.target.value })}>
            {HEALTH_SCENARIOS.map((row) => (
              <option key={row.id} value={row.id}>
                {row.label}
              </option>
            ))}
          </select>
          <p className="settings__note">{HEALTH_SCENARIOS.find((row) => row.id === settings.healthScenario)?.hint ?? ""}</p>
        </section>
      ) : null}

      <section className="settings__group">
        <h3>问答</h3>
        <label className="field">
          <span>召回条数 top_k</span>
          <input
            type="number"
            min={1}
            max={20}
            value={settings.topK ?? ""}
            placeholder="留空＝服务默认"
            onChange={(event) => {
              const value = Number(event.target.value);
              onChange({ topK: Number.isFinite(value) && value > 0 ? Math.min(20, Math.round(value)) : null });
            }}
          />
        </label>
        <div className="segmented" role="group" aria-label="回答方式">
          <button
            type="button"
            className={settings.answerMode === "stream" ? "is-active" : ""}
            onClick={() => onChange({ answerMode: "stream" as AnswerMode })}
          >
            流式问答
          </button>
          <button
            type="button"
            className={settings.answerMode === "once" ? "is-active" : ""}
            onClick={() => onChange({ answerMode: "once" as AnswerMode })}
          >
            一次性问答
          </button>
        </div>
        <p className="settings__note">
          一次性走 /qa（与 /qa/resume），一次典型十几秒到几十秒；流式走 /qa/stream（与 /qa/resume/stream）。
        </p>
      </section>
    </aside>
  );
}
