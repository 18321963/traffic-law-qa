import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createTransport } from "./api";
import { Composer } from "./components/Composer";
import { MaterialsPanel } from "./components/MaterialsPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { Sidebar } from "./components/Sidebar";
import { StatusBar } from "./components/StatusBar";
import { StepTrace } from "./components/StepTrace";
import { TurnView } from "./components/TurnView";
import { loadSettings, saveSettings, type Settings } from "./config";
import { useConversations } from "./state/conversation";
import { useDocuments } from "./state/documents";
import { useHealth } from "./state/health";

export const USAGE =
  "问答界面：左栏会话、中栏问答、右栏处理轨迹与材料；顶部状态条与设置的开关都在 Header。";

export function App() {
  const [settings, setSettings] = useState<Settings>(() => loadSettings());
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const [showSettings, setShowSettings] = useState(false);
  const transport = useMemo(() => createTransport(() => settingsRef.current), []);
  const conversations = useConversations(transport, settings);
  const health = useHealth(transport, `${settings.mode}|${settings.healthScenario}|${settings.base}`);
  const documents = useDocuments(transport, `${settings.mode}|${settings.base}`);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const patchSettings = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      saveSettings(next);
      return next;
    });
  }, []);

  const turns = conversations.active.turns;
  const last = turns.at(-1) ?? null;
  const turnCount = turns.length;

  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [turnCount]);

  const toggleDoc = useCallback(
    (docId: string) => {
      const current = conversations.active.docIds;
      conversations.setDocIds(current.includes(docId) ? current.filter((row) => row !== docId) : [...current, docId]);
    },
    [conversations],
  );

  return (
    <div className="shell">
      <header className="header">
        <div className="header__brand">
          <h1>交通法规问答</h1>
          <p>检索 → 规划 → 复核；答案里的每条结论都能落回条文。</p>
        </div>
        <div className="header__right">
          <span className={`origin origin--${settings.mode}`}>
            {settings.mode === "live" ? "直连服务" : "示例数据"}
          </span>
          <StatusBar health={health} mode={settings.mode} />
          <button type="button" className="btn btn--quiet" onClick={() => setShowSettings((value) => !value)}>
            设置
          </button>
        </div>
      </header>

      {showSettings ? (
        <SettingsPanel settings={settings} onChange={patchSettings} onClose={() => setShowSettings(false)} />
      ) : null}

      <main className="layout">
        <Sidebar
          conversations={conversations.conversations}
          activeId={conversations.active.id}
          onSelect={conversations.selectConversation}
          onCreate={conversations.startConversation}
          onRemove={conversations.removeConversation}
        />

        <section className="stream" ref={scrollRef}>
          {turns.length === 0 ? (
            <div className="welcome">
              <h2>问一条交通法规问题</h2>
              <p>
                回答会先以草稿流式出现，复核完成后整篇换成权威版本；涉及具体地区时，服务会先问按哪里的规定作答，再继续。
                每条结论后面都跟着可点的依据。
              </p>
              <ul>
                <li>依据不足、复核没跑、服务降级这几种情况，界面会明说，不假装答案经过复核。</li>
                <li>右栏是这一轮的处理轨迹：识别地区、规划、检索、生成、复核，一步一步都在。</li>
                <li>顶部状态条盯的是服务与检索通道的健康；降级不等于不可用，它会照常回答。</li>
              </ul>
            </div>
          ) : null}
          {turns.map((turn, index) => (
            <TurnView
              key={turn.id}
              turn={turn}
              busy={conversations.busy && index === turns.length - 1}
              onResume={conversations.resume}
              onRetry={() => conversations.ask(turn.question)}
              onDismissInterrupt={conversations.dismissInterrupt}
            />
          ))}
        </section>

        <aside className="rail">
          <StepTrace
            steps={last?.steps ?? []}
            streaming={last?.status === "streaming"}
            mode={last?.mode ?? settings.answerMode}
            elapsedMs={last?.elapsedMs ?? null}
          />
          <MaterialsPanel
            documents={documents}
            activeDocIds={conversations.active.docIds}
            onToggle={toggleDoc}
          />
        </aside>
      </main>

      <footer className="footer">
        <Composer
          busy={conversations.busy}
          answerMode={settings.answerMode}
          sessionId={conversations.active.sessionId}
          docIds={conversations.active.docIds}
          onAsk={conversations.ask}
          onCancel={conversations.cancel}
          onAnswerMode={(mode) => patchSettings({ answerMode: mode })}
        />
      </footer>
    </div>
  );
}
