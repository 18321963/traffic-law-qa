import { useEffect, useMemo, useRef, useState } from "react";
import { App as AntdApp } from "antd";
import { createTransport } from "./api";
import { ChatHeader } from "./components/ChatHeader";
import { Composer, type Material } from "./components/Composer";
import { RightRail, type RailTab } from "./components/RightRail";
import { SettingsPanel } from "./components/SettingsPanel";
import { Sidebar } from "./components/Sidebar";
import { TurnView } from "./components/TurnView";
import { Welcome } from "./components/Welcome";
import { loadSettings, saveSettings, type Settings } from "./config";
import { turnErrorOf, useConversations } from "./state/conversation";
import { useDocuments } from "./state/documents";
import { useHealth } from "./state/health";
import styles from "./App.module.css";

export const USAGE =
  "问答界面：左栏会话，中栏问答流，右栏依据 / 轨迹 / 材料；状态条与设置入口在页头，答案出完后右栏自动切到依据。";

const SAMPLES = ["闯红灯一次记多少分？", "深圳电动自行车载人怎么处理？", "醉酒驾驶机动车有什么处罚？"];

export function App() {
  const { message } = AntdApp.useApp();
  const [settings, setSettings] = useState<Settings>(() => loadSettings());
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const [showSettings, setShowSettings] = useState(false);
  const [railTab, setRailTab] = useState<RailTab>("materials");
  const [railTurnId, setRailTurnId] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<number | null>(null);
  const autoTabRef = useRef(false);
  const userTabRef = useRef(false);
  const prevCountRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const transport = useMemo(() => createTransport(() => settingsRef.current), []);
  const conversations = useConversations(transport, settings);
  const health = useHealth(transport, `${settings.mode}|${settings.healthScenario}|${settings.base}`);
  const documents = useDocuments(transport, `${settings.mode}|${settings.base}`);

  const patchSettings = (patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      saveSettings(next);
      return next;
    });
  };

  const active = conversations.active;
  const turns = active.turns;
  const last = turns.at(-1) ?? null;
  const turnCount = turns.length;
  const lastKey = last ? `${last.id}|${last.status}|${last.draft.length}` : "none";

  useEffect(() => {
    setRailTurnId(null);
    setHighlight(null);
  }, [active.id]);

  useEffect(() => {
    if (autoTabRef.current || userTabRef.current) return;
    if (!last || last.status !== "done") return;
    autoTabRef.current = true;
    setRailTab("evidence");
  }, [last]);

  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    const fresh = turnCount !== prevCountRef.current;
    prevCountRef.current = turnCount;
    const gap = node.scrollHeight - node.scrollTop - node.clientHeight;
    if (!fresh && gap > 240) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [turnCount, lastKey]);

  const askQuestion = (question: string) => {
    setRailTurnId(null);
    setHighlight(null);
    conversations.ask(question);
  };

  const removeConversation = (id: string) => {
    conversations.deleteConversation(id).catch((exc) => {
      message.error(`没删成：${turnErrorOf(exc).message}`);
    });
  };

  const cite = (turnId: string, index: number | null) => {
    userTabRef.current = true;
    setRailTurnId(turnId);
    setHighlight(index);
    setRailTab("evidence");
  };

  const selectTab = (tab: RailTab) => {
    userTabRef.current = true;
    setRailTab(tab);
  };

  const toggleDoc = (docId: string) => {
    const current = active.docIds;
    conversations.setDocIds(current.includes(docId) ? current.filter((row) => row !== docId) : [...current, docId]);
  };

  const attach = (file: File) => {
    documents.upload(file, "session").then((record) => {
      if (!record) return;
      conversations.appendDocuments([record.doc_id]);
      message.success(`已挂上「${record.display_name ?? file.name}」`);
    });
  };

  const materials = useMemo<Material[]>(
    () =>
      active.docIds.map((docId) => ({
        docId,
        name: documents.rows.find((row) => row.doc_id === docId)?.display_name ?? docId,
      })),
    [active.docIds, documents.rows],
  );

  useEffect(() => {
    if (documents.loading || documents.rows.length === 0) return;
    const known = new Set(documents.rows.map((row) => row.doc_id));
    const kept = active.docIds.filter((id) => known.has(id));
    if (kept.length !== active.docIds.length) conversations.setDocIds(kept);
  }, [documents.loading, documents.rows, active.docIds, conversations]);

  const focused = (railTurnId ? turns.find((row) => row.id === railTurnId) : undefined) ?? last;

  return (
    <div className={styles.shell}>
      <Sidebar
        conversations={conversations.conversations}
        activeId={active.id}
        mode={settings.mode}
        onSelect={conversations.selectConversation}
        onCreate={conversations.startConversation}
        onRemove={removeConversation}
        onOpenSettings={() => setShowSettings(true)}
      />

      <div className={styles.main}>
        <ChatHeader
          title={active.title}
          mode={settings.mode}
          health={health}
          onOpenSettings={() => setShowSettings(true)}
        />

        <section className={`${styles.stream} scrollbar`} ref={scrollRef}>
          <div className={styles.thread}>
            {turns.length === 0 ? <Welcome samples={SAMPLES} onAsk={askQuestion} /> : null}
            {turns.map((turn, index) => (
              <TurnView
                key={turn.id}
                turn={turn}
                busy={conversations.busy && index === turns.length - 1}
                onResume={conversations.resume}
                onRetry={() => conversations.ask(turn.question)}
                onDismissInterrupt={conversations.dismissInterrupt}
                onFocusEvidence={cite}
              />
            ))}
          </div>
        </section>

        <footer className={styles.footer}>
          <div className={styles.composer}>
            <Composer
              busy={conversations.busy}
              answerMode={settings.answerMode}
              sessionId={active.sessionId}
              materials={materials}
              onRemoveDoc={toggleDoc}
              onAsk={askQuestion}
              onCancel={conversations.cancel}
              onAttach={attach}
            />
          </div>
        </footer>
      </div>

      <RightRail
        tab={railTab}
        onTab={selectTab}
        turn={focused}
        highlight={highlight}
        documents={documents}
        activeDocIds={active.docIds}
        onToggleDoc={toggleDoc}
      />

      <SettingsPanel
        open={showSettings}
        settings={settings}
        onChange={patchSettings}
        onClose={() => setShowSettings(false)}
      />
    </div>
  );
}
