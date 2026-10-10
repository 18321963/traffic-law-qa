import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SESSION_KEY, type AnswerMode, type Settings } from "../config";
import { isAbort } from "../api/http";
import type { Transport } from "../api/transport";
import {
  ApiError,
  type DonePayload,
  type InterruptValue,
  type StepEvent,
  type StreamEvent,
} from "../api/types";

export const USAGE =
  "会话状态机：一次提问 = 一个 turn。turn 走 streaming → interrupted | done | error；done 到达时权威答案整篇替换草稿，替换与否记在 replaced 上。";

export type TurnStatus = "streaming" | "interrupted" | "done" | "error" | "cancelled";

export type TurnError = {
  message: string;
  status: number | null;
  kind: string;
  retryAfter: number | null;
};

export type ClarifyOutcome = { kind: "resumed" | "dismissed"; region?: string };

export type Turn = {
  id: string;
  question: string;
  status: TurnStatus;
  steps: StepEvent[];
  draft: string;
  answer: DonePayload | null;
  interrupt: { sessionId: string | null; value: InterruptValue | null; id: string | null } | null;
  clarifyOutcome: ClarifyOutcome | null;
  resumed: boolean;
  error: TurnError | null;
  replaced: boolean;
  mode: AnswerMode;
  startedAt: number;
  elapsedMs: number | null;
};

export type Conversation = {
  id: string;
  sessionId: string | null;
  title: string;
  createdAt: number;
  turns: Turn[];
  docIds: string[];
};

const HISTORY_LIMIT = 20;
const TURN_LIMIT = 40;
const SAVE_CHARS = 3000000;

function newId(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function emptyConversation(): Conversation {
  return { id: newId("conv"), sessionId: null, title: "新会话", createdAt: Date.now(), turns: [], docIds: [] };
}

function titleOf(question: string): string {
  const clean = question.replace(/\s+/g, " ").trim();
  return clean.length > 18 ? `${clean.slice(0, 18)}…` : clean || "新会话";
}

function normalize(text: string): string {
  return text.replace(/\s+/g, "");
}

export function turnErrorOf(exc: unknown): TurnError {
  if (exc instanceof ApiError) {
    return { message: exc.detail, status: exc.kind === "http" ? exc.status : null, kind: exc.kind, retryAfter: exc.retryAfter };
  }
  const message = exc instanceof Error ? exc.message : String(exc);
  return { message, status: null, kind: "unknown", retryAfter: null };
}

function loadConversations(): Conversation[] {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    if (!raw) return [];
    const rows = JSON.parse(raw) as Conversation[];
    if (!Array.isArray(rows)) return [];
    return rows.slice(0, HISTORY_LIMIT).map((row) => ({
      ...row,
      turns: (row.turns || []).map((turn) =>
        turn.status === "streaming" ? { ...turn, status: "cancelled" as TurnStatus } : turn,
      ),
    }));
  } catch {
    return [];
  }
}

export type ConversationApi = {
  conversations: Conversation[];
  active: Conversation;
  busy: boolean;
  ask: (question: string) => void;
  resume: (region: string) => void;
  dismissInterrupt: () => void;
  cancel: () => void;
  startConversation: () => void;
  selectConversation: (id: string) => void;
  removeConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  setDocIds: (ids: string[]) => void;
  appendDocuments: (ids: string[]) => void;
};

export function useConversations(transport: Transport, settings: Settings): ConversationApi {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const saved = loadConversations();
    return saved.length > 0 ? saved : [emptyConversation()];
  });
  const [activeId, setActiveId] = useState<string>(() => conversations[0]?.id ?? "");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const active = useMemo(
    () => conversations.find((row) => row.id === activeId) ?? conversations[0] ?? emptyConversation(),
    [conversations, activeId],
  );

  useEffect(() => {
    const rows = conversations.slice(0, HISTORY_LIMIT);
    let keep = rows.length;
    let text = JSON.stringify(rows);
    while (text.length > SAVE_CHARS && keep > 1) {
      keep -= 1;
      text = JSON.stringify(rows.slice(0, keep));
    }
    try {
      window.localStorage.setItem(SESSION_KEY, text);
    } catch {
      return;
    }
  }, [conversations]);

  const patchTurn = useCallback((convId: string, turnId: string, patch: (turn: Turn) => Turn) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === convId
          ? { ...conv, turns: conv.turns.map((turn) => (turn.id === turnId ? patch(turn) : turn)) }
          : conv,
      ),
    );
  }, []);

  const patchConversation = useCallback((convId: string, patch: (conv: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((conv) => (conv.id === convId ? patch(conv) : conv)));
  }, []);

  const finish = useCallback(
    (convId: string, turnId: string, controller: AbortController, patch: (turn: Turn) => Turn) => {
      patchTurn(convId, turnId, (turn) => ({ ...patch(turn), elapsedMs: Date.now() - turn.startedAt }));
      if (abortRef.current === controller) {
        abortRef.current = null;
        setBusy(false);
      }
    },
    [patchTurn],
  );

  const run = useCallback(
    async (
      convId: string,
      turnId: string,
      stream: AsyncGenerator<StreamEvent>,
      sessionId: string | null,
      controller: AbortController,
    ) => {
      let terminal = false;
      try {
        for await (const event of stream) {
          if (event.type === "step") {
            const step = event.data;
            patchTurn(convId, turnId, (turn) => ({ ...turn, steps: [...turn.steps, step] }));
            continue;
          }
          if (event.type === "delta") {
            const text = String(event.data.text ?? "");
            if (text !== "") {
              patchTurn(convId, turnId, (turn) => ({ ...turn, draft: turn.draft + text }));
            }
            continue;
          }
          if (event.type === "interrupt") {
            terminal = true;
            const nextSession = event.data.session_id ?? sessionId;
            patchTurn(convId, turnId, (turn) => ({
              ...turn,
              status: "interrupted",
              interrupt: {
                sessionId: nextSession,
                value: event.data.interrupt?.value ?? null,
                id: event.data.interrupt?.id ?? null,
              },
            }));
            if (nextSession) {
              patchConversation(convId, (conv) => ({ ...conv, sessionId: nextSession }));
            }
            continue;
          }
          if (event.type === "error") {
            terminal = true;
            const partial = String(event.data.partial ?? "");
            patchTurn(convId, turnId, (turn) => ({
              ...turn,
              status: "error",
              draft: turn.draft === "" ? partial : turn.draft,
              error: {
                message: String(event.data.message ?? "服务未给出原因"),
                status: null,
                kind: "frame",
                retryAfter: null,
              },
            }));
            continue;
          }
          terminal = true;
          const payload = event.data;
          const nextSession = payload.session_id ?? sessionId;
          patchTurn(convId, turnId, (turn) => ({
            ...turn,
            status: "done",
            answer: payload,
            replaced: normalize(payload.answer ?? "") !== normalize(turn.draft),
          }));
          if (nextSession) {
            patchConversation(convId, (conv) => ({ ...conv, sessionId: nextSession }));
          }
        }
        if (!terminal) {
          patchTurn(convId, turnId, (turn) => ({
            ...turn,
            status: "error",
            error: { message: "流已经结束，但没有收到 done / interrupt / error 收尾帧", status: null, kind: "truncated", retryAfter: null },
          }));
        }
      } catch (exc) {
        if (isAbort(exc)) {
          patchTurn(convId, turnId, (turn) => ({ ...turn, status: "cancelled" }));
        } else {
          patchTurn(convId, turnId, (turn) => ({ ...turn, status: "error", error: turnErrorOf(exc) }));
        }
      } finally {
        finish(convId, turnId, controller, (turn) => turn);
      }
    },
    [finish, patchConversation, patchTurn],
  );

  const startTurn = useCallback(
    (question: string, kind: "ask" | "resume", region?: string) => {
      if (busy) return;
      const conv = active;
      const turn: Turn = {
        id: newId("turn"),
        question,
        status: "streaming",
        steps: [],
        draft: "",
        answer: null,
        interrupt: null,
        clarifyOutcome: null,
        resumed: kind === "resume",
        error: null,
        replaced: false,
        mode: settingsRef.current.answerMode,
        startedAt: Date.now(),
        elapsedMs: null,
      };
      patchConversation(conv.id, (row) => ({
        ...row,
        title: row.turns.length === 0 ? titleOf(question) : row.title,
        turns: [...row.turns.slice(-TURN_LIMIT + 1), turn],
      }));
      setBusy(true);
      const settings = settingsRef.current;
      const sessionId = conv.sessionId;
      const controller = new AbortController();
      abortRef.current = controller;
      const generator =
        kind === "ask"
          ? settings.answerMode === "stream"
            ? transport.ask(
                { question, top_k: settings.topK, doc_ids: conv.docIds, session_id: sessionId },
                controller.signal,
              )
            : onceAsStream(
                transport,
                { question, top_k: settings.topK, doc_ids: conv.docIds, session_id: sessionId },
                region,
                "",
                controller.signal,
              )
          : settings.answerMode === "stream"
            ? transport.resume(
                { session_id: sessionId ?? "", value: { region: region ?? "national" } },
                controller.signal,
              )
            : onceAsStream(transport, null, region, sessionId ?? "", controller.signal);
      void run(conv.id, turn.id, generator, sessionId, controller);
    },
    [active, busy, patchConversation, run, transport],
  );

  const ask = useCallback((question: string) => startTurn(question, "ask"), [startTurn]);

  const resume = useCallback(
    (region: string) => {
      const last = active.turns.at(-1);
      if (last?.status === "interrupted") {
        patchTurn(active.id, last.id, (turn) => ({ ...turn, interrupt: null, clarifyOutcome: { kind: "resumed", region } }));
      }
      startTurn(last?.question ?? "", "resume", region);
    },
    [active, patchTurn, startTurn],
  );

  const dismissInterrupt = useCallback(() => {
    const last = active.turns.at(-1);
    if (last?.status === "interrupted") {
      patchTurn(active.id, last.id, (turn) => ({ ...turn, interrupt: null, clarifyOutcome: { kind: "dismissed" } }));
    }
  }, [active, patchTurn]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
  }, []);

  const startConversation = useCallback(() => {
    const conv = emptyConversation();
    setConversations((prev) => [conv, ...prev].slice(0, HISTORY_LIMIT));
    setActiveId(conv.id);
  }, []);

  const selectConversation = useCallback((id: string) => setActiveId(id), []);

  const removeConversation = useCallback((id: string) => {
    setConversations((prev) => {
      const next = prev.filter((row) => row.id !== id);
      const rows = next.length > 0 ? next : [emptyConversation()];
      setActiveId((current) => (current === id ? rows[0].id : current));
      return rows;
    });
  }, []);

  const renameConversation = useCallback(
    (id: string, title: string) => patchConversation(id, (conv) => ({ ...conv, title: title.trim() || conv.title })),
    [patchConversation],
  );

  const setDocIds = useCallback(
    (ids: string[]) => patchConversation(active.id, (conv) => ({ ...conv, docIds: ids })),
    [active.id, patchConversation],
  );

  const appendDocuments = useCallback(
    (ids: string[]) =>
      patchConversation(active.id, (conv) => ({ ...conv, docIds: Array.from(new Set([...conv.docIds, ...ids])) })),
    [active.id, patchConversation],
  );

  return {
    conversations,
    active,
    busy,
    ask,
    resume,
    dismissInterrupt,
    cancel,
    startConversation,
    selectConversation,
    removeConversation,
    renameConversation,
    setDocIds,
    appendDocuments,
  };
}

async function* onceAsStream(
  transport: Transport,
  askBody: { question: string; top_k: number | null; doc_ids: string[]; session_id: string | null } | null,
  region?: string,
  sessionId = "",
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  if (askBody) {
    const response = await transport.askOnce(askBody, signal);
    if (response.status === "interrupted") {
      yield {
        type: "interrupt",
        data: { session_id: response.session_id ?? null, interrupt: response.interrupt ?? null },
      };
      return;
    }
    const { status: _status, ...payload } = response;
    yield { type: "done", data: payload };
    return;
  }
  const response = await transport.resumeOnce(
    { session_id: sessionId, value: { region: region ?? "national" } },
    signal,
  );
  const { status: _ok, ...payload } = response;
  yield { type: "done", data: payload };
}
