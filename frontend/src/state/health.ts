import { useCallback, useEffect, useRef, useState } from "react";
import { isAbort } from "../api/http";
import type { Transport } from "../api/transport";
import { ApiError, type AgentHealth } from "../api/types";

export const USAGE =
  "健康轮询：默认 20 秒一次，窗口重新聚焦时补一次；/health 免鉴权，探活失败按连不上处理。";

export type HealthState = {
  status: "loading" | "ok" | "degraded" | "error";
  data: AgentHealth | null;
  error: string | null;
  status404: number | null;
  checkedAt: number | null;
  refresh: () => void;
};

export function useHealth(transport: Transport, revision: string, intervalMs = 20000): HealthState {
  const [state, setState] = useState<Omit<HealthState, "refresh">>({
    status: "loading",
    data: null,
    error: null,
    status404: null,
    checkedAt: null,
  });
  const timerRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const check = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const data = await transport.health(controller.signal);
      const rag = data.rag;
      const ragDown = !rag || (rag.status !== undefined && rag.status !== "ok") || Boolean(rag.detail);
      const degraded =
        data.status !== "ok" ||
        ragDown ||
        rag?.degraded === true ||
        rag?.llm_ready === false ||
        (typeof rag?.channels === "string" && rag.channels !== "稠密+BM25");
      setState({
        status: degraded ? "degraded" : "ok",
        data,
        error: null,
        status404: null,
        checkedAt: Date.now(),
      });
    } catch (exc) {
      if (isAbort(exc)) return;
      const message = exc instanceof Error ? exc.message : String(exc);
      setState({
        status: "error",
        data: null,
        error: exc instanceof ApiError && exc.kind === "http" ? exc.detail : message,
        status404: exc instanceof ApiError && exc.kind === "http" ? exc.status : null,
        checkedAt: Date.now(),
      });
    }
  }, [transport, revision]);

  useEffect(() => {
    void check();
    timerRef.current = window.setInterval(() => void check(), intervalMs);
    const onFocus = () => void check();
    window.addEventListener("focus", onFocus);
    return () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      window.removeEventListener("focus", onFocus);
      abortRef.current?.abort();
    };
  }, [check, intervalMs]);

  return { ...state, refresh: () => void check() };
}
