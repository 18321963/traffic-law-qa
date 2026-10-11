import type { Settings } from "../config";
import { networkFailure, readFailure } from "./http";
import { SseDecoder, toStreamEvent } from "./sse";
import { endpoint, requestHeaders, type Transport } from "./transport";
import type {
  AgentHealth,
  AnswerPayload,
  AskOk,
  AskResponse,
  DocumentList,
  DocumentRecord,
  QaRequest,
  ResumeRequest,
  RetrievalResult,
  StreamEvent,
} from "./types";

export const USAGE =
  "live 传输：/qa/stream 与 /qa/resume/stream 是 POST + SSE（fetch 流读，非 EventSource）；非流式面走 /qa 与 /qa/resume；鉴权头只在设了 key 时带。channel=rag 时改打同源 /rag（qa 检索 → answer/stream；勾了材料先 materials/search），忽略 base 与 key。";

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  return (text ? JSON.parse(text) : {}) as T;
}

async function* streamAt(
  url: string,
  headers: HeadersInit,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
  } catch (exc) {
    throw networkFailure(exc);
  }
  if (!response.ok) {
    throw await readFailure(response);
  }
  if (!response.body) {
    throw await readFailure(response);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const sse = new SseDecoder();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      for (const frame of sse.push(chunk)) {
        const event = toStreamEvent(frame);
        if (event) yield event;
      }
    }
    const tail = decoder.decode();
    if (tail) {
      for (const frame of sse.push(tail)) {
        const event = toStreamEvent(frame);
        if (event) yield event;
      }
    }
    for (const frame of sse.flush()) {
      const event = toStreamEvent(frame);
      if (event) yield event;
    }
  } finally {
    reader.releaseLock();
    if (!response.body.locked) {
      await response.body.cancel().catch(() => undefined);
    }
  }
}

async function* streamOf(
  path: string,
  body: unknown,
  settings: Settings,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  yield* streamAt(endpoint(settings.base, path), requestHeaders(settings, true), body, signal);
}

async function postAt<T>(
  url: string,
  headers: HeadersInit,
  body: unknown,
  signal?: AbortSignal,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
  } catch (exc) {
    throw networkFailure(exc);
  }
  if (!response.ok) {
    throw await readFailure(response);
  }
  return readJson<T>(response);
}

async function postJson<T>(
  path: string,
  body: unknown,
  settings: Settings,
  signal?: AbortSignal,
): Promise<T> {
  return postAt<T>(endpoint(settings.base, path), requestHeaders(settings, true), body, signal);
}

const RAG_JSON: HeadersInit = { "Content-Type": "application/json" };
const RAG_MODE_SEARCH = "search";
const RAG_MODE_ASK = "ask";

type RagRetrieval = RetrievalResult & { mode?: string; request_ms?: number };

async function ragRetrieval(
  question: string,
  topK: number | null | undefined,
  signal?: AbortSignal,
): Promise<RetrievalResult> {
  const raw = await postAt<RagRetrieval>("/rag/qa", RAG_JSON, {
    question,
    mode: RAG_MODE_SEARCH,
    top_k: topK ?? undefined,
  }, signal);
  const { mode: _mode, request_ms: _request_ms, ...retrieval } = raw;
  return retrieval;
}

async function ragMaterials(query: string, docIds: string[], signal?: AbortSignal): Promise<unknown[]> {
  const payload = await postAt<{ passages?: unknown[] }>("/rag/materials/search", RAG_JSON, {
    query,
    doc_ids: docIds,
  }, signal);
  return payload.passages ?? [];
}

async function* ragAsk(req: QaRequest, signal?: AbortSignal): AsyncGenerator<StreamEvent> {
  const docIds = req.doc_ids ?? [];
  const [retrieval, materials] = await Promise.all([
    ragRetrieval(req.question, req.top_k, signal),
    docIds.length > 0 ? ragMaterials(req.question, docIds, signal) : Promise.resolve<unknown[]>([]),
  ]);
  yield* streamAt("/rag/answer/stream", RAG_JSON, {
    question: { text: req.question, history: [] },
    retrieval,
    timeliness: [],
    materials,
  }, signal);
}

async function ragAnswerOnce(req: QaRequest, signal?: AbortSignal): Promise<AnswerPayload> {
  const docIds = req.doc_ids ?? [];
  if (docIds.length === 0) {
    const payload = await postAt<AnswerPayload & { mode?: string }>("/rag/qa", RAG_JSON, {
      question: req.question,
      mode: RAG_MODE_ASK,
      top_k: req.top_k ?? undefined,
    }, signal);
    const { mode: _mode, ...answer } = payload;
    return answer;
  }
  const [retrieval, materials] = await Promise.all([
    ragRetrieval(req.question, req.top_k, signal),
    ragMaterials(req.question, docIds, signal),
  ]);
  return postAt<AnswerPayload>("/rag/answer", RAG_JSON, {
    question: { text: req.question, history: [] },
    retrieval,
    timeliness: [],
    materials,
  }, signal);
}

export function createLiveTransport(getSettings: () => Settings): Transport {
  return {
    ask(req: QaRequest, signal?: AbortSignal) {
      const settings = getSettings();
      if (settings.channel === "rag") return ragAsk(req, signal);
      return streamOf("/qa/stream", req, settings, signal);
    },
    resume(req: ResumeRequest, signal?: AbortSignal) {
      return streamOf("/qa/resume/stream", req, getSettings(), signal);
    },
    async askOnce(req: QaRequest, signal?: AbortSignal) {
      if (getSettings().channel === "rag") {
        const payload = await ragAnswerOnce(req, signal);
        return { status: "ok", ...payload } as AskOk;
      }
      return postJson<AskResponse>("/qa", req, getSettings(), signal);
    },
    async resumeOnce(req: ResumeRequest, signal?: AbortSignal) {
      return postJson<AskOk>("/qa/resume", req, getSettings(), signal);
    },
    async health(signal?: AbortSignal) {
      const settings = getSettings();
      let response: Response;
      try {
        response = await fetch(endpoint(settings.base, "/health"), {
          headers: requestHeaders(settings, false),
          signal,
        });
      } catch (exc) {
        throw networkFailure(exc);
      }
      if (!response.ok) {
        throw await readFailure(response);
      }
      return readJson<AgentHealth>(response);
    },
    async listDocuments(mode?: string | null, signal?: AbortSignal) {
      const settings = getSettings();
      const query = mode ? `?mode=${encodeURIComponent(mode)}` : "";
      let response: Response;
      try {
        response = await fetch(endpoint(settings.base, `/documents${query}`), {
          headers: requestHeaders(settings, false),
          signal,
        });
      } catch (exc) {
        throw networkFailure(exc);
      }
      if (!response.ok) {
        throw await readFailure(response);
      }
      return readJson<DocumentList>(response);
    },
    async uploadDocument(file: File, mode: string, signal?: AbortSignal) {
      const settings = getSettings();
      const form = new FormData();
      form.append("file", file);
      if (mode) form.append("mode", mode);
      let response: Response;
      try {
        response = await fetch(endpoint(settings.base, "/documents"), {
          method: "POST",
          headers: requestHeaders(settings, false),
          body: form,
          signal,
        });
      } catch (exc) {
        throw networkFailure(exc);
      }
      if (!response.ok) {
        throw await readFailure(response);
      }
      return readJson<DocumentRecord>(response);
    },
    async deleteDocument(docId: string, signal?: AbortSignal) {
      const settings = getSettings();
      let response: Response;
      try {
        response = await fetch(endpoint(settings.base, `/documents/${encodeURIComponent(docId)}`), {
          method: "DELETE",
          headers: requestHeaders(settings, false),
          signal,
        });
      } catch (exc) {
        throw networkFailure(exc);
      }
      if (!response.ok) {
        throw await readFailure(response);
      }
      return readJson<unknown>(response);
    },
    async deleteSession(sessionId: string, signal?: AbortSignal) {
      const settings = getSettings();
      let response: Response;
      try {
        response = await fetch(endpoint(settings.base, `/sessions/${encodeURIComponent(sessionId)}`), {
          method: "DELETE",
          headers: requestHeaders(settings, false),
          signal,
        });
      } catch (exc) {
        throw networkFailure(exc);
      }
      if (!response.ok) {
        throw await readFailure(response);
      }
      return readJson<unknown>(response);
    },
  };
}
