import type { Settings } from "../config";
import { networkFailure, readFailure } from "./http";
import { SseDecoder, toStreamEvent } from "./sse";
import { endpoint, requestHeaders, type Transport } from "./transport";
import type {
  AgentHealth,
  AskOk,
  AskResponse,
  DocumentList,
  DocumentRecord,
  QaRequest,
  ResumeRequest,
  StreamEvent,
} from "./types";

export const USAGE =
  "live 传输：/qa/stream 与 /qa/resume/stream 是 POST + SSE（fetch 流读，非 EventSource）；非流式面走 /qa 与 /qa/resume；鉴权头只在设了 key 时带。";

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  return (text ? JSON.parse(text) : {}) as T;
}

async function* streamOf(
  path: string,
  body: unknown,
  settings: Settings,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  let response: Response;
  try {
    response = await fetch(endpoint(settings.base, path), {
      method: "POST",
      headers: requestHeaders(settings, true),
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

async function postJson<T>(
  path: string,
  body: unknown,
  settings: Settings,
  signal?: AbortSignal,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(endpoint(settings.base, path), {
      method: "POST",
      headers: requestHeaders(settings, true),
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

export function createLiveTransport(getSettings: () => Settings): Transport {
  return {
    ask(req: QaRequest, signal?: AbortSignal) {
      return streamOf("/qa/stream", req, getSettings(), signal);
    },
    resume(req: ResumeRequest, signal?: AbortSignal) {
      return streamOf("/qa/resume/stream", req, getSettings(), signal);
    },
    async askOnce(req: QaRequest, signal?: AbortSignal) {
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
  };
}
