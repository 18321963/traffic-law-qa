import type { Settings } from "../config";
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
  "传输端口：live（HTTP 直连）与 mock（内置示例数据）同签名，界面只认这个接口，切模式不改调用点。";

export type Transport = {
  ask(req: QaRequest, signal?: AbortSignal): AsyncGenerator<StreamEvent>;
  resume(req: ResumeRequest, signal?: AbortSignal): AsyncGenerator<StreamEvent>;
  askOnce(req: QaRequest, signal?: AbortSignal): Promise<AskResponse>;
  resumeOnce(req: ResumeRequest, signal?: AbortSignal): Promise<AskOk>;
  health(signal?: AbortSignal): Promise<AgentHealth>;
  listDocuments(mode?: string | null, signal?: AbortSignal): Promise<DocumentList>;
  uploadDocument(file: File, mode: string, signal?: AbortSignal): Promise<DocumentRecord>;
  deleteDocument(docId: string, signal?: AbortSignal): Promise<unknown>;
  deleteSession(sessionId: string, signal?: AbortSignal): Promise<unknown>;
};

export type TransportFactory = (settings: Settings) => Transport;

export function endpoint(base: string, path: string): string {
  const trimmed = (base || "").trim().replace(/\/+$/, "");
  return trimmed === "" ? path : `${trimmed}${path}`;
}

export function requestHeaders(settings: Settings, json: boolean): HeadersInit {
  const headers: Record<string, string> = {};
  if (json) headers["Content-Type"] = "application/json";
  const key = (settings.apiKey || "").trim();
  if (key !== "") headers["X-API-Key"] = key;
  return headers;
}
