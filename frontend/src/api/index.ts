import type { Settings } from "../config";
import { createLiveTransport } from "./live";
import { createMockTransport } from "./mock";
import type { Transport } from "./transport";

export const USAGE =
  "传输选择：mode=mock 用内置示例数据，mode=live 直连 VITE_API_BASE / 同源代理。每次调用都现读设置，切换不用刷新。";

export function createTransport(readSettings: () => Settings): Transport {
  const live = createLiveTransport(readSettings);
  const mock = createMockTransport(readSettings);
  const pick = (): Transport => (readSettings().mode === "live" ? live : mock);
  return {
    ask: (req, signal) => pick().ask(req, signal),
    resume: (req, signal) => pick().resume(req, signal),
    askOnce: (req, signal) => pick().askOnce(req, signal),
    resumeOnce: (req, signal) => pick().resumeOnce(req, signal),
    health: (signal) => pick().health(signal),
    listDocuments: (mode, signal) => pick().listDocuments(mode, signal),
    uploadDocument: (file, mode, signal) => pick().uploadDocument(file, mode, signal),
    deleteDocument: (docId, signal) => pick().deleteDocument(docId, signal),
    deleteSession: (sessionId, signal) => pick().deleteSession(sessionId, signal),
  };
}

export type { Transport };
