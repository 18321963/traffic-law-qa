import { useCallback, useEffect, useState } from "react";
import { isAbort } from "../api/http";
import type { Transport } from "../api/transport";
import { ApiError, type DocumentRecord } from "../api/types";
import { turnErrorOf, type TurnError } from "./conversation";

export const USAGE =
  "材料台账：POST /documents 上传（multipart），GET 列台账，DELETE 撤一份；永久入库的那份在这里删不掉，服务会给 400 与手工步骤。";

export type DocumentsState = {
  rows: DocumentRecord[];
  loading: boolean;
  error: TurnError | null;
  notice: string | null;
  reload: () => void;
  upload: (file: File, mode: string) => Promise<DocumentRecord | null>;
  remove: (docId: string) => Promise<void>;
  clearNotice: () => void;
};

export function useDocuments(transport: Transport, revision: string): DocumentsState {
  const [rows, setRows] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<TurnError | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    transport
      .listDocuments(null, controller.signal)
      .then((list) => {
        setRows(list.documents ?? []);
        setError(null);
      })
      .catch((exc) => {
        if (isAbort(exc)) return;
        setError(turnErrorOf(exc));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [transport, nonce, revision]);

  const upload = useCallback(
    async (file: File, mode: string) => {
      setError(null);
      setNotice(null);
      try {
        const record = await transport.uploadDocument(file, mode);
        const note = typeof record.note === "string" ? record.note : "已受理";
        setNotice(`${record.display_name ?? file.name}：${note}`);
        reload();
        return record;
      } catch (exc) {
        if (exc instanceof ApiError && exc.body && typeof exc.body === "object") {
          const note = (exc.body as { note?: unknown }).note;
          setError({ ...turnErrorOf(exc), message: typeof note === "string" ? `${exc.detail}：${note}` : exc.detail });
        } else {
          setError(turnErrorOf(exc));
        }
        return null;
      }
    },
    [reload, transport],
  );

  const remove = useCallback(
    async (docId: string) => {
      setError(null);
      setNotice(null);
      try {
        await transport.deleteDocument(docId);
        setNotice(`已撤下 ${docId}`);
        reload();
      } catch (exc) {
        setError(turnErrorOf(exc));
      }
    },
    [reload, transport],
  );

  return { rows, loading, error, notice, reload, upload, remove, clearNotice: () => setNotice(null) };
}
