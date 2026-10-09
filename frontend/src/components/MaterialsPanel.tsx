import { useRef, useState } from "react";
import type { DocumentsState } from "../state/documents";

type Props = {
  documents: DocumentsState;
  activeDocIds: string[];
  onToggle: (docId: string) => void;
};

export function MaterialsPanel({ documents, activeDocIds, onToggle }: Props) {
  const [mode, setMode] = useState("session");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const pick = async (file: File | null) => {
    if (!file) return;
    setBusy(true);
    await documents.upload(file, mode);
    setBusy(false);
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <section className="materials">
      <header className="materials__head">
        <h3>材料</h3>
        <button type="button" className="btn btn--quiet" onClick={documents.reload} disabled={documents.loading}>
          刷新
        </button>
      </header>
      <div className="materials__upload">
        <div className="segmented" role="group" aria-label="入库方式">
          <button type="button" className={mode === "session" ? "is-active" : ""} onClick={() => setMode("session")}>
            会话材料
          </button>
          <button type="button" className={mode === "permanent" ? "is-active" : ""} onClick={() => setMode("permanent")}>
            永久入库
          </button>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".docx,.md,.txt,.pdf"
          disabled={busy}
          onChange={(event) => void pick(event.target.files?.[0] ?? null)}
        />
        <p className="materials__note">
          {mode === "session"
            ? "会话材料只在这条会话里可检索，引用标 [材料N]。"
            : "永久入库要重建索引之后才可检索，引用标 [依据N]。"}
        </p>
      </div>
      {documents.notice ? <p className="materials__ok">{documents.notice}</p> : null}
      {documents.error ? <p className="materials__error">{documents.error.message}</p> : null}
      <ul className="materials__list">
        {documents.rows.length === 0 ? <li className="materials__empty">台账里还没有材料。</li> : null}
        {documents.rows.map((row) => {
          const used = activeDocIds.includes(row.doc_id);
          return (
            <li key={row.doc_id} className={used ? "is-used" : ""}>
              <label className="materials__row">
                <input type="checkbox" checked={used} onChange={() => onToggle(row.doc_id)} />
                <span className="materials__name">{row.display_name ?? row.doc_id}</span>
                <span className="materials__mode">{row.mode === "permanent" ? "永久" : "会话"}</span>
              </label>
              <button type="button" className="materials__remove" onClick={() => void documents.remove(row.doc_id)}>
                撤下
              </button>
            </li>
          );
        })}
      </ul>
      <p className="materials__note">勾上的材料会随本次提问带上 doc_ids；永久入库的那份撤销要手工移文件并重建索引。</p>
    </section>
  );
}
