import { useState } from "react";
import { Alert, Checkbox, Segmented, Upload } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import type { DocumentsState } from "../state/documents";
import styles from "./MaterialsPanel.module.css";

export const USAGE =
  "右栏材料页签：拖拽/点击上传（会话材料或永久入库）、台账勾选（随提问带 doc_ids）、撤下一份；服务给的 note 与拒收回执原样展示。";

type Props = {
  documents: DocumentsState;
  activeDocIds: string[];
  onToggle: (docId: string) => void;
};

export function MaterialsPanel({ documents, activeDocIds, onToggle }: Props) {
  const [mode, setMode] = useState("session");
  const [busy, setBusy] = useState(false);

  return (
    <div className={styles.materials}>
      <div className={styles.uploadCard}>
        <Segmented
          block
          size="small"
          value={mode}
          onChange={(value) => setMode(String(value))}
          options={[
            { label: "会话材料", value: "session" },
            { label: "永久入库", value: "permanent" },
          ]}
        />
        <Upload.Dragger
          accept=".docx,.md,.txt,.pdf"
          showUploadList={false}
          disabled={busy}
          customRequest={(options) => {
            setBusy(true);
            void documents
              .upload(options.file as File, mode)
              .then((record) => {
                if (record) options.onSuccess?.(record);
                else options.onError?.(new Error("上传未受理"));
              })
              .finally(() => setBusy(false));
          }}
        >
          <p className={styles.draggerIcon}>
            <InboxOutlined />
          </p>
          <p className={styles.draggerText}>点击或把文件拖到这里上传</p>
          <p className={styles.draggerHint}>支持 docx / md / txt / pdf</p>
        </Upload.Dragger>
        <p className={styles.note}>
          {mode === "session"
            ? "会话材料只在这条会话里可检索，引用标 [材料N]。"
            : "永久入库要重建索引之后才可检索，引用标 [依据N]。"}
        </p>
      </div>

      {documents.notice ? (
        <Alert type="success" showIcon closable message={documents.notice} onClose={documents.clearNotice} />
      ) : null}
      {documents.error ? <Alert type="error" showIcon message={documents.error.message} /> : null}

      <div className={styles.list}>
        {documents.rows.length === 0 ? (
          <p className={styles.empty}>台账里还没有材料。</p>
        ) : (
          documents.rows.map((row) => {
            const used = activeDocIds.includes(row.doc_id);
            return (
              <div key={row.doc_id} className={`${styles.doc} ${used ? styles.docUsed : ""}`}>
                <Checkbox checked={used} onChange={() => onToggle(row.doc_id)} />
                <span className={styles.docName} title={row.display_name ?? row.doc_id}>
                  {row.display_name ?? row.doc_id}
                </span>
                <span className={styles.docMode}>{row.mode === "permanent" ? "永久" : "会话"}</span>
                <button type="button" className={styles.docRemove} onClick={() => void documents.remove(row.doc_id)}>
                  撤下
                </button>
              </div>
            );
          })
        )}
      </div>

      <p className={styles.note}>勾上的材料会随本次提问带上 doc_ids；永久入库的那份撤销要手工移文件并重建索引。</p>
    </div>
  );
}
