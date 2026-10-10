import { useEffect, useRef, useState } from "react";
import { Button, Input, Tooltip } from "antd";
import { CloseOutlined, FileTextOutlined, PaperClipOutlined, SendOutlined, StopOutlined } from "@ant-design/icons";
import type { AnswerMode } from "../config";
import styles from "./Composer.module.css";

export const USAGE =
  "输入区：来源药丸（本会话挂上的材料，可单个摘掉）+ 发送卡；附件按钮上传会话材料，回车发送、Shift+回车换行。";

export type Material = { docId: string; name: string };

type Props = {
  busy: boolean;
  answerMode: AnswerMode;
  sessionId: string | null;
  materials: Material[];
  onRemoveDoc: (docId: string) => void;
  onAsk: (question: string) => void;
  onCancel: () => void;
  onAttach: (file: File) => void;
};

export function Composer({
  busy,
  answerMode,
  sessionId,
  materials,
  onRemoveDoc,
  onAsk,
  onCancel,
  onAttach,
}: Props) {
  const [text, setText] = useState("");
  const areaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!busy) areaRef.current?.focus();
  }, [busy]);

  const submit = () => {
    const question = text.trim();
    if (question === "" || busy) return;
    setText("");
    onAsk(question);
  };

  return (
    <div className={styles.wrap}>
      {materials.length > 0 ? (
        <div className={styles.source}>
          <span className={styles.sourceTitle}>来源</span>
          <div className={styles.sourceList}>
            {materials.map((row) => (
              <span key={row.docId} className={styles.sourceItem}>
                <FileTextOutlined className={styles.sourceIcon} />
                <span className={styles.sourceName} title={row.name}>
                  {row.name}
                </span>
                <button type="button" className={styles.sourceClose} onClick={() => onRemoveDoc(row.docId)}>
                  <CloseOutlined />
                </button>
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div className={styles.sender}>
        <Input.TextArea
          ref={areaRef}
          className={styles.input}
          value={text}
          autoSize={{ minRows: 2, maxRows: 6 }}
          placeholder="问一条交通法规问题。回车发送，Shift+回车换行。"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <div className={styles.actions}>
          <div className={styles.actionsLeft}>
            <input
              ref={fileRef}
              type="file"
              accept=".docx,.md,.txt,.pdf"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) onAttach(file);
                event.target.value = "";
              }}
            />
            <Tooltip title="上传会话材料（也可以拖到右栏材料页签）">
              <Button size="small" icon={<PaperClipOutlined />} disabled={busy} onClick={() => fileRef.current?.click()}>
                附件
              </Button>
            </Tooltip>
            {answerMode === "once" ? <span className={styles.onceTag}>一次性问答</span> : null}
          </div>
          <div className={styles.actionsRight}>
            {busy ? (
              <Button size="small" icon={<StopOutlined />} onClick={onCancel}>
                停止
              </Button>
            ) : null}
            <Button
              type="primary"
              className={styles.send}
              icon={<SendOutlined />}
              loading={busy}
              disabled={busy || text.trim() === ""}
              onClick={submit}
            >
              发送
            </Button>
          </div>
        </div>
      </div>

      <div className={styles.meta}>
        {sessionId ? `会话 ${sessionId.slice(0, 10)}…（跨轮记忆已开）` : "这条会话还没有 session_id（首问后会有）"}
        {materials.length > 0 ? ` · 材料 ${materials.length} 份` : ""}
      </div>
    </div>
  );
}
