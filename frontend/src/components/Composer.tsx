import { useEffect, useRef, useState } from "react";
import type { AnswerMode } from "../config";

type Props = {
  busy: boolean;
  answerMode: AnswerMode;
  sessionId: string | null;
  docIds: string[];
  onAsk: (question: string) => void;
  onCancel: () => void;
  onAnswerMode: (mode: AnswerMode) => void;
};

const SAMPLES = ["机动车闯红灯一次记多少分？", "深圳的电动自行车载人怎么规定的？", "醉酒驾驶机动车怎么处罚？"];

export function Composer({ busy, answerMode, sessionId, docIds, onAsk, onCancel, onAnswerMode }: Props) {
  const [text, setText] = useState("");
  const areaRef = useRef<HTMLTextAreaElement | null>(null);

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
    <div className="composer">
      <div className="composer__row">
        <textarea
          ref={areaRef}
          className="composer__input"
          value={text}
          rows={2}
          placeholder="问一条交通法规问题。回车发送，Shift+回车换行。"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <div className="composer__actions">
          {busy ? (
            <button type="button" className="btn btn--quiet" onClick={onCancel}>
              停止
            </button>
          ) : null}
          <button type="button" className="btn btn--primary" disabled={busy || text.trim() === ""} onClick={submit}>
            {busy ? "提问中…" : "提问"}
          </button>
        </div>
      </div>
      <div className="composer__meta">
        <div className="segmented" role="group" aria-label="回答方式">
          <button
            type="button"
            className={answerMode === "stream" ? "is-active" : ""}
            onClick={() => onAnswerMode("stream")}
          >
            流式
          </button>
          <button
            type="button"
            className={answerMode === "once" ? "is-active" : ""}
            onClick={() => onAnswerMode("once")}
          >
            一次性
          </button>
        </div>
        <span className="composer__session">
          {sessionId ? `会话 ${sessionId.slice(0, 10)}…（跨轮记忆已开）` : "这条会话还没有 session_id（首问后会有）"}
        </span>
        {docIds.length > 0 ? <span className="composer__materials">材料 {docIds.length} 份</span> : null}
      </div>
      {text.trim() === "" && !busy ? (
        <div className="composer__samples">
          {SAMPLES.map((sample) => (
            <button key={sample} type="button" className="chip" onClick={() => setText(sample)}>
              {sample}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
