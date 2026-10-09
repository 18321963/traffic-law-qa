import { useState } from "react";
import { reviewStateOf } from "../render";
import type { Turn } from "../state/conversation";
import { AnswerBlock, type DraftState } from "./AnswerBlock";
import { ErrorNotice } from "./ErrorNotice";
import { EvidenceList } from "./EvidenceList";
import { RegionClarify } from "./RegionClarify";

type Props = {
  turn: Turn;
  busy: boolean;
  onResume: (region: string) => void;
  onRetry: () => void;
  onDismissInterrupt: () => void;
};

function StatusLine({ turn }: { turn: Turn }) {
  if (turn.status === "cancelled") return <p className="turn__status">已取消：这一轮不再等待收尾帧。</p>;
  if (turn.clarifyOutcome?.kind === "dismissed") {
    return <p className="turn__status">已跳过澄清：这条会话的挂起中断就此作废，之后恢复会得到 409。</p>;
  }
  if (turn.clarifyOutcome?.kind === "resumed") {
    return <p className="turn__status">已按「{turn.clarifyOutcome.region}」恢复，答复在下面。</p>;
  }
  return null;
}

export function TurnView({ turn, busy, onResume, onRetry, onDismissInterrupt }: Props) {
  const [highlight, setHighlight] = useState<number | null>(null);
  const review = reviewStateOf(turn.answer);
  const streaming = turn.status === "streaming";
  const anchor = `turn-${turn.id}`;

  const cite = (index: number) => {
    setHighlight(index);
    const target = document.getElementById(`${anchor}-ev-${index}`);
    target?.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => setHighlight((current) => (current === index ? null : current)), 2400);
  };

  const showDraft = turn.draft !== "" && turn.answer === null;
  const draftState: DraftState = streaming
    ? "streaming"
    : turn.status === "error"
      ? "error"
      : turn.status === "cancelled"
        ? "cancelled"
        : "stopped";
  const showFinal = turn.status === "done" && turn.answer !== null;

  return (
    <article className={`turn turn--${turn.status}`}>
      <header className="turn__head">
        <span className="turn__mark">{turn.resumed ? "续" : "问"}</span>
        <h2 className="turn__question">{turn.question}</h2>
      </header>
      <StatusLine turn={turn} />

      {showDraft ? (
        <AnswerBlock text={turn.draft} answer={null} draftState={draftState} review={review} onCite={cite} />
      ) : null}

      {showFinal ? (
        <>
          {turn.answer?.notes && turn.answer.notes.length > 0 ? (
            <ul className="turn__notes">
              {turn.answer.notes.map((note, index) => (
                <li key={index}>{note}</li>
              ))}
            </ul>
          ) : null}
          <AnswerBlock
            text={turn.answer?.answer ?? ""}
            answer={turn.answer}
            draftState={null}
            review={review}
            onCite={cite}
          />
          {turn.replaced ? (
            <p className="turn__swap">复核后的答案与草稿不一致，界面已整篇换成权威版本。</p>
          ) : null}
          <EvidenceList answer={turn.answer} anchorPrefix={anchor} highlight={highlight} />
        </>
      ) : null}

      {turn.status === "interrupted" && turn.interrupt ? (
        <RegionClarify
          value={turn.interrupt.value}
          busy={busy}
          onResume={onResume}
          onDismiss={onDismissInterrupt}
        />
      ) : null}

      {turn.error ? <ErrorNotice error={turn.error} retryLabel="再试一次" onRetry={onRetry} /> : null}

      {streaming && turn.draft === "" ? (
        <p className="turn__status">
          {turn.mode === "once" ? "一次性请求已发出，等结果返回…" : "等第一个字符…"}
        </p>
      ) : null}
    </article>
  );
}
