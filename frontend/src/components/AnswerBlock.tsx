import { groupClass, paragraphsOf, segmentsOf, type ReviewState } from "../render";
import type { DonePayload } from "../api/types";

export type DraftState = "streaming" | "error" | "cancelled" | "stopped";

type Props = {
  text: string;
  answer: DonePayload | null;
  draftState: DraftState | null;
  review: ReviewState;
  onCite: (index: number) => void;
};

const DRAFT_HEADLINE: Record<DraftState, string> = {
  streaming: "正在生成，复核完成后整篇替换",
  error: "流已中断，下面是中断前收到的文字",
  cancelled: "已取消，下面是取消前收到的文字",
  stopped: "这一轮已停下，下面是已收到的文字",
};

function Meta({ answer, review }: { answer: DonePayload; review: ReviewState }) {
  const rows: string[] = [];
  if (answer.model) rows.push(`模型 ${answer.model}`);
  if (typeof answer.request_ms === "number") rows.push(`${(answer.request_ms / 1000).toFixed(1)} 秒`);
  if (typeof answer.retrieval?.elapsed_ms === "number") rows.push(`检索 ${Math.round(answer.retrieval.elapsed_ms)} 毫秒`);
  const usage = answer.usage?.total_tokens;
  if (typeof usage === "number") rows.push(`${usage} tokens`);
  if ((answer.evidences?.length ?? 0) > 0) rows.push(`依据 ${answer.evidences?.length} 条`);
  if (review.detail) rows.push(review.detail);
  return <p className="answer__meta">{rows.join(" · ")}</p>;
}

export function AnswerBlock({ text, answer, draftState, review, onCite }: Props) {
  const paragraphs = paragraphsOf(text);
  const body = paragraphs.length > 0 ? paragraphs : [text];
  const draft = draftState !== null;
  return (
    <section
      className={
        draft
          ? `answer answer--draft${draftState === "streaming" ? "" : ` answer--draft-${draftState}`}`
          : "answer answer--final"
      }
    >
      <header className="answer__head">
        {draft ? (
          <>
            <span className="tag tag--draft">草稿</span>
            <span className="answer__headline">{DRAFT_HEADLINE[draftState]}</span>
            {draftState === "streaming" ? <span className="caret" aria-hidden="true" /> : null}
          </>
        ) : (
          <>
            <span className={`tag tag--${review.tier}`}>{review.label}</span>
            <span className="answer__headline">答复</span>
          </>
        )}
      </header>
      <div className="answer__body">
        {body.map((paragraph, index) => (
          <p key={index}>
            {segmentsOf(paragraph).map((segment, offset) =>
              segment.kind === "text" ? (
                <span key={offset}>{segment.text}</span>
              ) : (
                <button
                  key={offset}
                  type="button"
                  className={`cite cite--${groupClass(segment.group)}`}
                  onClick={() => onCite(segment.index)}
                  title={`跳到${segment.group}${segment.index}`}
                >
                  {segment.label}
                </button>
              ),
            )}
          </p>
        ))}
      </div>
      {answer ? <Meta answer={answer} review={review} /> : null}
    </section>
  );
}
