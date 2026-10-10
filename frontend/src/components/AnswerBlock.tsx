import { groupClass, paragraphsOf, segmentsOf, type ReviewState } from "../render";
import type { DonePayload } from "../api/types";
import styles from "./AnswerBlock.module.css";

export const USAGE =
  "答案正文：草稿态配灰底与光标，终稿配复核徽章与一行元信息；[依据N]/[时效N]/[材料N] 切成可点引用片。";

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

const TIER_CLASS: Record<ReviewState["tier"], string> = {
  passed: styles.tierPassed,
  failed: styles.tierFailed,
  unfinished: styles.tierMuted,
  unscored: styles.tierMuted,
  none: styles.tierMuted,
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
  return <p className={styles.meta}>{rows.join(" · ")}</p>;
}

export function AnswerBlock({ text, answer, draftState, review, onCite }: Props) {
  const paragraphs = paragraphsOf(text);
  const body = paragraphs.length > 0 ? paragraphs : [text];
  const draft = draftState !== null;
  const streaming = draftState === "streaming";
  return (
    <section className={draft ? styles.draft : styles.final}>
      {draft ? (
        <div className={styles.headRow}>
          <span className={styles.draftChip}>草稿</span>
          <span className={styles.draftHeadline}>{DRAFT_HEADLINE[draftState]}</span>
        </div>
      ) : (
        <div className={styles.headRow}>
          <span className={`${styles.chip} ${TIER_CLASS[review.tier]}`}>{review.label}</span>
          <span className={styles.finalHeadline}>答复</span>
        </div>
      )}
      <div className={styles.body}>
        {body.map((paragraph, index) => (
          <p key={index}>
            {segmentsOf(paragraph).map((segment, offset) =>
              segment.kind === "text" ? (
                <span key={offset}>{segment.text}</span>
              ) : (
                <button
                  key={offset}
                  type="button"
                  className={`${styles.cite} ${styles[groupClass(segment.group)]}`}
                  onClick={() => onCite(segment.index)}
                  title={`跳到${segment.group}${segment.index}`}
                >
                  {segment.label}
                </button>
              ),
            )}
            {streaming && index === body.length - 1 ? <span className={styles.caret} aria-hidden="true" /> : null}
          </p>
        ))}
      </div>
      {answer ? <Meta answer={answer} review={review} /> : null}
    </section>
  );
}
