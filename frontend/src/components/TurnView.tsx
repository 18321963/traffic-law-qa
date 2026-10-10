import { App as AntdApp, Avatar, Button, Tooltip } from "antd";
import {
  CopyOutlined,
  ExclamationCircleFilled,
  FileSearchOutlined,
  InfoCircleOutlined,
  ReloadOutlined,
  RightOutlined,
  SyncOutlined,
} from "@ant-design/icons";
import { reviewStateOf } from "../render";
import type { Turn } from "../state/conversation";
import { AnswerBlock, type DraftState } from "./AnswerBlock";
import { BrandMark } from "./BrandMark";
import { ErrorNotice } from "./ErrorNotice";
import { RegionClarify } from "./RegionClarify";
import { StepTrace } from "./StepTrace";
import styles from "./TurnView.module.css";

export const USAGE =
  "一轮问答：右侧用户气泡 + 助手消息（处理轨迹 → 草稿/终稿 → 澄清卡/出错面 → 操作行）；续答轮不重复用户气泡。";

type Props = {
  turn: Turn;
  busy: boolean;
  onResume: (region: string) => void;
  onRetry: () => void;
  onDismissInterrupt: () => void;
  onFocusEvidence: (turnId: string, index: number | null) => void;
};

function timeOf(ms: number): string {
  const date = new Date(ms);
  const hour = `${date.getHours()}`.padStart(2, "0");
  const minute = `${date.getMinutes()}`.padStart(2, "0");
  return `${hour}:${minute}`;
}

function SystemLine({ children }: { children: React.ReactNode }) {
  return <p className={styles.sysLine}>{children}</p>;
}

export function TurnView({ turn, busy, onResume, onRetry, onDismissInterrupt, onFocusEvidence }: Props) {
  const { message } = AntdApp.useApp();
  const review = reviewStateOf(turn.answer);
  const streaming = turn.status === "streaming";
  const showDraft = turn.draft !== "" && turn.answer === null;
  const draftState: DraftState = streaming
    ? "streaming"
    : turn.status === "error"
      ? "error"
      : turn.status === "cancelled"
        ? "cancelled"
        : "stopped";
  const showFinal = turn.status === "done" && turn.answer !== null;
  const original = turn.answer?.review?.original_text;
  const originalText = original && original !== turn.answer?.answer ? original : null;
  const evidenceCount = turn.answer?.evidences?.length ?? 0;
  const hitCount = turn.answer?.retrieval?.articles?.length ?? 0;

  const copy = () => {
    const text = turn.answer?.answer ?? "";
    void navigator.clipboard
      .writeText(text)
      .then(() => message.success("答案已复制"))
      .catch(() => message.error("复制失败：浏览器拒绝了剪贴板访问"));
  };

  return (
    <article className={styles.turn}>
      {turn.question !== "" && !turn.resumed ? (
        <div className={styles.userRow}>
          <div className={styles.userBubble}>{turn.question}</div>
        </div>
      ) : null}

      {turn.clarifyOutcome?.kind === "dismissed" ? (
        <SystemLine>已跳过澄清：这条会话的挂起中断就此作废，之后恢复会得到 409。</SystemLine>
      ) : null}
      {turn.clarifyOutcome?.kind === "resumed" ? (
        <SystemLine>已按「{turn.clarifyOutcome.region}」恢复，答复在下面一轮。</SystemLine>
      ) : null}
      {turn.status === "cancelled" ? <SystemLine>已取消：这一轮不再等待收尾帧。</SystemLine> : null}

      <div className={styles.assistant}>
        <div className={styles.sender}>
          <Avatar size={30} className={styles.avatar} icon={<BrandMark size={30} />} />
          <span className={styles.name}>法规助手</span>
          {turn.resumed ? <span className={styles.resumeTag}>续答</span> : null}
        </div>

        <div className={styles.content}>
          <StepTrace
            steps={turn.steps}
            streaming={streaming}
            mode={turn.mode}
            elapsedMs={turn.elapsedMs}
          />

          {showDraft ? (
            <AnswerBlock
              text={turn.draft}
              answer={null}
              draftState={draftState}
              review={review}
              onCite={() => undefined}
            />
          ) : null}

          {showFinal ? (
            <>
              {turn.answer?.notes && turn.answer.notes.length > 0 ? (
                <ul className={styles.notes}>
                  {turn.answer.notes.map((note, index) => (
                    <li key={index}>
                      <InfoCircleOutlined className={styles.noteIcon} />
                      <span>{note}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
              <AnswerBlock
                text={turn.answer?.answer ?? ""}
                answer={turn.answer}
                draftState={null}
                review={review}
                onCite={(index) => onFocusEvidence(turn.id, index)}
              />
              {review.tier === "failed" ? (
                <div className={styles.downgrade}>
                  <ExclamationCircleFilled className={styles.downgradeIcon} />
                  复核未过：服务没给结论，下面是降级答复与未经复核确认的候选法条，需人工复审。
                </div>
              ) : turn.replaced ? (
                <p className={styles.swap}>
                  <SyncOutlined className={styles.swapIcon} />
                  复核后的答案与草稿不一致，界面已整篇换成权威版本。
                </p>
              ) : null}
              {originalText ? (
                <details className={styles.original}>
                  <summary>复核前的草稿（模型原稿）</summary>
                  <div className={styles.originalBody}>{originalText}</div>
                </details>
              ) : null}
              <div className={styles.actions}>
                <span className={styles.time}>{timeOf(turn.startedAt)}</span>
                <Tooltip title="复制答案">
                  <Button type="text" shape="circle" size="small" icon={<CopyOutlined />} onClick={copy} />
                </Tooltip>
                <Tooltip title="重新提问">
                  <Button
                    type="text"
                    shape="circle"
                    size="small"
                    icon={<ReloadOutlined />}
                    disabled={busy}
                    onClick={onRetry}
                  />
                </Tooltip>
                {evidenceCount > 0 || hitCount > 0 ? (
                  <button type="button" className={styles.evidenceChip} onClick={() => onFocusEvidence(turn.id, null)}>
                    <FileSearchOutlined className={styles.evidenceChipIcon} />
                    <b>依据 {evidenceCount} 条</b>
                    <span className={styles.evidenceChipMeta}>命中 {hitCount}</span>
                    <RightOutlined className={styles.evidenceChipArrow} />
                  </button>
                ) : null}
              </div>
            </>
          ) : null}

          {turn.status === "interrupted" && turn.interrupt ? (
            <RegionClarify value={turn.interrupt.value} busy={busy} onResume={onResume} onDismiss={onDismissInterrupt} />
          ) : null}

          {turn.error ? <ErrorNotice error={turn.error} retryLabel="再试一次" onRetry={onRetry} /> : null}

          {streaming && turn.draft === "" ? (
            <div className={styles.searching}>
              <span className={styles.searchingIcon} aria-hidden="true" />
              <span className={styles.searchingText}>
                {turn.mode === "once" ? "一次性请求已发出，等结果返回…" : "正在检索与规划…"}
              </span>
            </div>
          ) : null}
        </div>
      </div>
    </article>
  );
}
