import { useState } from "react";
import {
  AimOutlined,
  DownOutlined,
  EditOutlined,
  EnvironmentOutlined,
  QuestionCircleOutlined,
  RightOutlined,
  SafetyCertificateOutlined,
  SearchOutlined,
  ThunderboltOutlined,
  UpOutlined,
} from "@ant-design/icons";
import { nodeLabel, stepDetail } from "../render";
import type { StepEvent } from "../api/types";
import styles from "./StepTrace.module.css";

export const USAGE =
  "处理轨迹：inline 变体挂在回答里（流式中默认展开、收尾后收起，用户手动开合优先），panel 变体放在右栏轨迹页签。";

type Props = {
  steps: StepEvent[];
  streaming: boolean;
  mode: string;
  elapsedMs: number | null;
  variant?: "inline" | "panel";
  question?: string | null;
};

type IconSpec = { icon: typeof AimOutlined; className: string };

function iconOf(node: string): IconSpec {
  if (node === "region") return { icon: EnvironmentOutlined, className: styles.iconRegion };
  if (node === "clarify") return { icon: QuestionCircleOutlined, className: styles.iconClarify };
  if (node === "agent") return { icon: ThunderboltOutlined, className: styles.iconAgent };
  if (node === "tools") return { icon: SearchOutlined, className: styles.iconTools };
  if (node === "finalize") return { icon: EditOutlined, className: styles.iconFinalize };
  if (node === "review") return { icon: SafetyCertificateOutlined, className: styles.iconReview };
  return { icon: RightOutlined, className: styles.iconDefault };
}

function StepRow({ step }: { step: StepEvent }) {
  const { icon: Icon, className } = iconOf(step.node);
  return (
    <li className={styles.step}>
      <span className={`${styles.stepIcon} ${className}`}>
        <Icon />
      </span>
      <div className={styles.stepBody}>
        <span className={styles.stepNode}>{nodeLabel(step.node)}</span>
        <span className={styles.stepDetail}>{stepDetail(step)}</span>
      </div>
    </li>
  );
}

export function StepTrace({ steps, streaming, mode, elapsedMs, variant = "inline", question = null }: Props) {
  const [override, setOverride] = useState<boolean | null>(null);
  const open = override !== null ? override : streaming;

  if (variant === "panel") {
    return (
      <div className={styles.panel}>
        {question ? <p className={styles.panelQuestion}>{question}</p> : null}
        {mode === "once" ? (
          <p className={styles.empty}>这一次走的是一次性请求（/qa），没有步骤流。</p>
        ) : steps.length === 0 ? (
          <p className={styles.empty}>{streaming ? "等待第一个步骤…" : "这次提问没有步骤记录。"}</p>
        ) : (
          <ol className={styles.steps}>
            {steps.map((step, index) => (
              <StepRow key={`${step.node}-${index}`} step={step} />
            ))}
          </ol>
        )}
        <p className={styles.panelFoot}>
          {streaming ? "进行中…" : elapsedMs !== null && elapsedMs !== undefined ? `本轮耗时 ${(elapsedMs / 1000).toFixed(1)} 秒` : "本轮已收尾"}
        </p>
      </div>
    );
  }

  if (steps.length === 0) return null;

  return (
    <section className={styles.inline}>
      <button type="button" className={styles.head} onClick={() => setOverride(!open)} aria-expanded={open}>
        <span className={streaming ? styles.liveDot : styles.doneDot} aria-hidden="true" />
        <span className={styles.headTitle}>处理轨迹</span>
        <span className={styles.headMeta}>{streaming ? "进行中" : `${steps.length} 步`}</span>
        {open ? <UpOutlined className={styles.headCaret} /> : <DownOutlined className={styles.headCaret} />}
      </button>
      {open ? (
        <ol className={styles.steps}>
          {steps.map((step, index) => (
            <StepRow key={`${step.node}-${index}`} step={step} />
          ))}
        </ol>
      ) : null}
    </section>
  );
}
