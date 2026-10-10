import { useState } from "react";
import { Button, Popover } from "antd";
import { DownOutlined, ReloadOutlined, UpOutlined } from "@ant-design/icons";
import type { HealthState } from "../state/health";
import styles from "./StatusBar.module.css";

export const USAGE =
  "健康状态条：药丸按钮点开 Popover，先给一句人话结论，再列 /health 的原始口径；探活失败按连不上处理。";

type Props = {
  health: HealthState;
  mode: string;
};

function headlineOf(health: HealthState): { tone: string; text: string } {
  if (health.status === "loading") return { tone: "wait", text: "探活中…" };
  if (health.status === "error") return { tone: "down", text: `服务不健康：${health.error ?? "未知原因"}` };
  const rag = health.data?.rag;
  if (health.status === "degraded") {
    if (!rag || rag.status !== "ok") return { tone: "down", text: `知识库不可达：${rag?.detail ?? "未给原因"}` };
    const reasons: string[] = [];
    if (rag.degraded) reasons.push(`权重指纹 ${rag.weights ?? "unknown"}`);
    if (rag.llm_ready === false) reasons.push("生成层不可用（没有模型密钥，答问会拒答）");
    if (typeof rag.channels === "string" && rag.channels !== "稠密+BM25") reasons.push(`通道降级为${rag.channels}`);
    if (reasons.length === 0) reasons.push("服务自报降级");
    return { tone: "warn", text: `降级：${reasons.join("；")}` };
  }
  return { tone: "ok", text: `服务正常 · 通道 ${rag?.channels ?? "未知"}` };
}

export function StatusBar({ health, mode }: Props) {
  const [open, setOpen] = useState(false);
  const { tone, text } = headlineOf(health);
  const rag = health.data?.rag;
  const rows: [string, string][] = [
    ["数据来源", mode === "live" ? "真实服务" : "内置示例数据"],
    ["agent 状态", health.data?.status ?? "—"],
    ["版本", health.data?.version ?? "—"],
    ["生成层", rag?.llm_ready === undefined ? "—" : rag.llm_ready ? "可用" : "不可用"],
    ["检索通道", rag?.channels ?? "—"],
    ["重排模型", rag?.rerank === false || rag?.rerank === undefined ? "未启用" : String(rag.rerank)],
    ["权重指纹", rag?.weights ?? "—"],
    ["索引动作", rag?.index ?? "—"],
    ["法条 / 行数", rag ? `${rag.laws ?? "—"} / ${rag.rows ?? "—"}` : "—"],
    ["装配耗时", rag?.boot_ms === undefined ? "—" : `${Math.round(rag.boot_ms)} 毫秒`],
    ["最近一次探活", health.checkedAt ? new Date(health.checkedAt).toLocaleTimeString("zh-CN") : "—"],
  ];

  const panel = (
    <div className={styles.panel}>
      <dl className={styles.rows}>
        {rows.map(([label, value]) => (
          <div key={label} className={styles.row}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <Button size="small" icon={<ReloadOutlined />} onClick={() => health.refresh()}>
        立刻探活
      </Button>
    </div>
  );

  return (
    <Popover
      content={panel}
      trigger="click"
      open={open}
      onOpenChange={setOpen}
      placement="bottomRight"
      arrow={false}
      styles={{ body: { padding: 12 } }}
    >
      <button type="button" className={`${styles.pill} ${styles[tone]}`} aria-expanded={open}>
        <span className={styles.dot} aria-hidden="true" />
        <span className={styles.text}>{text}</span>
        {open ? <UpOutlined className={styles.caret} /> : <DownOutlined className={styles.caret} />}
      </button>
    </Popover>
  );
}
