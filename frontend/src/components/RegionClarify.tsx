import { useState } from "react";
import { Button, Input } from "antd";
import { EnvironmentOutlined } from "@ant-design/icons";
import type { InterruptValue } from "../api/types";
import styles from "./RegionClarify.module.css";

export const USAGE =
  "地区澄清卡：由 interrupt 帧驱动；按「地区」或「只按全国法」走 /qa/resume/stream 恢复，也可以换个地区名或直接跳过（跳过澄清会作废）。";

type Props = {
  value: InterruptValue | null;
  busy: boolean;
  onResume: (region: string) => void;
  onDismiss: () => void;
};

export function RegionClarify({ value, busy, onResume, onDismiss }: Props) {
  const [custom, setCustom] = useState("");
  const place = value?.place?.trim() || "";
  const laws = value?.laws ?? [];
  const message = value?.message?.trim() || "先确认按哪里的规定回答。";
  const submitCustom = () => {
    const region = custom.trim();
    if (region === "" || busy) return;
    onResume(region);
  };
  return (
    <section className={styles.clarify}>
      <div className={styles.head}>
        <EnvironmentOutlined className={styles.headIcon} />
        <span className={styles.headTitle}>需要确认 · 按哪里的规定作答？</span>
      </div>
      <p className={styles.message}>{message}</p>
      {laws.length > 0 ? <p className={styles.laws}>库内地方性法规：{laws.join("、")}</p> : null}
      <div className={styles.choices}>
        {place ? (
          <Button type="primary" className={styles.primary} disabled={busy} onClick={() => onResume(place)}>
            按「{place}」作答
          </Button>
        ) : null}
        <Button disabled={busy} onClick={() => onResume("national")}>
          只按全国法
        </Button>
      </div>
      <div className={styles.custom}>
        <Input
          value={custom}
          placeholder="别的地区名"
          aria-label="地区名"
          disabled={busy}
          onChange={(event) => setCustom(event.target.value)}
          onPressEnter={submitCustom}
        />
        <Button type="text" color="primary" disabled={busy || custom.trim() === ""} onClick={submitCustom}>
          按该地区作答
        </Button>
      </div>
      <button type="button" className={styles.skip} disabled={busy} onClick={onDismiss}>
        先不恢复，直接问新问题
      </button>
      <p className={styles.note}>恢复走 /qa/resume/stream；挂着澄清的会话直接提新问题，澄清会作废（此后恢复会得到 409）。</p>
    </section>
  );
}
