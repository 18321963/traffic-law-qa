import { useState } from "react";
import type { InterruptValue } from "../api/types";

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
  const message = value?.message?.trim() || "这个问题涉及具体地区，先确认按哪里的规定回答。";
  return (
    <section className="clarify">
      <header className="clarify__head">
        <span className="tag tag--ask">需要确认</span>
        <span>按哪里的规定作答？</span>
      </header>
      <p className="clarify__message">{message}</p>
      {laws.length > 0 ? (
        <p className="clarify__laws">
          库内地方性法规：{laws.join("、")}
        </p>
      ) : null}
      <div className="clarify__choices">
        {place ? (
          <button type="button" className="btn btn--primary" disabled={busy} onClick={() => onResume(place)}>
            按「{place}」作答
          </button>
        ) : null}
        <button type="button" className="btn" disabled={busy} onClick={() => onResume("national")}>
          只按全国法
        </button>
        <form
          className="clarify__custom"
          onSubmit={(event) => {
            event.preventDefault();
            const region = custom.trim();
            if (region === "" || busy) return;
            onResume(region);
          }}
        >
          <input
            type="text"
            value={custom}
            placeholder="别的地区名"
            aria-label="地区名"
            onChange={(event) => setCustom(event.target.value)}
          />
          <button type="submit" className="btn btn--quiet" disabled={busy || custom.trim() === ""}>
            按该地区作答
          </button>
        </form>
        <button type="button" className="btn btn--quiet" disabled={busy} onClick={onDismiss}>
          先不恢复，直接问新问题
        </button>
      </div>
      <p className="clarify__note">
        恢复走 /qa/resume/stream；挂着澄清的会话直接提新问题，澄清会作废（此后恢复会得到 409）。
      </p>
    </section>
  );
}
