import { useEffect, useState } from "react";
import type { TurnError } from "../state/conversation";

type Props = {
  error: TurnError;
  retryLabel: string;
  onRetry: () => void;
};

function waitLabel(seconds: number): string {
  if (seconds >= 600) return `${Math.ceil(seconds / 60)} 分钟`;
  return `${seconds} 秒`;
}

const RESUME_DEGRADED = "规划模型暂不可用";

function hintOf(error: TurnError): string {
  if (error.status === 429) {
    return error.message.startsWith("日额度") ? "日额度按 UTC 0 点重置；换 key 或等重置。" : "限流按分钟窗口计，等窗口过后再试。";
  }
  if (error.status === 503) {
    return error.message.includes(RESUME_DEGRADED)
      ? "恢复要重新规划才能带上你选的地区，服务选择不静默丢弃它。直接重新提问即可。"
      : "服务未就绪：容器可能还在启动，或装配失败（探活见右上角状态）。";
  }
  if (error.status === 409) return "这条会话没有等待中的澄清：可能已经答完，或挂着的中断被新问题作废了。";
  if (error.status === 400) return "会话存储不可用：这条服务实例没有接会话库，去掉跨轮记忆再问。";
  if (error.status === 401) return "鉴权没通过：检查设置里的 X-API-Key 是否与服务端配置的一致。";
  if (error.status === 422) return "请求体不合法：问题不能为空；恢复时的地区值不能是空串。";
  if (error.status === 500) return "服务内部错误：可稍后重试，或把这条消息转给运维。";
  if (error.kind === "network") return "连不上服务：确认基地址、容器是否在跑、本机是否走了代理。";
  if (error.kind === "parse") return "流的帧坏了：这条流已中断，重试会重新发起一次问答。";
  if (error.kind === "truncated") return "连接被中途掐断：可能是网络或服务重启。";
  if (error.kind === "frame") return "流内异常会以 error 帧收尾，这一轮不会有权威答案。";
  return "";
}

export function ErrorNotice({ error, retryLabel, onRetry }: Props) {
  const [left, setLeft] = useState(() => (error.retryAfter ? Math.ceil(error.retryAfter) : 0));

  useEffect(() => {
    if (!error.retryAfter) {
      setLeft(0);
      return;
    }
    setLeft(Math.ceil(error.retryAfter));
    const timer = window.setInterval(() => {
      setLeft((value) => {
        if (value <= 1) {
          window.clearInterval(timer);
          return 0;
        }
        return value - 1;
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [error.retryAfter]);

  const blocked = left > 0;
  const title = error.status ? `请求被拒（${error.status}）` : "这一轮没能完成";
  const hint = hintOf(error);
  return (
    <section className="failure" role="alert">
      <header className="failure__head">
        <span className="tag tag--wrong">出错</span>
        <span>{title}</span>
      </header>
      <p className="failure__message">{error.message}</p>
      {hint ? <p className="failure__hint">{hint}</p> : null}
      {blocked ? (
        <p className="failure__countdown">
          {error.status === 429 && error.message.startsWith("日额度") ? "距额度重置" : "距可重试"} {waitLabel(left)}
        </p>
      ) : null}
      <div className="failure__actions">
        <button type="button" className="btn" disabled={blocked} onClick={onRetry}>
          {blocked ? `${retryLabel}（${waitLabel(left)}后）` : retryLabel}
        </button>
      </div>
    </section>
  );
}
