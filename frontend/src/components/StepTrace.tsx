import { nodeLabel, stepDetail } from "../render";
import type { StepEvent } from "../api/types";

type Props = {
  steps: StepEvent[];
  streaming: boolean;
  mode: string;
  elapsedMs: number | null;
};

export function StepTrace({ steps, streaming, mode, elapsedMs }: Props) {
  return (
    <section className="trace">
      <header className="trace__head">
        <h3>处理轨迹</h3>
        {streaming ? <span className="trace__live">进行中</span> : null}
      </header>
      {mode === "once" ? (
        <p className="trace__empty">这一次走的是一次性请求（/qa），没有步骤流。</p>
      ) : steps.length === 0 ? (
        <p className="trace__empty">{streaming ? "等待第一个步骤…" : "这次提问没有步骤记录。"}</p>
      ) : (
        <ol className="trace__list">
          {steps.map((step, index) => (
            <li key={`${step.node}-${index}`} className={`trace__item trace__item--${step.node}`}>
              <div className="trace__node">{nodeLabel(step.node)}</div>
              <div className="trace__detail">{stepDetail(step)}</div>
            </li>
          ))}
        </ol>
      )}
      {elapsedMs !== null && !streaming ? (
        <p className="trace__foot">本轮耗时 {(elapsedMs / 1000).toFixed(1)} 秒</p>
      ) : null}
    </section>
  );
}
