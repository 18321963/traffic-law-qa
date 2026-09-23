from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from .. import config
from ..obs import Recorder, Tracer
from .graph import AgentRunner
from .langfuse_tracer import from_env
from .trace import render_timing, render_trace

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m traffic_law_qa --agent", add_help=False)
    parser.add_argument("question", help="用户问题")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--langfuse", action="store_true")
    parser.add_argument("--no-langfuse", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    try:
        options = _parser().parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    question = options.question

    overrides = {}
    if options.max_steps:
        overrides["max_steps"] = options.max_steps
    cfg = replace(config.agent_config(), **overrides) if overrides else None

    color = sys.stdout.isatty()
    langfuse = None if options.no_langfuse else from_env()
    recorder = Recorder() if options.timing else None
    tracer = langfuse or recorder or Tracer()

    try:
        runner = AgentRunner.load(
            with_vector=not options.no_vector,
            cfg=cfg,
            top_k=options.top_k,
            tracer=tracer,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"装配失败：{exc}")
        print("提示：Milvus 没起就先 docker compose up -d standalone；索引与 docx 不一致会自动重建"
              "（要手动重建：python -m traffic_law_qa.pipeline build）")
        return 1

    print(f"[agent] {runner.describe()}")

    try:
        state = runner.invoke(question)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    finally:
        tracer.flush()
    if not options.json:
        print(render_trace(state, color=color))
    answer = state.get("answer")

    if options.timing and isinstance(tracer, Recorder):
        print(render_timing(tracer.summary(), color=color), file=sys.stderr)

    if answer is None:
        print("未产出答案")
        return 1

    if options.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print()
    print(answer.text)
    if answer.evidences:
        print()
        print("依据：")
        for evidence in answer.evidences:
            print(f"  {evidence.label} {evidence.citation}")
    if answer.timeliness:
        print()
        print("时效提示（联网检索，非本库法条）：")
        for finding in answer.timeliness:
            print(f"  {finding.label} {finding.title} {finding.url}")
    if answer.materials:
        print()
        print("本次会话材料（未入知识库，仅供参照）：")
        for passage in answer.materials:
            print(f"  {passage.label} {passage.citation}")
    if answer.usage:
        print(f"\n[tokens] {answer.usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit('已收口：请用 python -m traffic_law_qa "问题" --agent（清单见 README「入口」）')
