"""Agent 模式的实现在这里；**入口不在这里** —— 门是包级的 `../__main__.py`。

    python -m traffic_law_qa "问题" --agent [选项]

`--agent` 由门摘掉之后才调进来（`main(argv)` 拿到的是摘干净的那串），所以本模块不认
`--agent`，也没有自己的 `-h` —— 说明书与公共旗标都在门里那一份 `USAGE`（两处各写一份
必漂）。这里只留 agent 自己的旗标，`_parser()` 负责把不认识的开关与取不到值的开关变成
错误；文件末尾那个 `__main__` 块不是入口，是老命令的指路闸。

原本的 `--linear` 已退役：不带 `--agent` 的线性路径就是基线本身，而且走的是 `api.qa()`
那条带一致性检查与自动重建的路，比这里的 `runner.rag.ask()` 更该作对照组。
"""

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
    """只做校验的解析器。

    `-h/--help` 与「一个参数都不给」由门开头那个分支接住（打印那一份 `USAGE`，无参数
    返回 2），所以这里 `add_help=False`。这个解析器的唯一职责是**把不认识的开关和取不到
    值的开关变成错误**，而不是像原先的 `option()` 那样静默忽略。
    """
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
    # 兜底的 `Tracer()` 不能省：`tracer=None` 对 `load()` 的含义是「没指定，去问 .env」，
    # 而 `--no-langfuse` 要的正好是相反的事。
    tracer = langfuse or recorder or Tracer()

    try:
        runner = AgentRunner.load(
            with_vector=not options.no_vector,
            cfg=cfg,
            top_k=options.top_k,
            tracer=tracer,
        )
    except Exception as exc:  # noqa: BLE001 - Milvus 未起、索引未建等
        print(f"装配失败：{exc}")
        print("提示：先 docker compose up -d standalone，并确认索引已建（python -m traffic_law_qa.pipeline index）")
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
    if answer.usage:
        print(f"\n[tokens] {answer.usage}")
    return 0


# 老习惯敲 `-m traffic_law_qa.agent.cli` 的兜底：不给这个闸的话模块级代码跑完就退 0，
# 敲的人以为问到了答案（`…agent` 那条路径本身会报 No module named，见 README）。
if __name__ == "__main__":
    raise SystemExit('已收口：请用 python -m traffic_law_qa "问题" --agent（清单见 README「所有入口」）')
