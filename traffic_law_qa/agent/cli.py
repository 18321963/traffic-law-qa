"""`python -m traffic_law_qa.agent "问题"` 的入口。

`agent/__main__.py` 只是一次转发，让包级 `-m` 路径能落到这里。

选项的文档在 `USAGE` 里，改参数时两处一起改。
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

__all__ = ["main", "USAGE"]


USAGE = """用法：python -m traffic_law_qa.agent "问题" [选项]

  --trace          打印完整决策链（**这本来就是默认**；写出来只为让默认可点名，
                   给不给都一样。想反过来——要 JSON 也要轨迹——目前没有开关）
  --json           输出 JSON（Answer.to_dict）
  --timing         额外打印每个节点的耗时表，走 stderr（不污染 --json）
  --max-steps N    规划轮数上限，默认 3。设 1 可做近似基线的 A/B
  --top-k N        证据条数上限，默认取 RAG_TOP_K
  --no-vector      只用 BM25 检索
  --langfuse       把节点状态、模型调用与检索上报到 Langfuse 云端
                   （**这本来就是默认** —— .env 里配了 LANGFUSE_PUBLIC_KEY /
                   LANGFUSE_SECRET_KEY 就自动开，写出来只为让默认可点名，同 --trace。
                   还需 pip install -e ".[langfuse]"；没配就一个字节都不外发）
  --no-langfuse    强制关掉上报（跑批时不想让每一题都往云端灌就加这个）
  --linear         走单轮管道作对照（不发规划轮请求）

决策链在终端里会给「没取到」那种行上色（黄）；重定向到文件时自动不上色。
"""


def _parser() -> argparse.ArgumentParser:
    """只做校验的解析器。

    `--help` 与「一个参数都不给」由 `main` 开头那个分支负责打印 `USAGE`，所以这里
    `add_help=False` —— argparse 自动生成的帮助不如那份 `USAGE` 写得全，不该抢它的活。
    这个解析器的唯一职责是**把不认识的开关和取不到值的开关变成错误**，
    而不是像原先的 `option()` 那样静默忽略。
    """
    parser = argparse.ArgumentParser(prog="python -m traffic_law_qa.agent", add_help=False)
    parser.add_argument("question", help="用户问题")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--langfuse", action="store_true")
    parser.add_argument("--no-langfuse", action="store_true")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args or not args:
        print(USAGE)
        return 0

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
        print("提示：先 docker compose up -d standalone，并确认索引已建（python -m traffic_law_qa.kb.indexer）")
        return 1

    print(f"[agent] {runner.describe()}")

    if options.linear:
        answer = runner.rag.ask(question)
    else:
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


if __name__ == "__main__":
    raise SystemExit(main())
