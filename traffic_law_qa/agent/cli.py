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
from ..obs import Recorder
from .graph import AgentRunner
from .langfuse_tracer import LangfuseTracer
from .trace import render_timing, render_trace

__all__ = ["main", "USAGE"]


USAGE = """用法：python -m traffic_law_qa.agent "问题" [选项]

  --trace          打印完整决策链（**这本来就是默认**；写出来只为让默认可点名，
                   给不给都一样。想反过来——要 JSON 也要轨迹——目前没有开关）
  --json           输出 JSON（Answer.to_dict）
  --timing         额外打印每个节点的耗时表，走 stderr（不污染 --json）
  --max-steps N    规划轮数上限，默认 2。设 1 可做近似基线的 A/B
  --top-k N        证据条数上限，默认取 RAG_TOP_K
  --no-vector      只用 BM25 检索
  --langfuse       把节点状态、模型调用与检索上报到 Langfuse 云端
                   （需 .env 里配 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，
                   并先装 pip install -e ".[langfuse]"；不配就只是不打这份报告）
  --linear         走单轮管道作对照（不发规划轮请求）

决策链在终端里会对判别行上色（够了=绿，还不够/未取到=黄）；重定向到文件时自动不上色。
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
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _langfuse_tracer(enabled: bool) -> LangfuseTracer | None:
    """`--langfuse` → 一个真 tracer；任何一步不成就退回 None（= 不观测）。

    **失败只降级、不中断**：没配 key、没装 extra、SDK 版本不对，都只是这一次不上报。
    提示一律走 stderr —— `--json` 的 stdout 一个字节都不许被污染。
    打印只出现 host：两个 key 是秘密，任何一部分都不进日志。
    """
    if not enabled:
        return None
    cfg = config.langfuse_config()
    if not cfg.ready:
        print("--langfuse 忽略：未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY", file=sys.stderr)
        return None
    try:
        tracer = LangfuseTracer(cfg)
    except Exception as exc:  # noqa: BLE001 - 观测装不上，不该拦住提问
        print(f'--langfuse 忽略：{exc}（装法：pip install -e ".[langfuse]"）', file=sys.stderr)
        return None
    print(f"[langfuse] 观测已开启：{cfg.host}（key 不打印）", file=sys.stderr)
    return tracer


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
    langfuse = _langfuse_tracer(options.langfuse)
    recorder = Recorder() if options.timing else None
    tracer = langfuse or recorder

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
            if langfuse is not None:
                langfuse.flush()
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
