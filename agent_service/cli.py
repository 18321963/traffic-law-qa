from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from rag_contracts import config

from . import container
from .trace import render_timing, render_trace

__all__ = ["main"]

USAGE = """用法：python -m agent_service "问题" [选项]

走完整 Agent 循环（入口判地区 → 规划 → 工具调用 → 复核 → 生成）。检索与生成都在 rag 服务里，
本进程只跑 agent 自己 —— 先把它起起来（默认 http://127.0.0.1:8000，可用 --http 换地方）。

    --http URL       rag 服务地址（默认取 RAG_BASE_URL，没配则 http://127.0.0.1:8000）
    --top-k N        覆盖默认召回条数 / 证据条数（默认取 rag 服务的 RAG_TOP_K，6）
    --max-steps N    规划轮数上限，默认 3。设 1 可做近似基线的 A/B
    --trace          打印完整决策链（**这本来就是默认**；写出来只为让默认可点名，
                     给不给都一样。想反过来——要 JSON 也要轨迹——目前没有开关）
    --timing         额外打印每个节点的耗时表，走 stderr（不污染 --json）
    --langfuse       把节点状态、模型调用与检索上报到 Langfuse 云端
                     （**这本来就是默认** —— .env 里配了 LANGFUSE_PUBLIC_KEY /
                     LANGFUSE_SECRET_KEY 就自动开，写出来只为让默认可点名，同 --trace。
                     还需 pip install -e ".[langfuse]"；没配就一个字节都不外发）
    --no-langfuse    强制关掉上报（跑批时不想让每一题都往云端灌就加这个）
    --json           输出 JSON 而不是人读文本

决策链在终端里会给「没取到」那种行上色（黄）；重定向到文件时自动不上色。

示例：
    python -m agent_service "在深圳，不按规定使用安全带罚多少？" --timing
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_service", add_help=False)
    parser.add_argument("question", nargs="?", help="用户问题")
    parser.add_argument("--http", default=None, help="rag 服务地址")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--langfuse", action="store_true")
    parser.add_argument("--no-langfuse", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _print_timing(result: object) -> None:
    retrieval = getattr(result, "retrieval", None)
    if retrieval is None:
        return
    parts = []
    usage = getattr(result, "usage", None)
    if usage:
        parts.append(f"tokens {usage}")
    parts.append(f"检索 {retrieval.elapsed_ms:.0f}ms")
    parts.append(f"总计 {getattr(result, 'elapsed_ms', 0.0):.0f}ms")
    print(f"\n[{' | '.join(parts)}]")


def _run(options: argparse.Namespace) -> int:
    from rag_contracts.observability.langfuse import from_env
    from rag_contracts.observability.tracer import Recorder, Tracer

    overrides = {}
    if options.max_steps:
        overrides["max_steps"] = options.max_steps
    cfg = replace(config.agent_config(), **overrides) if overrides else None

    color = sys.stdout.isatty()
    langfuse = None if options.no_langfuse else from_env()
    recorder = Recorder() if options.timing else None
    tracer = langfuse or recorder or Tracer()

    client = container.build_client(base_url=options.http, top_k=options.top_k)
    try:
        runner = container.boot_agent_runner(client, cfg=cfg, tracer=tracer)
    except Exception as exc:  # noqa: BLE001
        print(f"装配失败：{exc}")
        print(f"提示：先起 rag 服务（{config.COMPOSE} up -d app），agent 的检索与生成都在它那边")
        client.close()
        return 1

    print(f"[agent] {runner.describe()}")

    try:
        state = runner.invoke(options.question)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    finally:
        tracer.flush()
        client.close()
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
    print(answer.render())
    if answer.usage:
        print(f"\n[tokens] {answer.usage}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if any(a in ("-h", "--help") for a in args):
        print(USAGE)
        return 0
    if not args:
        print(USAGE)
        return 2

    try:
        options = _parser().parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    if not options.question:
        print(USAGE)
        return 2
    return _run(options)


if __name__ == "__main__":
    raise SystemExit(main())
