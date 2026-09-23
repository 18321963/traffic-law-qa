from __future__ import annotations

import argparse
import json
import sys

from .api import MODE_SEARCH, QaError, qa, render
from .contracts import Answer

USAGE = """用法：python -m traffic_law_qa "问题" [选项]

默认走线性管道（确保索引就绪 → 检索 → 生成，也就是 `qa()`，评测的基线）；
加 `--agent` 走 Agent 循环（模型自己决定查什么、查几轮）。

HTTP 是另一个入口：python -m traffic_law_qa.main（/qa 问答、/documents 上传材料或入库、
/reindex 立刻重建索引）。命令行这边不认上传。

两条路都认：
    --top-k N       覆盖默认召回条数 / 证据条数（默认取 RAG_TOP_K，6）
    --no-vector     只用 BM25 通道
    --json          输出 JSON 而不是人读文本

线性（默认）另认：
    --search        只检索，不调用大模型（不花 LLM 的钱）
    --debug         打印每条法条被稠密向量 / BM25 各自排到第几
    --rebuild       无视一致性检查强制重建索引（改了切分/解析逻辑后用）

Agent（--agent）另认（解析在 agents/cli.py，这里只列名目）：
    --trace          打印完整决策链（**这本来就是默认**；写出来只为让默认可点名，
                     给不给都一样。想反过来——要 JSON 也要轨迹——目前没有开关）
    --timing         额外打印每个节点的耗时表，走 stderr（不污染 --json）
    --max-steps N    规划轮数上限，默认 3。设 1 可做近似基线的 A/B
    --langfuse       把节点状态、模型调用与检索上报到 Langfuse 云端
                     （**这本来就是默认** —— .env 里配了 LANGFUSE_PUBLIC_KEY /
                     LANGFUSE_SECRET_KEY 就自动开，写出来只为让默认可点名，同 --trace。
                     还需 pip install -e ".[langfuse]"；没配就一个字节都不外发）
    --no-langfuse    强制关掉上报（跑批时不想让每一题都往云端灌就加这个）

决策链在终端里会给「没取到」那种行上色（黄）；重定向到文件时自动不上色。

示例：
    python -m traffic_law_qa "醉驾怎么处罚"
    python -m traffic_law_qa "深圳 行人 在机动车道 罚款多少" --search --debug
    python -m traffic_law_qa "在深圳，不按规定使用安全带罚多少？" --agent --timing
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m traffic_law_qa", add_help=False)
    parser.add_argument("question", nargs="?", help="用户问题")
    parser.add_argument("--search", action="store_true", help="只检索，不调用大模型")
    parser.add_argument("--debug", action="store_true", help="打印双通道排名")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖默认召回条数")
    parser.add_argument("--no-vector", action="store_true", help="只用 BM25 通道")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
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


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if any(a in ("-h", "--help") for a in args):
        print(USAGE)
        return 0
    if not args:
        print(USAGE)
        return 2

    if "--agent" in args:
        args.pop(args.index("--agent"))
        from .agents.cli import main as agent_main

        return agent_main(args)

    try:
        options = _parser().parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    if not options.question:
        print(USAGE)
        return 2

    debug = options.debug
    try:
        result = qa(
            options.question,
            mode=MODE_SEARCH if options.search else "ask",
            top_k=options.top_k,
            debug=debug,
            with_vector=not options.no_vector,
            rebuild=options.rebuild,
        )
    except QaError as exc:
        print(str(exc))
        return 1

    if options.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(render(result, debug=debug))
    if isinstance(result, Answer):
        _print_timing(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
