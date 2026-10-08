from __future__ import annotations

import argparse
import json
import sys

from rag_contracts.domain.answer import Answer

from ..api.facade import QaError, qa, render
from ..query.dispatch import MODE_ASK, MODE_SEARCH

__all__ = ["main"]

USAGE = """用法：python -m rag_service "问题" [选项]

走线性管道（确保索引就绪 → 检索 → 生成，也就是 `qa()`，评测的基线）。
Agent 循环是另一个包：python -m agent_service "问题"（它按 RAG_BASE_URL 调本服务的 HTTP 面）。

HTTP 是另一个入口：python -m rag_service.cli.serve（/qa 问答、/documents 上传材料或入库、
/reindex 立刻重建索引）。命令行这边不认上传。

认得这些选项：
    --top-k N       覆盖默认召回条数 / 证据条数（默认取 RAG_TOP_K，6）
    --no-vector     只用 BM25 通道
    --json          输出 JSON 而不是人读文本
    --search        只检索，不调用大模型（不花 LLM 的钱）
    --debug         打印每条法条被稠密向量 / BM25 各自排到第几
    --rebuild       无视一致性检查强制重建索引（改了切分/解析逻辑后用）

示例：
    python -m rag_service "醉驾怎么处罚"
    python -m rag_service "深圳 行人 在机动车道 罚款多少" --search --debug
    python -m rag_service "在深圳，不按规定使用安全带罚多少？" --json
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag_service", add_help=False)
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
            mode=MODE_SEARCH if options.search else MODE_ASK,
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
