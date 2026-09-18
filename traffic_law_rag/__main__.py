"""`python -m traffic_law_rag "问题"` —— 一条命令拿到答案（等价于一次 `qa()` 调用）。

    python -m traffic_law_rag "醉驾怎么处罚"
    python -m traffic_law_rag "深圳 行人 在机动车道 罚款多少" --search --debug

失败只打印一行中文提示（含该执行的命令），不糊 pymilvus 的堆栈。
"""

from __future__ import annotations

import json
import sys

from .api import MODE_SEARCH, QaError, qa, render
from .contracts import Answer

USAGE = """用法：
    python -m traffic_law_rag "问题" [选项]

选项：
    --search        只检索，不调用大模型（不花 LLM 的钱）
    --debug         打印每条法条被稠密向量 / BM25 各自排到第几
    --top-k N       覆盖默认召回条数（默认取 RAG_TOP_K，6）
    --no-vector     只用 BM25 通道
    --rebuild       无视一致性检查强制重建索引（改了切分/解析逻辑后用）
    --json          输出 JSON 而不是人读文本

示例：
    python -m traffic_law_rag "醉驾怎么处罚"
    python -m traffic_law_rag "深圳 行人 在机动车道 罚款多少" --search --debug
"""


def _parse(args: list[str]) -> tuple[str | None, int | None, set[str]]:
    """拆出（问题, top_k, 开关集合）。"""
    question: str | None = None
    top_k: int | None = None
    flags: set[str] = set()
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--top-k":
            index += 1
            if index < len(args) and args[index].isdigit():
                top_k = int(args[index])
        elif token.startswith("--"):
            flags.add(token)
        elif question is None:
            question = token
        index += 1
    return question, top_k, flags


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
    if not args:
        print(USAGE)
        return 2
    if args[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    question, top_k, flags = _parse(args)
    if not question:
        print(USAGE)
        return 2

    debug = "--debug" in flags
    try:
        result = qa(
            question,
            mode=MODE_SEARCH if "--search" in flags else "ask",
            top_k=top_k,
            debug=debug,
            with_vector="--no-vector" not in flags,
            rebuild="--rebuild" in flags,
        )
    except QaError as exc:
        print(str(exc))  # 一行中文提示即可，不要把堆栈糊在脸上
        return 1

    if "--json" in flags:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(render(result, debug=debug))
    if isinstance(result, Answer):
        _print_timing(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
