"""一条命令看完全貌：python examples/quickstart.py

对 4 个典型问题依次跑「确保索引就绪 → 混合检索 → 生成」，打印：

    问题 → 命中的法条号（含被稠密向量 / BM25 哪一路捞到）→ 答案要点

加 --search 只跑检索，不调用大模型：秒出、不花钱，调检索时用这个。

调用的就是对外那一个接口：traffic_law_rag.qa()
"""

from __future__ import annotations

import sys
from pathlib import Path

# 未安装包时也能从仓库根目录直接跑（python examples/quickstart.py）：
# 脚本所在目录是 examples/，仓库根不在 sys.path 上。装过就自然能导入，这行无害。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traffic_law_rag import QaError, qa  # noqa: E402
from traffic_law_rag.contracts import Answer, RetrievalResult  # noqa: E402

# 4 个问题各盯着一个已知的技术点：
#   1) 口语词「醉驾」需要改写成法条用语「醉酒驾驶」
#   2) 问深圳的事，不能被国家法律的一般条款盖过（法名线索加成）
#   3) 同一部特别条例的召回
#   4) 最普通的一条：确认基线没被前三个改动影响
QUESTIONS = (
    "醉驾怎么处罚",
    "深圳 行人 在机动车道 罚款多少",
    "智能网联汽车 道路测试要满足什么条件",
    "开车不系安全带怎么处罚",
)

RULE = "─" * 72
TOP_N = 3


def _ranking(result: RetrievalResult) -> list[str]:
    """每条命中：条号 + 被哪一路捞到、排第几。"""
    lines = []
    for index, hit in enumerate(result.articles[:TOP_N], start=1):
        marks = []
        if hit.vector_rank is not None:
            marks.append(f"向量#{hit.vector_rank}")
        if hit.bm25_rank is not None:
            marks.append(f"BM25#{hit.bm25_rank}")
        suffix = f"  ({' '.join(marks)})" if marks else ""
        hint = f"  法名线索「{hit.law_hint}」" if hit.law_hint else ""
        lines.append(f"    {index}. {hit.citation}{suffix}{hint}")
    return lines


def _digest(text: str, *, limit: int = 5) -> list[str]:
    """答案要点：前几行非空文字。"""
    rows = [row.strip() for row in text.strip().splitlines() if row.strip()]
    shown = rows[:limit] + (["…"] if len(rows) > limit else [])
    return [f"    {row}" for row in shown]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    search_only = "--search" in args

    print(f"交通法规问答 ｜ {len(QUESTIONS)} 个问题 ｜ {'只检索' if search_only else '检索 + 生成'}")
    print("首次运行若索引缺失会自动建库（约 30~60 秒），已就绪则直接复用\n")

    failures = 0
    for index, question in enumerate(QUESTIONS, start=1):
        print(RULE)
        print(f"【{index}】{question}")
        try:
            result = qa(question, mode="search" if search_only else "ask", debug=True)
        except QaError as exc:
            print(f"    {exc}")
            failures += 1
            continue

        retrieval = result if isinstance(result, RetrievalResult) else result.retrieval
        if retrieval is not None:
            print("  命中：")
            print("\n".join(_ranking(retrieval)))
            for note in retrieval.notes:
                print(f"  提示：{note}")
        if isinstance(result, Answer):
            print("  答案：")
            print("\n".join(_digest(result.text)))

    print(RULE)
    print(f"完成：{len(QUESTIONS) - failures}/{len(QUESTIONS)} 个问题拿到结果")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
