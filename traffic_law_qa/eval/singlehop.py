"""题集 A（82 道单跳域内题）上的 rag vs agent 对照。

    python -m traffic_law_qa.eval.singlehop --compare --limit 5 --out data/traces/single5.json
    python -m traffic_law_qa.eval.singlehop --compare --out data/traces/single82.json

**与多跳那套正相反**：题集 A 的 gold 只有一条法条，问题本身**不需要多跳**
（题面既不含条号也不含法名，构造时就滤掉了）。所以这里问的是
「Agent 在不需要它的题上加了多少、又花了多少」。

**两条口径必须分开报**，混在一起就是拿「多打几枪」冒充「打得准」：

  - 同预算：rag 单次 top-6   vs   agent **首轮** top-6
            —— 唯一能直接比 hit@k 的一格
  - 全预算：rag 单次         vs   agent **全部轮次合并**
            —— 候选条数不再相同（agent 最多 2×top_k），hit@k 必然虚高，
               所以这一格只报「查到几条 gold / 共几条」

跑法与落盘格式复用多跳那套（`multihop.trace`），两条臂怎么跑只此一处实现。
两侧都真调 LLM：rag 那侧要 ask 一次（为了「引用 gold」那一列），agent 那侧是多轮循环。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["run", "metrics", "main"]

KS = (1, 3, 6)


def run(
    *,
    limit: int | None = None,
    top_k: int = 6,
    out: Path | None = None,
    verbose: bool = True,
) -> list[dict]:
    """跑题集 A 的两条臂，落盘，返回与多跳那套同格式的行。"""
    from .. import config
    from .harness import _kb_index, build_cases
    from .multihop import trace

    resolver, _known, id_of = _kb_index()
    cases, _no_gold, _named = build_cases(config.EVAL_CORPUS_PATH, resolver, id_of)
    if not cases:
        from ..api import QaError

        raise QaError("题集 A 里没有可用的评测题")

    return trace(cases=cases, limit=limit, out=out, top_k=top_k, verbose=verbose)


def _rank(ids: list[str], gold: set[str]) -> int | None:
    """第一个 gold 的名次；None = 这一格没命中。与 harness 同一套口径。"""
    return next((i for i, parent_id in enumerate(ids, start=1) if parent_id in gold), None)


def _merged(searches: list[dict]) -> list[str]:
    """全部轮次按「先到先得」合并去重 —— 名次就是它第一次被捞到的位置。"""
    ids: list[str] = []
    seen: set[str] = set()
    for search in searches:
        for hit in search.get("articles") or ():
            if hit["parent_id"] not in seen:
                seen.add(hit["parent_id"])
                ids.append(hit["parent_id"])
    return ids


def _first(searches: list[dict]) -> list[str]:
    return [h["parent_id"] for h in searches[0]["articles"]] if searches else []


def metrics(rows: list[dict], *, top_k: int = 6) -> str:
    """三格 hit@k/MRR + agent 的循环成本，外加一句别误读的话。"""
    if not rows:
        return "（没有可统计的行）"

    arms = {"rag 单次": [], "agent 首轮": [], "agent 合并": []}
    candidates = dict.fromkeys(arms, 0)
    plans = searches = reviews = 0

    for row in rows:
        gold = set(row["gold_ids"])
        first = _first(row["agent"]["searches"])
        merged = _merged(row["agent"]["searches"])
        rag_ids = [h["parent_id"] for h in row["rag"]["retrieved"]]

        for name, ids in (("rag 单次", rag_ids), ("agent 首轮", first), ("agent 合并", merged)):
            arms[name].append(_rank(ids, gold))
            candidates[name] += len(ids)

        searches += len(row["agent"]["searches"])
        plans += len(row["agent"]["usage"])
        reviews += len(row["agent"]["reflections"])

    n = len(rows)
    lines = [
        f"单跳题 rag vs agent（{n} 题 ｜ top_k={top_k}）",
        "",
        f"  {'':<12}{'hit@1':>8}{'hit@3':>8}{'hit@6':>8}{'MRR':>8}{'候选/题':>9}",
    ]
    for name, ranks in arms.items():
        rates = [sum(1 for r in ranks if r is not None and r <= k) / n for k in KS]
        mrr = sum(1.0 / r for r in ranks if r is not None) / n
        cells = "".join(f"{rate:>8.1%}" for rate in rates)
        lines.append(f"  {name:<12}{cells}{mrr:>8.3f}{candidates[name] / n:>9.1f}")

    lines += [
        "",
        f"  agent 平均：{plans / n:.1f} 轮 LLM 规划 / {searches / n:.1f} 次检索 / {reviews / n:.1f} 轮审核",
        "",
        "  ⚠ 「agent 合并」的候选数是另两格的两倍，**不同分母，别直接比 hit@k** ——",
        "     它只说明查得多，不说明查得准。要判断 agent 强不强，看「agent 首轮」那一格。",
    ]
    return "\n".join(lines)


USAGE = __doc__


def _parser() -> argparse.ArgumentParser:
    """只做校验的解析器（`--help` 与「不带参数」由 main 开头那个分支打印 `USAGE`，
    所以 `add_help=False`）。

    它唯一的职责是把「不认识的开关」和「取不到值的开关」变成错误。原先的 `option()`
    只做 `if name in args`，两样都静默忽略 —— 而这里的安静是有价格的：
    `--limit` 敲错一个字母，`option()` 返回 None，`run(limit=None)` 就是**全量 82 道**，
    两条臂都真调 LLM。文档里那句「先跑 5 道看链路」于是变成一次全额付费。
    """
    parser = argparse.ArgumentParser(prog="python -m traffic_law_qa.eval.singlehop", add_help=False)
    parser.add_argument("--compare", action="store_true", help="跑两臂对照（不给这个就跑说明书，因为要花钱）")
    parser.add_argument("--quiet", action="store_true", help="不打印逐题进度")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 道（不给=全量 82 道）")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖默认召回条数（6）")
    parser.add_argument("--out", default=None, help="把逐题原始产物落盘到该路径")
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

    if not options.compare:
        print(USAGE)
        return 0

    from .multihop import full_budget_summary

    top_k = options.top_k or 6
    rows = run(
        limit=options.limit,
        top_k=top_k,
        out=Path(options.out) if options.out else None,
        verbose=not options.quiet,
    )
    print()
    print(metrics(rows, top_k=top_k))
    print()
    print(full_budget_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
