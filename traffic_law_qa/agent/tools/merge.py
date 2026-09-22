"""N 次检索 → 一个 `RetrievalResult`：收尾节点与生成层之间的唯一接口。

单独成文件是因为它是**跨工具**的那一步 —— search_law 与 get_article 的结果都进同一个
`search_log`，由它去重、交错、截断，再交给既有的生成器。
"""

from __future__ import annotations

from ...contracts import ParentChunk, RetrievalResult, RetrievedArticle


def merge_retrievals(
    logs: list[dict],
    *,
    question: str,
    parents: dict[str, ParentChunk],
    max_evidence: int,
) -> RetrievalResult:
    """把 N 次检索的结果合并成一个 `RetrievalResult`。

    **这是收尾节点与生成层之间的唯一接口。** 有了它，`AnswerGenerator.generate()`
    可以原样接收 Agent 的证据 —— 提示词不是「保持一致」，而是「就是同一段代码」。

    合并规则：

    1. **去重**：同一条法条被多次命中只留一条（首次出现的那条）。
    2. **轮转交错排序**，不按分数排：query1 的 #1、query2 的 #1、query3 的 #1、
       query1 的 #2… 理由是**跨查询的 RRF 分数不可比** —— 不同查询的通道数可能不同，
       而且命中法名线索的那次检索整条被乘了 1.5（见 retriever 的 law_hint_boost）。
       按分数排会让「查询里带了法规名」的那次系统性压过其他次，而那未必是更相关的那次。
       轮转交错是纯排名操作，不需要跨查询比分数，且保证每次检索的头名都进得来。
    3. **截断**到 `max_evidence`（默认与 `top_k` 相同 —— 证据槽位和基线一样多，
       hit@k 才谈得上直接可比）。

    正文从 `parents` 回灌（`parent_id` → `ParentChunk`）。回灌不到就跳过并记一条 note，
    不静默丢依据 —— 那意味着 Milvus 里的块和本地块文件不同步，是需要被看见的异常。
    """
    ordered: list[RetrievedArticle] = []
    seen: set[str] = set()
    missing = 0

    depth = max((len(row.get("articles") or ()) for row in logs), default=0)
    for rank in range(depth):
        for row in logs:
            articles = row.get("articles") or ()
            if rank >= len(articles):
                continue
            item = articles[rank]
            parent_id = item.get("parent_id", "")
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)

            parent = parents.get(parent_id)
            if parent is None:
                missing += 1
                continue
            ordered.append(
                RetrievedArticle(
                    article=parent,
                    score=item.get("score", 0.0),
                    vector_rank=item.get("vector_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    vector_score=item.get("vector_score"),
                    bm25_score=item.get("bm25_score"),
                    hit_chunks=tuple(item.get("hit_chunks") or ()),
                    law_hint=item.get("law_hint"),
                )
            )

    deduped = len(ordered)
    truncated = max(0, deduped - max_evidence)
    if max_evidence > 0:
        ordered = ordered[:max_evidence]

    notes: list[str] = []
    for number, row in enumerate(logs, start=1):
        for note in row.get("notes") or ():
            notes.append(f"检索#{number}：{note}")
    if len(logs) > 1:
        notes.append(f"共 {len(logs)} 轮工具调用，合并去重后 {deduped} 条")
    if truncated:
        notes.append(f"证据按排名截断至 {max_evidence} 条（合并后共 {deduped} 条）")
    if missing:
        notes.append(f"有 {missing} 条命中无法回灌原文（索引与本地块文件不同步），已跳过")

    return RetrievalResult(
        query=question,
        articles=tuple(ordered),
        used_vector=any(bool(row.get("used_vector")) for row in logs),
        used_bm25=any(bool(row.get("used_bm25")) for row in logs),
        elapsed_ms=sum(float(row.get("elapsed_ms") or 0.0) for row in logs),
        notes=tuple(notes),
    )
