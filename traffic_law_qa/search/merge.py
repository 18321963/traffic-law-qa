from __future__ import annotations

from ..contracts.disk import ParentChunk
from ..contracts.retrieval import RetrievalResult, RetrievedArticle
from ..tools.render import LOG_SEARCH


def merge_retrievals(
    logs: list[dict],
    *,
    question: str,
    parents: dict[str, ParentChunk],
    max_evidence: int,
) -> RetrievalResult:
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
            notes.append(f"{LOG_SEARCH}{number}：{note}")
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
