from __future__ import annotations

from rag_contracts.domain.retrieval import LOG_SEARCH, RetrievalResult, RetrievedArticle

__all__ = ["merge_retrievals"]


def merge_retrievals(logs: list[dict], *, question: str, max_evidence: int) -> RetrievalResult:
    ordered: list[RetrievedArticle] = []
    seen: set[str] = set()

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
            ordered.append(RetrievedArticle.from_dict(item))

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

    return RetrievalResult(
        query=question,
        articles=tuple(ordered),
        used_vector=any(bool(row.get("used_vector")) for row in logs),
        used_bm25=any(bool(row.get("used_bm25")) for row in logs),
        elapsed_ms=sum(float(row.get("elapsed_ms") or 0.0) for row in logs),
        notes=tuple(notes),
    )
