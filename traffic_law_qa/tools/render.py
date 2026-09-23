from __future__ import annotations

from ..contracts import ParentChunk, RetrievalResult

CHAPEAU_CHARS = 40
MIN_BODY_CHARS = 40

_DROP_NOTES = ("法名线索：",)
"""不进提示词的 note 前缀。

`法名线索：X（命中法规的分数 ×1.5）` 讲的是检索层怎么调分，模型据此动不了任何事。
`口语对齐：A → B` 留着 —— 它说的是模型自己的问法被怎么理解，那是能据以改写的反馈。

**只在渲染这层过滤，不动 retriever 的产出**：同一批 note 还要进 `Answer.notes`
给人看（`--json`、轨迹、`RetrievalResult.render()`），在源头删掉等于把人的那份也删了。
"""


def _cite(article: ParentChunk) -> str:
    return f"《{article.law_name}》{article.article_no}"


def _bigrams(text: str) -> set[str]:
    cleaned = "".join(ch for ch in text if ch.strip())
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)} or {cleaned}


def _chapeau(text: str, limit: int) -> str:
    head = text[:limit]
    cut = head.rfind("：")
    return head[: cut + 1] if cut >= 0 else head


def _snippet(text: str, query: str, width: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= width:
        return text

    grams = _bigrams(query)
    if not grams or not query.strip():
        return text[:width] + "…"

    counts: dict[str, int] = {}
    for i in range(len(text) - 1):
        gram = text[i : i + 2]
        if gram in grams:
            counts[gram] = counts.get(gram, 0) + 1

    marks = [0.0] * len(text)
    for i in range(len(text) - 1):
        gram = text[i : i + 2]
        if gram in counts:
            marks[i] = 1.0 / counts[gram]
    prefix = [0.0] * (len(text) + 1)
    for i, mark in enumerate(marks):
        prefix[i + 1] = prefix[i] + mark

    if prefix[-1] == 0:
        return text[:width] + "…"

    best_start, best_score = 0, -1
    for start in range(0, len(text) - width + 1):
        score = prefix[start + width] - prefix[start]
        if score > best_score:
            best_start, best_score = start, score

    window_end = best_start + width + 1

    if best_start <= CHAPEAU_CHARS:
        end = min(len(text), window_end)
        return f"{text[:end]}{'…' if end < len(text) else ''}"

    head = _chapeau(text, CHAPEAU_CHARS)
    body_len = width - len(head) - 1
    if body_len < MIN_BODY_CHARS:
        return f"…{text[best_start:window_end]}…"

    tail = "…" if best_start + body_len < len(text) else ""
    return f"{head}…{text[best_start : best_start + body_len]}{tail}"


def render_tool_result(
    result: RetrievalResult,
    *,
    index: int,
    seen: set[str] | None = None,
    snippet_chars: int = 120,
    match_text: str | None = None,
) -> str:
    seen = seen or set()
    total = len(result.articles)
    fresh = sum(1 for a in result.articles if a.article.parent_id not in seen)
    lines = [f"检索#{index}「{result.query}」｜命中 {total} 条（新增 {fresh} / 已知 {total - fresh}）"]
    if not total:
        lines.append("没有命中的法条。请换一组更接近法条原文的关键词，或补上具体违法情形与地点后重试。")
    elif not fresh:
        lines.append("⚠ 本轮没有新增法条，与之前的检索重复。换个完全不同的角度，或直接结束检索。")

    emitted = set(seen)
    for order, hit in enumerate(result.articles, start=1):
        known = hit.article.parent_id in emitted
        emitted.add(hit.article.parent_id)
        tail = "（上一轮已给过原文）" if known else ""
        lines.append(f"【{order}】{_cite(hit.article)}{tail}")
        if not known:
            lines.append(f"    {_snippet(hit.article.text, match_text or result.query, snippet_chars)}")
        if hit.article.refs:
            lines.append(f"    相关条：{'、'.join(hit.article.refs)}")

    for note in result.notes:
        if not note.startswith(_DROP_NOTES):
            lines.append(f"提示：{note}")
    return "\n".join(lines)
