from __future__ import annotations

import re

from ..contracts import ParentChunk, RetrievalResult, RetrievedArticle
from ..kb.law_parser import cn_to_int


def resolve_law_id(
    law_name: str, parents: dict[str, ParentChunk]
) -> tuple[str | None, list[str]]:
    table = {p.law_name: p.law_id for p in parents.values()}
    names = sorted(table)
    name = (law_name or "").strip()
    if not name:
        return None, names

    if name in table:
        return table[name], names

    suffixed = [n for n in names if n.endswith(name)]
    if len(suffixed) == 1:
        return table[suffixed[0]], names

    contained = [n for n in names if name in n or n in name]
    if len(contained) == 1:
        return table[contained[0]], names
    return None, names


RE_ARTICLE_CN = re.compile(r"第\s*([零一二三四五六七八九十百千]+)\s*条")
RE_ARTICLE_AR = re.compile(r"第\s*(\d{1,4})\s*条")
RE_LAW_TITLE = re.compile(r"《([^》]{2,80})》")

_LAW_PREFIX = "中华人民共和国"


def parse_article_no(raw: str) -> int | None:
    text = (raw or "").strip()
    if not text:
        return None
    text = text.removeprefix("第").removesuffix("条").strip()
    if text.isdigit():
        return int(text)
    try:
        return cn_to_int(text)
    except ValueError:
        return None


def build_article_index(
    parents: dict[str, ParentChunk],
) -> dict[tuple[str, int], ParentChunk]:
    index: dict[tuple[str, int], ParentChunk] = {}
    for parent in parents.values():
        number = parse_article_no(parent.article_no)
        if number is None:
            continue
        index.setdefault((parent.law_id, number), parent)
    return index


def _law_names_in(text: str) -> list[str]:
    return [m.group(1).strip() for m in RE_LAW_TITLE.finditer(text)]


def find_article(
    text: str,
    *,
    parents: dict[str, ParentChunk],
    index: dict[tuple[str, int], ParentChunk],
) -> ParentChunk | None:
    match = RE_ARTICLE_CN.search(text) or RE_ARTICLE_AR.search(text)
    if match is None:
        return None
    number = parse_article_no(match.group(0))
    if number is None:
        return None

    bracketed = {
        law_id
        for name in _law_names_in(text)
        if (law_id := resolve_law_id(name, parents)[0]) is not None
    }
    if len(bracketed) == 1:
        return index.get((bracketed.pop(), number))
    if bracketed:
        return None

    spans: list[tuple[int, int, str]] = []
    for law_id, law_name in {(p.law_id, p.law_name) for p in parents.values()}:
        for needle in {law_name, law_name.removeprefix(_LAW_PREFIX)}:
            start = text.find(needle)
            while start >= 0:
                spans.append((start, start + len(needle), law_id))
                start = text.find(needle, start + 1)
    outer = [
        span
        for span in spans
        if not any(
            other[0] <= span[0] and span[1] <= other[1] and other[1] - other[0] > span[1] - span[0]
            for other in spans
        )
    ]
    law_ids = {law_id for _, _, law_id in outer}
    if len(law_ids) == 1:
        return index.get((law_ids.pop(), number))
    if law_ids:
        return None

    candidates = [chunk for (_, no), chunk in index.items() if no == number]
    return candidates[0] if len(candidates) == 1 else None


def _single(parent: ParentChunk, *, query: str) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        articles=(RetrievedArticle(article=parent, score=1.0),),
        used_vector=False,
        used_bm25=False,
        elapsed_ms=0.0,
        notes=("精确取条，未走向量/BM25 检索",),
    )


def _out_of_range(
    parents: dict[str, ParentChunk], law_id: str, number: int
) -> str:
    numbers = [
        n
        for p in parents.values()
        if p.law_id == law_id and (n := parse_article_no(p.article_no)) is not None
    ]
    law_name = next(p.law_name for p in parents.values() if p.law_id == law_id)
    if not numbers:
        return f"《{law_name}》里没有第 {number} 条。"
    return (
        f"《{law_name}》共 {len(numbers)} 条"
        f"（第 {min(numbers)} 条..第 {max(numbers)} 条），没有第 {number} 条。"
    )


def lookup_article(
    article_no: str,
    law_name: str | None,
    *,
    parents: dict[str, ParentChunk],
    index: dict[tuple[str, int], ParentChunk],
) -> tuple[RetrievalResult | None, str]:
    number = parse_article_no(article_no)
    if number is None:
        return None, (
            f"无法识别条号「{article_no}」。请给出「第九十条」或「90」这样的形式。"
        )

    if law_name:
        law_id, known = resolve_law_id(law_name, parents)
        if law_id is None:
            return None, (
                f"无法确定法规「{law_name}」。库内只有这几部，请从中选一个"
                f"（或省略 law_name 检索全部）：{'；'.join(known)}"
            )
        parent = index.get((law_id, number))
        if parent is None:
            return None, _out_of_range(parents, law_id, number)
        return _single(parent, query=article_no), ""

    candidates = [chunk for (_, no), chunk in index.items() if no == number]
    if not candidates:
        return None, f"库里没有任何一部法规有第 {number} 条。"
    if len(candidates) > 1:
        names = "；".join(sorted(chunk.law_name for chunk in candidates))
        return None, (
            f"第 {number} 条在库内有 {len(candidates)} 部法规都有：{names}。"
            f"请指明是哪一部（用 law_name 参数）。"
        )
    return _single(candidates[0], query=article_no), ""
