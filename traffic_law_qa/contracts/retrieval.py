from __future__ import annotations

from dataclasses import dataclass

from .disk import ParentChunk

__all__ = [
    "REGION_UNKNOWN",
    "RewrittenQuery",
    "Query",
    "RetrievedArticle",
    "RetrievalResult",
    "WebFinding",
    "MaterialPassage",
]

REGION_UNKNOWN = "?"


@dataclass(frozen=True)
class RewrittenQuery:

    original: str
    expanded: str
    matched_aliases: tuple[str, ...] = ()
    expansions: tuple[str, ...] = ()
    law_hints: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.expansions)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "expanded": self.expanded,
            "matched_aliases": list(self.matched_aliases),
            "expansions": list(self.expansions),
            "law_hints": list(self.law_hints),
        }


@dataclass(frozen=True)
class Query:

    text: str
    top_k: int = 6
    candidates: int = 20
    use_vector: bool = True
    use_bm25: bool = True
    law_filter: tuple[str, ...] = ()
    channel_debug: bool = False

    def normalized(self) -> "Query":
        top_k = max(1, self.top_k)
        return Query(
            text=self.text.strip(),
            top_k=top_k,
            candidates=max(top_k, self.candidates),
            use_vector=self.use_vector,
            use_bm25=self.use_bm25,
            law_filter=self.law_filter,
            channel_debug=self.channel_debug,
        )


@dataclass(frozen=True)
class RetrievedArticle:

    article: ParentChunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    hit_chunks: tuple[str, ...] = ()
    law_hint: str | None = None

    @property
    def citation(self) -> str:
        return f"{self.article.citation}{self.article.article_no}"

    def to_dict(self) -> dict:
        return {
            "parent_id": self.article.parent_id,
            "citation": self.citation,
            "score": round(self.score, 6),
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "vector_score": None if self.vector_score is None else round(self.vector_score, 6),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 4),
            "hit_chunks": list(self.hit_chunks),
            "law_hint": self.law_hint,
        }


@dataclass(frozen=True)
class RetrievalResult:

    query: str
    articles: tuple[RetrievedArticle, ...]
    used_vector: bool
    used_bm25: bool
    elapsed_ms: float
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.articles

    def citations(self) -> list[str]:
        return [a.citation for a in self.articles]

    def render(self) -> str:
        lines = [
            f"查询：{self.query}",
            f"通道：向量={'开' if self.used_vector else '关'} BM25={'开' if self.used_bm25 else '关'} "
            f"| 耗时 {self.elapsed_ms:.0f}ms | 命中 {len(self.articles)} 条",
        ]
        for index, hit in enumerate(self.articles, start=1):
            raw = []
            if hit.bm25_score is not None:
                raw.append(f"BM25分 {hit.bm25_score:.2f}")
            if hit.vector_score is not None:
                raw.append(f"向量分 {hit.vector_score:.3f}")
            suffix = f" [{' '.join(raw)}]" if raw else ""
            hint = f" +法名线索「{hit.law_hint}」" if hit.law_hint else ""
            lines.append(
                f"  {index}. {hit.score:.4f}  {hit.citation}  "
                f"(向量#{hit.vector_rank} BM25#{hit.bm25_rank}){suffix}{hint}"
            )
            lines.append(f"     {hit.article.text.replace(chr(10), ' ')[:80]}…")
        for note in self.notes:
            lines.append(f"  提示：{note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "used_vector": self.used_vector,
            "used_bm25": self.used_bm25,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "notes": list(self.notes),
            "articles": [a.to_dict() for a in self.articles],
        }


@dataclass(frozen=True)
class WebFinding:

    label: str
    title: str
    url: str
    snippet: str
    site: str = ""
    published: str = ""

    def render(self) -> str:
        head = f"{self.label} {self.title}"
        if self.site:
            head += f"（{self.site}）"
        if self.published:
            head += f" {self.published}"
        return f"{head}\n{self.snippet}\n{self.url}"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "site": self.site,
            "published": self.published,
        }


@dataclass(frozen=True)
class MaterialPassage:

    label: str
    doc_id: str
    display_name: str
    index: int
    text: str
    score: float = 0.0

    @property
    def citation(self) -> str:
        return f"{self.display_name} 第 {self.index} 段"

    def render(self) -> str:
        return f"{self.label} {self.citation}\n{self.text}"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "doc_id": self.doc_id,
            "display_name": self.display_name,
            "index": self.index,
            "text": self.text,
            "score": round(self.score, 6),
        }
