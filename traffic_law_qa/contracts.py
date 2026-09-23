from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Iterable


class QaError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paragraph:

    index: int
    text: str
    style: str | None = None


@dataclass(frozen=True)
class Heading:

    no: str
    title: str

    @property
    def display(self) -> str:
        return f"{self.no} {self.title}".strip()

    def to_dict(self) -> dict:
        return {"no": self.no, "title": self.title}

    @classmethod
    def from_dict(cls, data: dict) -> "Heading":
        return cls(no=data["no"], title=data["title"])


@dataclass(frozen=True)
class Article:

    article_no: str
    article_index: int
    chapter: str | None
    section: str | None
    text: str
    paragraphs: tuple[str, ...]
    refs: tuple[str, ...]

    @property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "article_no": self.article_no,
            "article_index": self.article_index,
            "chapter": self.chapter,
            "section": self.section,
            "text": self.text,
            "paragraphs": list(self.paragraphs),
            "refs": list(self.refs),
            "paragraph_count": self.paragraph_count,
            "char_count": self.char_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Article":
        return cls(
            article_no=data["article_no"],
            article_index=data["article_index"],
            chapter=data.get("chapter"),
            section=data.get("section"),
            text=data["text"],
            paragraphs=tuple(data.get("paragraphs") or ()),
            refs=tuple(data.get("refs") or ()),
        )


@dataclass(frozen=True)
class LawDocument:

    law_id: str
    law_name: str
    version: str
    source_file: str
    citation: str
    articles: tuple[Article, ...]
    chapters: tuple[Heading, ...] = ()
    sections: tuple[Heading, ...] = ()
    toc: tuple[str, ...] = ()
    preamble: tuple[str, ...] = ()
    leftovers: tuple[str, ...] = ()

    @property
    def article_count(self) -> int:
        return len(self.articles)

    @property
    def char_count(self) -> int:
        return sum(a.char_count for a in self.articles)

    def article_map(self) -> dict[str, Article]:
        return {a.article_no: a for a in self.articles}

    def to_dict(self) -> dict:
        return {
            "law_id": self.law_id,
            "law_name": self.law_name,
            "version": self.version,
            "source_file": self.source_file,
            "citation": self.citation,
            "article_count": self.article_count,
            "char_count": self.char_count,
            "toc": list(self.toc),
            "preamble": list(self.preamble),
            "chapters": [c.to_dict() for c in self.chapters],
            "sections": [s.to_dict() for s in self.sections],
            "leftovers": list(self.leftovers),
            "articles": [a.to_dict() for a in self.articles],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LawDocument":
        return cls(
            law_id=data["law_id"],
            law_name=data["law_name"],
            version=data["version"],
            source_file=data.get("source_file", ""),
            citation=data.get("citation") or f"《{data['law_name']}》({data['version']})",
            articles=tuple(Article.from_dict(a) for a in data.get("articles", [])),
            chapters=tuple(Heading.from_dict(c) for c in data.get("chapters", [])),
            sections=tuple(Heading.from_dict(s) for s in data.get("sections", [])),
            toc=tuple(data.get("toc") or ()),
            preamble=tuple(data.get("preamble") or ()),
            leftovers=tuple(data.get("leftovers") or ()),
        )


@dataclass(frozen=True)
class ParentChunk:

    parent_id: str
    law_id: str
    law_name: str
    version: str
    citation: str
    article_no: str
    article_index: int
    chapter: str | None
    section: str | None
    text: str
    refs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["refs"] = list(self.refs)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ParentChunk":
        return cls(
            parent_id=data["parent_id"],
            law_id=data["law_id"],
            law_name=data["law_name"],
            version=data["version"],
            citation=data["citation"],
            article_no=data["article_no"],
            article_index=data["article_index"],
            chapter=data.get("chapter"),
            section=data.get("section"),
            text=data["text"],
            refs=tuple(data.get("refs") or ()),
        )


@dataclass(frozen=True)
class Chunk:

    chunk_id: str
    parent_id: str
    law_id: str
    law_name: str
    version: str
    citation: str
    article_no: str
    article_index: int
    part_index: int
    part_total: int
    text: str
    embed_text: str
    chapter: str | None = None
    section: str | None = None
    refs: tuple[str, ...] = ()

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["refs"] = list(self.refs)
        data["char_count"] = self.char_count
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            parent_id=data["parent_id"],
            law_id=data["law_id"],
            law_name=data["law_name"],
            version=data["version"],
            citation=data["citation"],
            article_no=data["article_no"],
            article_index=data["article_index"],
            part_index=data.get("part_index", 0),
            part_total=data.get("part_total", 1),
            text=data["text"],
            embed_text=data.get("embed_text") or data["text"],
            chapter=data.get("chapter"),
            section=data.get("section"),
            refs=tuple(data.get("refs") or ()),
        )


@dataclass(frozen=True)
class ChunkSet:

    parents: tuple[ParentChunk, ...]
    chunks: tuple[Chunk, ...]
    stats: dict[str, Any] = field(default_factory=dict)

    def parent_map(self) -> dict[str, ParentChunk]:
        return {p.parent_id: p for p in self.parents}

    def write(self, chunks_path: Path, parents_path: Path) -> None:
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(chunks_path, (c.to_dict() for c in self.chunks))
        _write_jsonl(parents_path, (p.to_dict() for p in self.parents))

    @classmethod
    def read(cls, chunks_path: Path, parents_path: Path) -> "ChunkSet":
        parents = tuple(ParentChunk.from_dict(row) for row in _read_jsonl(parents_path))
        chunks = tuple(Chunk.from_dict(row) for row in _read_jsonl(chunks_path))
        stats = {
            "parents": len(parents),
            "chunks": len(chunks),
            "laws": len({p.law_id for p in parents}),
            "chars": sum(c.char_count for c in chunks),
        }
        return cls(parents=parents, chunks=chunks, stats=stats)


@dataclass(frozen=True)
class IndexStats:

    collection: str
    uri: str
    rows: int
    dense_rows: int
    sparse_rows: int
    embedding_model: str | None
    embedding_dim: int | None
    vector_enabled: bool
    built_at: str
    elapsed_ms: float
    schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexStats":
        known = {field.name for field in dataclass_fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


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
                f"  {index}. RRF {hit.score:.4f}  {hit.citation}  "
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
class Evidence:

    label: str
    citation: str
    text: str
    score: float
    article: ParentChunk

    def render(self) -> str:
        return f"{self.label} {self.citation}\n{self.text}"


@dataclass(frozen=True)
class Question:

    text: str
    history: tuple[tuple[str, str], ...] = ()
    top_k: int | None = None


@dataclass(frozen=True)
class Review:

    score: float | None
    threshold: float
    total: int
    supported: int
    unsupported: tuple[str, ...]
    original_text: str
    model: str
    passed: bool

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "threshold": self.threshold,
            "total": self.total,
            "supported": self.supported,
            "unsupported": list(self.unsupported),
            "passed": self.passed,
            "model": self.model,
            "original_text": self.original_text,
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


@dataclass(frozen=True)
class Answer:

    question: str
    text: str
    evidences: tuple[Evidence, ...]
    model: str
    elapsed_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalResult | None = None
    notes: tuple[str, ...] = ()
    review: Review | None = None
    """末端复核的结果；`None` = 没跑复核（阈值关掉、或线性管道这条路根本没有复核）。"""

    timeliness: tuple[WebFinding, ...] = ()
    """联网检索到的时效性信息，标 `[时效N]`。`()` = 本次没走网搜。

    **不进 `[依据N]` 那条复核判据** —— `review.CITE_RE` 只认「依据」，这两块天然不拉低分母。
    """

    materials: tuple[MaterialPassage, ...] = ()
    """本次会话上传材料的片段，标 `[材料N]`。`()` = 没有上传件。

    未入知识库，所以与时效提示同样不进复核判据；差别只在产品口径：入库了才进依据链。
    """

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "model": self.model,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "usage": self.usage,
            "notes": list(self.notes),
            "citations": [e.citation for e in self.evidences],
            "retrieval": None if self.retrieval is None else self.retrieval.to_dict(),
            "review": None if self.review is None else self.review.to_dict(),
            "timeliness": [w.to_dict() for w in self.timeliness],
            "materials": [m.to_dict() for m in self.materials],
        }

    def render(self, *, show_citations: bool = True) -> str:
        parts = [self.text.strip()]
        if show_citations and self.evidences:
            parts.append("")
            parts.append("参考文献：")
            for e in self.evidences:
                snippet = e.text.replace("\n", " ")
                parts.append(f"  {e.label} {e.citation} — {snippet[:60]}…")
        if self.timeliness:
            parts.append("")
            parts.append("时效提示（联网检索，非本库法条）：")
            for w in self.timeliness:
                snippet = w.snippet.replace("\n", " ")
                parts.append(f"  {w.label} {w.title} — {snippet[:60]}…")
                parts.append(f"      {w.url}")
        if self.materials:
            parts.append("")
            parts.append("本次会话材料（未入知识库，仅供参照）：")
            for m in self.materials:
                snippet = m.text.replace("\n", " ")
                parts.append(f"  {m.label} {m.citation} — {snippet[:60]}…")
        if self.notes:
            parts.append("")
            parts += [f"注：{n}" for n in self.notes]
        return "\n".join(parts)


@dataclass(frozen=True)
class StageReport:

    name: str
    input_desc: str
    output_desc: str
    ok: bool
    elapsed_ms: float
    detail: str = ""
    skipped: bool = False


@dataclass(frozen=True)
class PipelineReport:

    stages: tuple[StageReport, ...]
    law_count: int
    article_count: int
    chunk_count: int
    index: IndexStats | None = None

    @property
    def elapsed_ms(self) -> float:
        return sum(s.elapsed_ms for s in self.stages)

    def render(self) -> str:
        width = max(len(s.name) for s in self.stages) if self.stages else 10
        lines = []
        for stage in self.stages:
            mark = "skip" if stage.skipped else ("ok" if stage.ok else "FAIL")
            lines.append(
                f"  [{mark:>4}] {stage.name:<{width}}  {stage.input_desc} → {stage.output_desc}"
                f"  {stage.elapsed_ms:>8.1f}ms  {stage.detail}"
            )
        lines.append(
            f"  法规 {self.law_count} 部 / 条文 {self.article_count} 条 / 索引块 {self.chunk_count} 个"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class CorpusStats:

    articles: int
    chunks: int
    dense: bool | None
    collection: str


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件：{path}（请先运行切块阶段）")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


__all__ = [
    "QaError",
    "Paragraph",
    "Heading",
    "Article",
    "LawDocument",
    "ParentChunk",
    "Chunk",
    "ChunkSet",
    "IndexStats",
    "RewrittenQuery",
    "Query",
    "RetrievedArticle",
    "RetrievalResult",
    "Evidence",
    "Question",
    "Review",
    "Answer",
    "StageReport",
    "PipelineReport",
    "CorpusStats",
]
