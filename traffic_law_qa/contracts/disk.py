from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "Paragraph",
    "Heading",
    "Article",
    "LawDocument",
    "ParentChunk",
    "Chunk",
    "ChunkSet",
    "IndexStats",
]


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
