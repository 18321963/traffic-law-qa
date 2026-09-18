"""层间数据契约（整条管道的唯一真源）。

每个阶段只认相邻两层的契约对象，不关心对方的实现：

| 层 | 类 | 输入 | 输出 |
|----|----|------|------|
| read     | `DocxReader`       | `Path`                        | `list[Paragraph]` |
| parse    | `LawParser`        | `list[Paragraph]`             | `LawDocument` |
| chunk    | `LawChunker`       | `LawDocument`                 | `ChunkSet` |
| index    | `Indexer`          | `ChunkSet`                    | `IndexStats` |
| retrieve | `HybridRetriever`  | `Query`                       | `RetrievalResult` |
| generate | `AnswerGenerator`  | `Question` + `RetrievalResult`| `Answer` |

契约对象自己负责 JSON 读写（`to_dict` / `from_dict`），所以磁盘格式的变更
只需要改这一个文件。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ============================================================ 1. read 层
@dataclass(frozen=True)
class Paragraph:
    """docx 里的一个段落（原样文本 + 段落样式名）。"""

    index: int
    text: str
    style: str | None = None


# ============================================================ 2. parse 层
@dataclass(frozen=True)
class Heading:
    """章 / 节标题。"""

    no: str      # 例：第一章
    title: str   # 例：总则（已去掉排版用全角空格）

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
    """一条法条：定位信息 + 原文 + 款 + 交叉引用。"""

    article_no: str            # 例：第十九条
    article_index: int         # 例：19（1-based，按正文出现顺序）
    chapter: str | None        # 所属章，例：第二章 车辆和驾驶人
    section: str | None        # 所属节，无则为 None
    text: str                  # 完整条文（多款用 \n 连接，含条号前缀）
    paragraphs: tuple[str, ...]  # 逐款原文，[0] 含"第X条"前缀
    refs: tuple[str, ...]      # 本条内提到的其他条号，例：("第九十九条",)

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
    """一部法规的完整结构：法 → 章 → 节 → 条。"""

    law_id: str
    law_name: str
    version: str                # 例：2021-04-29
    source_file: str
    citation: str               # 例：《中华人民共和国道路交通安全法》(2021-04-29)
    articles: tuple[Article, ...]
    chapters: tuple[Heading, ...] = ()
    sections: tuple[Heading, ...] = ()
    toc: tuple[str, ...] = ()       # 目录条目原文
    preamble: tuple[str, ...] = ()  # 正文前的标题 / 修订说明 / 目录
    leftovers: tuple[str, ...] = ()  # 归不到任何条文的正文段落（应为空）

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


# ============================================================ 3. chunk 层
@dataclass(frozen=True)
class ParentChunk:
    """条级父块：检索命中子块后，回灌给 LLM 的就是它。"""

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
    """款级子块：向量库与 BM25 的索引单元。"""

    chunk_id: str
    parent_id: str
    law_id: str
    law_name: str
    version: str
    citation: str
    article_no: str
    article_index: int
    part_index: int            # 第几款，从 0 开始
    part_total: int            # 该条共几款
    text: str                  # 原文（展示用）
    embed_text: str            # 带法名/章/条号前缀的向量化文本
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
    """一次切块的完整产物：父块 + 子块 + 统计。"""

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


# ============================================================ 4. index 层
@dataclass(frozen=True)
class IndexStats:
    """建索引的结果摘要，落到 index/index_meta.json。"""

    collection: str
    chunk_count: int
    bm25_docs: int
    vector_count: int
    embedding_model: str | None
    embedding_dim: int | None
    vector_enabled: bool
    built_at: str
    elapsed_ms: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexStats":
        return cls(**data)


# ============================================================ 5. rewrite 层
@dataclass(frozen=True)
class RewrittenQuery:
    """检索前的查询改写结果（口语词对齐 + 法名线索）。"""

    original: str
    expanded: str                                # 实际送去检索的文本（原文 + 法条用语）
    matched_aliases: tuple[str, ...] = ()        # 命中的口语词
    expansions: tuple[str, ...] = ()             # 实际追加的法条用语
    law_hints: tuple[str, ...] = ()              # 命中法名的片段

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
    """检索请求。"""

    text: str
    top_k: int = 6
    candidates: int = 20
    use_vector: bool = True
    use_bm25: bool = True
    law_filter: tuple[str, ...] = ()   # 只在这些 law_id 内检索；空 = 全库

    def normalized(self) -> "Query":
        return Query(
            text=self.text.strip(),
            top_k=max(1, self.top_k),
            candidates=max(self.top_k, self.candidates),
            use_vector=self.use_vector,
            use_bm25=self.use_bm25,
            law_filter=self.law_filter,
        )


@dataclass(frozen=True)
class RetrievedArticle:
    """一条被召回的父块（法条）+ 融合分数 + 各通道排名。"""

    article: ParentChunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    hit_chunks: tuple[str, ...] = ()   # 命中的子块 id
    law_hint: str | None = None        # 命中的法名片段（用于法名加成）

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
    """检索结果集 —— 这是 RAG 工具的标准输出。"""

    query: str
    articles: tuple[RetrievedArticle, ...]
    used_vector: bool
    used_bm25: bool
    elapsed_ms: float
    notes: tuple[str, ...] = ()   # 降级/警告信息，例：向量未启用

    @property
    def is_empty(self) -> bool:
        return not self.articles

    def citations(self) -> list[str]:
        return [a.citation for a in self.articles]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "used_vector": self.used_vector,
            "used_bm25": self.used_bm25,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "notes": list(self.notes),
            "articles": [a.to_dict() for a in self.articles],
        }


# ============================================================ 6. generate 层
@dataclass(frozen=True)
class Evidence:
    """喂给 LLM 的一条证据（= 一条法条）。"""

    label: str                 # 【依据1】
    citation: str              # 《…法》(版本) 第十九条
    text: str
    score: float
    article: ParentChunk

    def render(self) -> str:
        return f"{self.label} {self.citation}\n{self.text}"


@dataclass(frozen=True)
class Question:
    """生成请求：问题 + 可选的多轮上下文。"""

    text: str
    history: tuple[tuple[str, str], ...] = ()   # ((role, content), ...)
    top_k: int | None = None


@dataclass(frozen=True)
class Answer:
    """管道的最终输出。"""

    question: str
    text: str
    evidences: tuple[Evidence, ...]
    model: str
    elapsed_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalResult | None = None
    notes: tuple[str, ...] = ()

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
        }

    def render(self, *, show_citations: bool = True) -> str:
        parts = [self.text.strip()]
        if show_citations and self.evidences:
            parts.append("")
            parts.append("参考文献：")
            for e in self.evidences:
                snippet = e.text.replace("\n", " ")
                parts.append(f"  {e.label} {e.citation} — {snippet[:60]}…")
        if self.notes:
            parts.append("")
            parts += [f"注：{n}" for n in self.notes]
        return "\n".join(parts)


# ============================================================ 7. 编排层
@dataclass(frozen=True)
class StageReport:
    """单个阶段的执行结果。"""

    name: str
    input_desc: str
    output_desc: str
    ok: bool
    elapsed_ms: float
    detail: str = ""
    skipped: bool = False


@dataclass(frozen=True)
class PipelineReport:
    """整条管道的执行结果。"""

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


# ============================================================ jsonl 工具
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
    "Answer",
    "StageReport",
    "PipelineReport",
]
