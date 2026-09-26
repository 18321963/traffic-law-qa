from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .disk import ParentChunk
from .retrieval import MaterialPassage, RetrievalResult, WebFinding

__all__ = [
    "Evidence",
    "Question",
    "Review",
    "Answer",
]


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
