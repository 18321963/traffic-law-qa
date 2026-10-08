from __future__ import annotations

from rag_contracts.domain.answer import Answer, Question
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.retrieval import RetrievalResult
from rag_contracts.ports import RagService

__all__ = ["MODE_ASK", "MODE_SEARCH", "run"]

MODE_ASK = "ask"
MODE_SEARCH = "search"


def run(
    rag: RagService,
    question: str,
    *,
    mode: str = MODE_ASK,
    top_k: int | None = None,
    channel_debug: bool = False,
    law_filter: tuple[str, ...] = (),
    candidates: int | None = None,
) -> Answer | RetrievalResult:
    if mode not in (MODE_ASK, MODE_SEARCH):
        raise QaError(f"未知 mode：{mode!r}，可选 {MODE_ASK!r} 或 {MODE_SEARCH!r}")
    retrieval = rag.search(
        question,
        top_k=top_k,
        channel_debug=channel_debug,
        law_filter=law_filter,
        candidates=candidates,
    )
    if mode == MODE_SEARCH:
        return retrieval
    return rag.answer(Question(text=question, top_k=top_k), retrieval)
