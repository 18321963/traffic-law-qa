from __future__ import annotations

from ..contracts.answer import Answer, Question
from ..contracts.errors import QaError
from ..contracts.retrieval import RetrievalResult
from ..ports import RagService

__all__ = ["MODE_AGENT", "MODE_ASK", "MODE_SEARCH", "run"]

MODE_ASK = "ask"
MODE_SEARCH = "search"
MODE_AGENT = "agent"


def run(
    rag: RagService,
    question: str,
    *,
    mode: str = MODE_ASK,
    top_k: int | None = None,
    channel_debug: bool = False,
) -> Answer | RetrievalResult:
    if mode not in (MODE_ASK, MODE_SEARCH):
        raise QaError(f"未知 mode：{mode!r}，可选 {MODE_ASK!r} 或 {MODE_SEARCH!r}")
    retrieval = rag.search(question, top_k=top_k, channel_debug=channel_debug)
    if mode == MODE_SEARCH:
        return retrieval
    return rag.answer(Question(text=question, top_k=top_k), retrieval)
