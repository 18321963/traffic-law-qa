from __future__ import annotations

from ..contracts import Answer, QaError, Question, RetrievalResult
from ..ready import ReadyState, ensure_ready, stale_reason

__all__ = ["qa", "QaError", "ReadyState", "ensure_ready", "render", "stale_reason", "MODE_ASK", "MODE_SEARCH"]

MODE_ASK = "ask"
MODE_SEARCH = "search"


def qa(
    question: str,
    *,
    mode: str = MODE_ASK,
    top_k: int | None = None,
    debug: bool = False,
    with_vector: bool = True,
    rebuild: bool = False,
) -> Answer | RetrievalResult:
    if mode not in (MODE_ASK, MODE_SEARCH):
        raise QaError(f"未知 mode：{mode!r}，可选 {MODE_ASK!r} 或 {MODE_SEARCH!r}")
    if not question or not question.strip():
        raise QaError("问题为空")

    state = ensure_ready(with_vector=with_vector, rebuild=rebuild)
    _announce(state, debug)

    from ..qa.rag import LegalRAG

    rag = LegalRAG.load(with_vector=with_vector)
    if mode == MODE_SEARCH:
        return rag.search(question, top_k=top_k, channel_debug=debug)

    return rag.ask(Question(text=question, top_k=top_k), channel_debug=debug)


_announced = False


def _announce(state: ReadyState, debug: bool) -> None:
    global _announced
    if debug and not _announced:
        print(f"[qa] {state.describe()}")
    _announced = True


def render(result: Answer | RetrievalResult, *, debug: bool = False) -> str:
    if isinstance(result, RetrievalResult):
        return result.render()
    text = result.render()
    if debug and result.retrieval is not None:
        text += "\n\n" + result.retrieval.render()
    return text
