from __future__ import annotations

from rag_contracts.domain.answer import Answer
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.retrieval import RetrievalResult

from ..indexing.readiness import ReadyState
from ..query.dispatch import MODE_ASK, run

__all__ = ["qa", "QaError", "ReadyState", "render"]


def qa(
    question: str,
    *,
    mode: str = MODE_ASK,
    top_k: int | None = None,
    debug: bool = False,
    with_vector: bool = True,
    rebuild: bool = False,
) -> Answer | RetrievalResult:
    if not question or not question.strip():
        raise QaError("问题为空")

    from .. import container

    state = container.readiness(with_vector=with_vector, rebuild=rebuild)
    _announce(state, debug)
    rag = container.build_rag(with_vector=with_vector)
    return run(rag, question, mode=mode, top_k=top_k, channel_debug=debug)


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
