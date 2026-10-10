from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from rag_contracts.domain.answer import Answer
from rag_contracts.domain.retrieval import MaterialPassage, WebFinding

__all__ = ["AgentState"]


class AgentState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]
    session_id: str
    stream_tokens: bool
    top_k: int
    max_steps: int

    messages: Annotated[list[dict], operator.add]
    search_log: Annotated[list[dict], operator.add]
    external: Annotated[list[WebFinding], operator.add]
    materials: Annotated[list[MaterialPassage], operator.add]
    material_ids: list[str]
    steps: Annotated[int, operator.add]
    usage: Annotated[list[dict], operator.add]
    usage_extra: Annotated[list[dict], operator.add]
    truncated: Annotated[int, operator.add]
    call_keys: Annotated[list[str], operator.add]

    region: str
    region_scope: tuple[str, ...]
    place: str
    region_note: str

    conversation: Annotated[list[dict], operator.add]
    answer: Answer | None
