from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from rag_contracts.domain.laws import LawInfo
from rag_contracts.domain.retrieval import REGION_UNKNOWN
from rag_contracts.ports import LLM

from ..prompts import REGION_SYSTEM_PROMPT
from .state import AgentState

__all__ = [
    "REGION_UNKNOWN",
    "HISTORY_ANSWER_CHARS",
    "law_scope",
    "national_law_ids",
    "make_region_node",
]

HISTORY_ANSWER_CHARS = 300


def national_law_ids(laws: Sequence[LawInfo]) -> tuple[str, ...]:
    return tuple(sorted({law.law_id for law in laws if not law.local}))


def law_scope(region: str, laws: Sequence[LawInfo]) -> tuple[str, ...]:
    if not region or region == REGION_UNKNOWN:
        return ()
    if region in {law.law_name for law in laws}:
        return ()
    local = tuple(sorted({law.law_id for law in laws if region in law.law_name}))
    if not local:
        return ()
    return tuple(sorted(set(national_law_ids(laws)) | set(local)))


def _parse_region(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            region = str(data.get("region") or "").strip() or REGION_UNKNOWN
            place = str(data.get("place") or "").strip()
            return region, place
    return REGION_UNKNOWN, ""


def _clip_answer(text: str) -> str:
    if len(text) <= HISTORY_ANSWER_CHARS:
        return text
    return text[:HISTORY_ANSWER_CHARS] + "…"


def _pairs_to_entries(pairs: Iterable[Sequence[str]]) -> list[dict]:
    entries: list[dict] = []
    pending: str | None = None
    for role, content in pairs:
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            entries.append({"question": pending, "answer": content})
            pending = None
    return entries


def _build_history(
    conversation: Iterable[dict],
    fallback: Iterable[Sequence[str]],
    turns: int,
) -> list[tuple[str, str]]:
    entries = [entry for entry in conversation if isinstance(entry, dict)]
    if not entries:
        entries = _pairs_to_entries(fallback)
    history: list[tuple[str, str]] = []
    for entry in entries[-turns:] if turns > 0 else []:
        history.append(("user", str(entry.get("question") or "")))
        history.append(("assistant", _clip_answer(str(entry.get("answer") or ""))))
    return history


def make_region_node(llm: LLM, laws: Sequence[LawInfo], *, history_turns: int = 5):
    from langgraph.types import Overwrite

    prompt_head = REGION_SYSTEM_PROMPT % {
        "law_count": len(laws),
        "laws": "\n".join(f"- {law.law_name}" for law in laws),
    }

    def region_node(state: AgentState) -> dict:
        reset: dict = {
            "messages": Overwrite([]),
            "search_log": Overwrite([]),
            "external": Overwrite([]),
            "materials": Overwrite([]),
            "steps": Overwrite(0),
            "usage": Overwrite([]),
            "truncated": Overwrite(0),
            "answer": None,
            "history": _build_history(
                state.get("conversation") or (),
                state.get("history") or (),
                history_turns,
            ),
        }
        if not llm.available:
            return {**reset, "region": REGION_UNKNOWN, "region_scope": (), "place": ""}

        prompt: list[dict] = [{"role": "system", "content": prompt_head}]
        prompt.extend({"role": role, "content": content} for role, content in reset["history"])
        prompt.append({"role": "user", "content": state["question"]})

        reply, _usage, _finish = llm.chat(prompt, temperature=0.0, name="llm.region")
        region, place = _parse_region(reply.get("content") or "")
        return {**reset, "region": region, "region_scope": law_scope(region, laws), "place": place}

    return region_node
