from __future__ import annotations

from collections.abc import Sequence

from rag_contracts.domain.laws import LawInfo

from .region import law_scope, national_law_ids
from .state import AgentState

__all__ = ["CLARIFY_MESSAGE", "make_clarify_node"]

CLARIFY_MESSAGE = (
    "这个问题还没拿准按哪里的规定回答，先确认一下："
    "回复一个地区名（如「深圳」）就按该地区法规加上全国法作答；"
    "回复 national 则只按全国法作答。"
)


def _resume_region(value: object) -> str:
    raw = value.get("region") if isinstance(value, dict) else ""
    cleaned = str(raw or "").strip()
    return cleaned or "national"


def make_clarify_node(laws: Sequence[LawInfo]):
    from langgraph.types import Overwrite, interrupt

    def clarify_node(state: AgentState) -> dict:
        value = interrupt(
            {
                "type": "region_clarify",
                "place": str(state.get("place") or ""),
                "message": CLARIFY_MESSAGE,
                "laws": [law.law_name for law in laws if law.local],
            }
        )
        region = _resume_region(value)
        scope = national_law_ids(laws) if region == "national" else law_scope(region, laws)
        note = ""
        if not scope:
            note = f"回复的「{region}」未匹配到库内地区，本次按全国法作答"
            region = "national"
            scope = national_law_ids(laws)
        return {
            "region": region,
            "region_scope": scope,
            "region_note": note,
            "usage_extra": Overwrite([]),
        }

    return clarify_node
