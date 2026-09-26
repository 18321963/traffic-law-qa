from __future__ import annotations

import json

from ..contracts.disk import ParentChunk
from ..contracts.retrieval import REGION_UNKNOWN
from ..ports import LLM
from ..prompts import REGION_SYSTEM_PROMPT
from .state import AgentState

__all__ = [
    "REGION_UNKNOWN",
    "law_scope",
    "national_law_ids",
    "make_region_node",
]

_LOCAL_MARKERS = ("省", "市", "自治区", "经济特区")
"""法规名里带这些词 = 地方性法规。

判据放在**法规名**上而不是写死一份 law_id 清单：新增一部地方条例时，没人会记得来这里改。
代价是名字里含「市」却不针对某地的法规（《城市道路管理条例》那种）会被误判成地方性 ——
库内 6 部没有这种；真出现了再收紧成「『省』『市』前面还得有地名」。
"""


def national_law_ids(parents: dict[str, ParentChunk]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                parent.law_id
                for parent in parents.values()
                if not any(marker in parent.law_name for marker in _LOCAL_MARKERS)
            }
        )
    )


def law_scope(region: str, parents: dict[str, ParentChunk]) -> tuple[str, ...]:
    if not region or region == REGION_UNKNOWN:
        return ()
    if region in {p.law_name for p in parents.values()}:
        return ()
    local = tuple(sorted({p.law_id for p in parents.values() if region in p.law_name}))
    if not local:
        return ()
    return tuple(sorted(set(national_law_ids(parents)) | set(local)))


def _parse_region(text: str) -> str:
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return str(data.get("region") or "").strip() or REGION_UNKNOWN
    return REGION_UNKNOWN


def make_region_node(
    llm: LLM, parents: dict[str, ParentChunk], laws: list[str]
):

    prompt_head = REGION_SYSTEM_PROMPT % {
        "law_count": len(laws),
        "laws": "\n".join(f"- {name}" for name in laws),
    }

    def region_node(state: AgentState) -> dict:
        if not llm.available:
            return {"region": REGION_UNKNOWN, "region_scope": ()}

        prompt: list[dict] = [{"role": "system", "content": prompt_head}]
        prompt.extend({"role": r, "content": c} for r, c in state.get("history") or ())
        prompt.append({"role": "user", "content": state["question"]})

        reply, _usage = llm.chat(prompt, temperature=0.0, name="llm.region")
        region = _parse_region(reply.get("content") or "")
        return {"region": region, "region_scope": law_scope(region, parents)}

    return region_node
