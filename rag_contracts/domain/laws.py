from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from .disk import ParentChunk

__all__ = ["LawInfo", "laws_of", "resolve_law"]

_LOCAL_MARKERS = ("省", "市", "自治区", "经济特区")
"""法规名里带这些词 = 地方性法规。

判据放在**法规名**上而不是写死一份 law_id 清单：新增一部地方条例时，没人会记得来这里改。
代价是名字里含「市」却不针对某地的法规（《城市道路管理条例》那种）会被误判成地方性 ——
库内 8 部没有这种；真出现了再收紧成「『省』『市』前面还得有地名」。
"""


def _is_local(law_name: str) -> bool:
    return any(marker in law_name for marker in _LOCAL_MARKERS)


@dataclass(frozen=True)
class LawInfo:

    law_id: str
    law_name: str
    version: str
    articles: int
    local: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LawInfo":
        return cls(
            law_id=str(data.get("law_id") or ""),
            law_name=str(data.get("law_name") or ""),
            version=str(data.get("version") or ""),
            articles=int(data.get("articles") or 0),
            local=bool(data.get("local")),
        )


def laws_of(parents: Iterable[ParentChunk]) -> tuple[LawInfo, ...]:
    label: dict[str, tuple[str, str]] = {}
    numbers: dict[str, set[str]] = {}
    for parent in parents:
        label[parent.law_id] = (parent.law_name, parent.version)
        numbers.setdefault(parent.law_id, set()).add(parent.article_no)
    return tuple(
        LawInfo(law_id, name, version, len(numbers[law_id]), local=_is_local(name))
        for law_id, (name, version) in sorted(label.items(), key=lambda item: item[1][0])
    )


def resolve_law(name: str, laws: Sequence[LawInfo]) -> tuple[str | None, list[str]]:
    table = {law.law_name: law.law_id for law in laws}
    known = sorted(table)
    cleaned = (name or "").strip()
    if not cleaned:
        return None, known

    if cleaned in table:
        return table[cleaned], known

    suffixed = [item for item in known if item.endswith(cleaned)]
    if len(suffixed) == 1:
        return table[suffixed[0]], known

    contained = [law for law in known if cleaned in law or law in cleaned]
    if len(contained) == 1:
        return table[contained[0]], known
    return None, known
