from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from .disk import ParentChunk

__all__ = ["LawInfo", "laws_of", "resolve_law"]


@dataclass(frozen=True)
class LawInfo:

    law_id: str
    law_name: str
    version: str
    articles: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LawInfo":
        return cls(
            law_id=str(data.get("law_id") or ""),
            law_name=str(data.get("law_name") or ""),
            version=str(data.get("version") or ""),
            articles=int(data.get("articles") or 0),
        )


def laws_of(parents: Iterable[ParentChunk]) -> tuple[LawInfo, ...]:
    label: dict[str, tuple[str, str]] = {}
    numbers: dict[str, set[str]] = {}
    for parent in parents:
        label[parent.law_id] = (parent.law_name, parent.version)
        numbers.setdefault(parent.law_id, set()).add(parent.article_no)
    return tuple(
        LawInfo(law_id, name, version, len(numbers[law_id]))
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
