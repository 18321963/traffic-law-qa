from __future__ import annotations

import json
import re
from typing import Sequence

from .. import config
from ..contracts import MaterialPassage

__all__ = ["load_materials", "search_materials"]

MAX_PASSAGE_CHARS = 600

_SPACE = re.compile(r"\s+")

NO_MATERIALS_NOTE = "本次会话没有上传材料，材料检索用不上 —— 法条问题请用 search_law。"
EMPTY_NOTE = (
    "材料里没有任何一段与「{query}」有共同字词。共有 {total} 段 / {docs} 份材料，"
    "头一段是：{head}… 换用材料里真出现过的说法再试。"
)


def _norm(text: str) -> str:
    return _SPACE.sub("", text).lower()


def _grams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)} if len(text) > 1 else {text}


def load_materials(doc_ids: Sequence[str]) -> tuple[MaterialPassage, ...]:
    out: list[MaterialPassage] = []
    for doc_id in doc_ids:
        directory = config.upload_dir(doc_id)
        meta_path = directory / "meta.json"
        chunks_path = directory / "chunks.jsonl"
        if not (meta_path.exists() and chunks_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        display_name = str(meta.get("display_name") or doc_id)
        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                out.append(
                    MaterialPassage(
                        label=f"[材料{len(out) + 1}]",
                        doc_id=doc_id,
                        display_name=display_name,
                        index=int(row.get("index") or 0),
                        text=text[:MAX_PASSAGE_CHARS],
                    )
                )
    return tuple(out)


_LABEL_NO = re.compile(r"\d+")


def _label_no(label: str) -> str:
    match = _LABEL_NO.search(label)
    return match.group() if match else "0"


def search_materials(
    query: str, passages: Sequence[MaterialPassage], *, top_k: int = 5
) -> tuple[list[MaterialPassage], str]:
    if not passages:
        return [], NO_MATERIALS_NOTE

    grams = _grams(_norm(query))
    scored: list[tuple[float, int, MaterialPassage]] = []
    for position, passage in enumerate(passages):
        overlap = len(grams & _grams(_norm(passage.text)))
        if overlap:
            scored.append((overlap / max(1, len(grams)), position, passage))

    if not scored:
        head = passages[0]
        return [], f"材料#0「{query}」｜命中 0 段\n" + EMPTY_NOTE.format(
            query=query,
            total=len(passages),
            docs=len({p.doc_id for p in passages}),
            head=head.text[:60],
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    hits = [
        MaterialPassage(
            label=passage.label,
            doc_id=passage.doc_id,
            display_name=passage.display_name,
            index=passage.index,
            text=passage.text,
            score=round(score, 6),
        )
        for score, _, passage in scored[: max(1, top_k)]
    ]
    lines = [
        f"材料#{_label_no(hits[0].label)}「{query}」｜命中 {len(hits)} 段"
        f"（共 {len(passages)} 段，{len({p.doc_id for p in passages})} 份材料）"
    ]
    lines += [hit.render() for hit in hits]
    lines.append("（会话材料，未入知识库 —— 只能标 [材料N]，标成 [依据N] 会让整篇答案作废）")
    return hits, "\n".join(lines)
