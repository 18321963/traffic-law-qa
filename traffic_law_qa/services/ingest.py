from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..contracts import Paragraph
from ..kb.law_parser import LawParser, reader_for

__all__ = [
    "IngestPlan",
    "canonical_name",
    "dry_run",
    "promote",
    "read_paragraphs",
    "store_material",
    "load_material_meta",
]

MAX_BYTES = 8 * 1024 * 1024
MAX_LEFTOVER_RATIO = 0.2

SUFFIX_NOTE = "只收 {suffixes}；PDF 暂不支持，请先转成 docx 或 md 再传。"
TOO_BIG_NOTE = "文件 {size} 字节，超过 {limit} 字节上限。"
EMPTY_NOTE = "文件里没有可解析的段落（是空的，或者读取后一个字都没有）。"
NO_ARTICLE_NOTE = (
    "没解析出任何条文（第…条）。本库只收「有条号编号的法规文本」——"
    "通篇散文、表格、纯目录式清单都解析不成法条。"
)
LEFTOVER_NOTE = (
    "条文之外的残留段落占 {ratio:.0%}（上限 {limit:.0%}，共 {leftovers} 段 / {paragraphs} 段）。"
    "段落编号格式与库内不一致时会这样，先看解析产物的 leftovers 再决定要不要收。"
)
DUPLICATE_NOTE = (
    "同样内容（sha1 相同）已经以 {name} 入库了，不用重复上传。"
    "换个文件名不等于换一部法规 —— 两份都留会让库内条文翻倍。"
)


@dataclass(frozen=True)
class IngestPlan:

    filename: str
    display_name: str
    suffix: str
    sha1: str
    size: int
    law_id: str
    law_name: str
    version: str
    citation: str
    articles: int
    chapters: int
    sections: int
    leftovers: int
    paragraphs: int

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "display_name": self.display_name,
            "suffix": self.suffix,
            "sha1": self.sha1,
            "bytes": self.size,
            "law_id": self.law_id,
            "law_name": self.law_name,
            "version": self.version,
            "citation": self.citation,
            "articles": self.articles,
            "chapters": self.chapters,
            "sections": self.sections,
            "leftovers": self.leftovers,
            "paragraphs": self.paragraphs,
        }


def canonical_name(law_name: str, version: str, suffix: str) -> str:
    return f"{law_name}_{version.replace('-', '')}{suffix}"


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _write_temp(data: bytes, name: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="tlq_ingest_"))
    path = tmp / name
    path.write_bytes(data)
    return path


def read_paragraphs(data: bytes, display_name: str) -> list[Paragraph]:
    name = Path(display_name).name
    reader = reader_for(name)
    return reader.read(_write_temp(data, name))


def dry_run(
    data: bytes, display_name: str, *, source_dir: Path | None = None
) -> tuple[IngestPlan | None, str]:
    name = Path(display_name).name
    suffix = Path(name).suffix.lower()
    if suffix not in config.SOURCE_SUFFIXES:
        return None, SUFFIX_NOTE.format(suffixes="、".join(config.SOURCE_SUFFIXES))
    if not data:
        return None, EMPTY_NOTE
    if len(data) > MAX_BYTES:
        return None, TOO_BIG_NOTE.format(size=len(data), limit=MAX_BYTES)

    digest = _sha1(data)
    try:
        paragraphs = read_paragraphs(data, name)
    except Exception as exc:  # noqa: BLE001
        return None, f"读不出内容：{exc}"
    if not paragraphs:
        return None, EMPTY_NOTE

    parser = LawParser()
    guessed_version = LawParser._guess_version(name)
    version = _today() if guessed_version == "unknown" else guessed_version

    try:
        first = parser.parse(paragraphs, source_file=name)
        filename = canonical_name(first.law_name, version, suffix)
        law = parser.parse(
            paragraphs, source_file=filename, law_name=first.law_name, version=version
        )
    except ValueError as exc:
        return None, f"解析失败：{exc}"

    if law.article_count < 1:
        return None, NO_ARTICLE_NOTE
    ratio = len(law.leftovers) / len(paragraphs)
    if ratio > MAX_LEFTOVER_RATIO:
        return None, LEFTOVER_NOTE.format(
            ratio=ratio,
            limit=MAX_LEFTOVER_RATIO,
            leftovers=len(law.leftovers),
            paragraphs=len(paragraphs),
        )

    root = Path(source_dir or config.SOURCE_DIR)
    for existing in config.source_files(root):
        if _sha1(existing.read_bytes()) == digest:
            return None, DUPLICATE_NOTE.format(name=existing.name)

    return (
        IngestPlan(
            filename=filename,
            display_name=name,
            suffix=suffix,
            sha1=digest,
            size=len(data),
            law_id=law.law_id,
            law_name=law.law_name,
            version=law.version,
            citation=law.citation,
            articles=law.article_count,
            chapters=len(law.chapters),
            sections=len(law.sections),
            leftovers=len(law.leftovers),
            paragraphs=len(paragraphs),
        ),
        f"可以入库：{law.citation}，{law.article_count} 条（落盘为 {filename}）",
    )


def promote(data: bytes, plan: IngestPlan, *, source_dir: Path | None = None) -> Path:
    root = Path(source_dir or config.SOURCE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    target = root / plan.filename
    target.write_bytes(data)
    return target


def store_material(
    doc_id: str, data: bytes, display_name: str, paragraphs: list[Paragraph]
) -> Path:
    directory = config.upload_dir(doc_id)
    directory.mkdir(parents=True, exist_ok=True)
    name = Path(display_name).name
    (directory / f"source{Path(name).suffix.lower()}").write_bytes(data)
    rows = [{"index": p.index, "text": p.text} for p in paragraphs]
    chunks_path = directory / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (directory / "meta.json").write_text(
        json.dumps(
            {"doc_id": doc_id, "display_name": name, "paragraphs": len(rows)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return chunks_path


def load_material_meta(doc_id: str) -> dict:
    path = config.upload_dir(doc_id) / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
