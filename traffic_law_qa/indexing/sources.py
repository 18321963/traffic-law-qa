from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..contracts.disk import Paragraph

__all__ = ["DocxReader", "TextReader", "READER_BY_SUFFIX", "reader_for"]


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOCUMENT_XML = "word/document.xml"


class DocxReader:

    layer = "read"
    input_desc = "docx 路径 (Path)"
    output_desc = "list[Paragraph]"

    def __init__(self, *, keep_empty: bool = False) -> None:
        self.keep_empty = keep_empty

    def read(self, path: str | Path) -> list[Paragraph]:
        path = Path(path)
        root = ET.fromstring(self._read_document_xml(path))
        parent_map = {child: parent for parent in root.iter() for child in parent}

        paragraphs: list[Paragraph] = []
        for node in root.iter(f"{{{W}}}p"):
            if _has_paragraph_ancestor(node, parent_map):
                continue
            text = _paragraph_text(node)
            if not text and not self.keep_empty:
                continue
            paragraphs.append(
                Paragraph(index=len(paragraphs), text=text, style=_paragraph_style(node))
            )
        return paragraphs

    def read_text(self, path: str | Path, *, sep: str = "\n") -> str:
        return sep.join(p.text for p in self.read(path))

    def read_many(self, paths: list[str | Path]) -> dict[str, list[Paragraph]]:
        return {Path(p).name: self.read(p) for p in paths}

    @staticmethod
    def _read_document_xml(path: Path) -> bytes:
        if not path.exists():
            raise FileNotFoundError(f"找不到 docx 文件：{path}")
        with zipfile.ZipFile(path) as zf:
            if DOCUMENT_XML not in zf.namelist():
                raise ValueError(f"{path.name} 不是有效的 docx：缺少 {DOCUMENT_XML}")
            return zf.read(DOCUMENT_XML)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _has_paragraph_ancestor(node: ET.Element, parent_map: dict[ET.Element, ET.Element]) -> bool:
    parent = parent_map.get(node)
    while parent is not None:
        if _local(parent.tag) == "p":
            return True
        parent = parent_map.get(parent)
    return False


def _paragraph_text(p: ET.Element) -> str:
    parts: list[str] = []
    for node in p.iter():
        tag = _local(node.tag)
        if tag == "t" and node.text:
            parts.append(node.text)
        elif tag == "tab":
            parts.append("\t")
        elif tag in ("br", "cr"):
            parts.append("\n")
    return "".join(parts).strip()


def _paragraph_style(p: ET.Element) -> str | None:
    ppr = p.find(f"{{{W}}}pPr")
    if ppr is None:
        return None
    style = ppr.find(f"{{{W}}}pStyle")
    if style is None:
        return None
    return style.get(f"{{{W}}}val")


RE_HEADING = re.compile(r"^#{1,6}[ \t]*")
RE_BLANK = re.compile(r"\n\s*\n")


def _dedupe_headings(blocks: list[str]) -> list[str]:
    """剥掉 `#`，并丢掉与正文重复的那一行标题。

    `LawLibrary.save` 写出的 `text/*.md` 是这个形状：`#### 第九十条` 之后另起一段
    `第九十条　正文…` —— **条号写了两遍**。`LawParser` 的 `RE_ARTICLE` 是 `^第…条` 锚定的，
    两行都命中：留着标题行会把条号与正文错开并让条数翻倍（实测 82 → 164）；
    一个都不剥则标题行不命中 `RE_CHAPTER`/`RE_ARTICLE`，被 `buffer` 追加到**上一条**的正文里
    （实测每条正文尾部都挂着下一行的 `#### 第X条`）。所以：剥 `#`，再按「下一段以它开头」去重。

    用户自己写的 md 若正文不重复条号（`#### 第九十条` 后直接跟正文），标题行照常保留、照常成条。
    """
    out: list[str] = []
    for index, block in enumerate(blocks):
        body = RE_HEADING.sub("", block).strip()
        if body != block and index + 1 < len(blocks):
            nxt = RE_HEADING.sub("", blocks[index + 1]).strip()
            if nxt.startswith(body):
                continue
        if body:
            out.append(body)
    return out


class TextReader:

    layer = "read"
    input_desc = "md/txt 路径 (Path)"
    output_desc = "list[Paragraph]"

    def __init__(self, *, encoding: str = "utf-8-sig") -> None:
        self.encoding = encoding

    def read(self, path: str | Path) -> list[Paragraph]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到文本文件：{path}")
        text = path.read_text(encoding=self.encoding)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        blocks = [block.strip() for block in RE_BLANK.split(text) if block.strip()]
        return [
            Paragraph(index=index, text=block, style=None)
            for index, block in enumerate(_dedupe_headings(blocks))
        ]


READER_BY_SUFFIX: dict[str, Any] = {".docx": DocxReader, ".md": TextReader, ".txt": TextReader}


def reader_for(path: str | Path) -> Any:
    try:
        return READER_BY_SUFFIX[Path(path).suffix.lower()]()
    except KeyError:
        raise ValueError(
            f"{Path(path).name} 的后缀不在可读范围内（{'、'.join(sorted(READER_BY_SUFFIX))}）"
        ) from None

if __name__ == "__main__":
    raise SystemExit("已收口：请用 python -m traffic_law_qa.cli.build docx（清单见 README「入口」）")
