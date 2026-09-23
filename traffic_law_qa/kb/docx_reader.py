from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from ..contracts import Paragraph

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


if __name__ == "__main__":
    raise SystemExit("已收口：请用 python -m traffic_law_qa.pipeline docx（清单见 README「入口」）")
