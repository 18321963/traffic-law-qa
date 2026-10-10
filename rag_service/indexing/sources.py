from __future__ import annotations

import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from rag_contracts.domain.disk import Paragraph

__all__ = ["DocxReader", "PdfReader", "TextReader", "READER_BY_SUFFIX", "reader_for"]


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


RE_PDF_TS = re.compile(r"^\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}")
RE_PDF_NOISE = ("中国政府网", "gov.cn", "网站标识码")
RE_PDF_PAGE = re.compile(r"^\d{1,3}\s*/\s*\d{1,3}$")
RE_PDF_NUMBER = re.compile(r"^\d+$")
RE_PDF_SLASH = re.compile(r"^/$")
RE_PDF_RULE = re.compile(r"^_+$")
RE_PDF_URL = re.compile(r"^https?://")
RE_PDF_PUNCT_SPACE = re.compile(r"(?<=[　-〿＀-￯])[ \t]+(?=[一-鿿“”‘’])")
RE_PDF_PUNCT = re.compile(r"[，。；、：！？（）《》〈〉“”‘’,.!?;:()\[\]{}<>]")
RE_PDF_BLOCK = re.compile(
    r"^第[一-鿿]+[章节](?=\s|$)"
    r"|^第[一-鿿]+条(?=\s|$)"
    r"|^附\s?则$"
)


def _pdf_noise(line: str) -> bool:
    if RE_PDF_TS.match(line) or any(key in line for key in RE_PDF_NOISE):
        return True
    return bool(
        RE_PDF_PAGE.match(line)
        or RE_PDF_NUMBER.match(line)
        or RE_PDF_SLASH.match(line)
        or RE_PDF_RULE.match(line)
        or RE_PDF_URL.match(line)
    )


def _denoise_lines(raw_lines: list[str]) -> list[str]:
    kept: list[str] = []
    for raw in raw_lines:
        line = RE_PDF_PUNCT_SPACE.sub("", raw.strip())
        if line and not _pdf_noise(line):
            kept.append(line)
    return kept


def _drop_frequent_lines(lines: list[str]) -> list[str]:
    counts = Counter(lines)
    return [
        line for line in lines if counts[line] < 3 or len(line) > 30 or RE_PDF_PUNCT.search(line)
    ]


def _to_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    for line in lines:
        if not blocks or RE_PDF_BLOCK.match(line):
            blocks.append(line)
        else:
            blocks[-1] += line
    return blocks


def _load_pypdf() -> Any:
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError('未安装 pypdf：读 PDF 要 pip install -e ".[pdf]"') from exc
    return pypdf


class PdfReader:

    layer = "read"
    input_desc = "pdf 路径 (Path)"
    output_desc = "list[Paragraph]"

    def read(self, path: str | Path) -> list[Paragraph]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到 PDF 文件：{path}")
        pypdf = _load_pypdf()
        pages = pypdf.PdfReader(str(path)).pages
        raw_lines = [line for page in pages for line in (page.extract_text() or "").splitlines()]
        blocks = _to_blocks(_drop_frequent_lines(_denoise_lines(raw_lines)))
        return [
            Paragraph(index=index, text=block, style=None) for index, block in enumerate(blocks)
        ]


READER_BY_SUFFIX: dict[str, Any] = {
    ".docx": DocxReader,
    ".md": TextReader,
    ".pdf": PdfReader,
    ".txt": TextReader,
}


def reader_for(path: str | Path) -> Any:
    try:
        return READER_BY_SUFFIX[Path(path).suffix.lower()]()
    except KeyError:
        raise ValueError(
            f"{Path(path).name} 的后缀不在可读范围内（{'、'.join(sorted(READER_BY_SUFFIX))}）"
        ) from None

if __name__ == "__main__":
    raise SystemExit("已收口：请用 python -m rag_service.cli.build docx（清单见 README「入口」）")
