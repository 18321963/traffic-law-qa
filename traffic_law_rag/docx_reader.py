"""Layer 1 · read：docx 文件 → 段落列表。

    Input : Path（.docx 文件）
    Output: list[Paragraph]

为什么直读 XML：本项目语料是原生 DOCX、纯段落、无扫描无表格，
"第X条必须独占一段"是整条管道的根基，版面重排型解析器（docling 等）反而会破坏这个边界。
解析用标准库 `xml.etree.ElementTree`，不引入 lxml/python-docx 等额外依赖。
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .contracts import Paragraph

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOCUMENT_XML = "word/document.xml"


class DocxReader:
    """DOCX 读取器：只负责把 docx 变成段落，不做任何业务清洗。"""

    layer = "read"
    input_desc = "docx 路径 (Path)"
    output_desc = "list[Paragraph]"

    def __init__(self, *, keep_empty: bool = False) -> None:
        """
        Args:
            keep_empty: 是否保留空段落。默认丢弃，避免解析阶段产生噪声。
        """
        self.keep_empty = keep_empty

    # -------------------------------------------------------------- 主接口
    def read(self, path: str | Path) -> list[Paragraph]:
        """读取 docx 的全部段落，按文档顺序返回。"""
        path = Path(path)
        root = ET.fromstring(self._read_document_xml(path))
        parent_map = {child: parent for parent in root.iter() for child in parent}

        paragraphs: list[Paragraph] = []
        for node in root.iter(f"{{{W}}}p"):
            if _has_paragraph_ancestor(node, parent_map):
                continue  # 跳过文本框等嵌套段落，避免同段文字被重复收集
            text = _paragraph_text(node)
            if not text and not self.keep_empty:
                continue
            paragraphs.append(
                Paragraph(index=len(paragraphs), text=text, style=_paragraph_style(node))
            )
        return paragraphs

    def read_text(self, path: str | Path, *, sep: str = "\n") -> str:
        """把 docx 拍平成一整段纯文本（人工比对原文用）。"""
        return sep.join(p.text for p in self.read(path))

    def read_many(self, paths: list[str | Path]) -> dict[str, list[Paragraph]]:
        """批量读取，返回 {文件名: 段落列表}。"""
        return {Path(p).name: self.read(p) for p in paths}

    # -------------------------------------------------------------- 内部
    @staticmethod
    def _read_document_xml(path: Path) -> bytes:
        if not path.exists():
            raise FileNotFoundError(f"找不到 docx 文件：{path}")
        with zipfile.ZipFile(path) as zf:
            if DOCUMENT_XML not in zf.namelist():
                raise ValueError(f"{path.name} 不是有效的 docx：缺少 {DOCUMENT_XML}")
            return zf.read(DOCUMENT_XML)


# ------------------------------------------------------------------ XML 工具
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
    """按文档顺序拼出段落文本，w:tab / w:br 还原为制表符与换行。"""
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


# ------------------------------------------------------------------ 调试入口
def main(argv: list[str] | None = None) -> int:
    """python -m traffic_law_rag.docx_reader [docx 路径 ...]"""
    from . import config

    args = list(sys.argv[1:] if argv is None else argv)
    targets = [Path(a) for a in args] if args else sorted(config.DOCX_DIR.glob("*.docx"))
    reader = DocxReader()

    for target in targets:
        paragraphs = reader.read(target)
        chars = sum(len(p.text) for p in paragraphs)
        print(f"\n=== {target.name} | 段落 {len(paragraphs)} | 字符 {chars}")
        for para in paragraphs[:12]:
            print(f"  [{para.index:>3}] style={para.style!s:<10} {para.text[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
