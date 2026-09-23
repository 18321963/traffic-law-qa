from __future__ import annotations

import re
from pathlib import Path

from ..contracts import Paragraph

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
