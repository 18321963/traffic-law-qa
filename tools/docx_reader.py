#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取 .docx 法规文件，输出纯文本或结构化 JSON。

依赖：python-docx  （pip install python-docx）

用法::

    # 1) 快速预览前 30 个非空段落
    python tools/docx_reader.py 中华人民共和国道路交通安全法_20210429.docx

    # 2) 导出纯文本（清理掉中文词中间的异常空格）
    python tools/docx_reader.py xxx.docx --text out.txt

    # 3) 导出结构化 JSON（章 / 节 / 条）
    python tools/docx_reader.py xxx.docx --json out.json

    # 4) 批量处理当前目录所有 docx
    python tools/docx_reader.py --all --json-dir 法规知识库/parsed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import docx

# 汉字 + 常用中文标点 + 全角字符（用于识别"词中被塞了空格"的情况）
CJK = r"\u3400-\u4dbf\u4e00-\u9fff\u3001-\u303f\uff00-\uffef"
# 半角空格/制表符夹在"汉字或数字"之间 → 判为转换残留，删掉
_SPURIOUS_SPACE = re.compile(rf"(?<=[{CJK}0-9])[ \t]+(?=[{CJK}0-9])")

_NUM = "一二三四五六七八九十百零〇两"
RE_CHAPTER = re.compile(rf"^第[{_NUM}]+章")
RE_SECTION = re.compile(rf"^第[{_NUM}]+节")
RE_ARTICLE = re.compile(rf"^第[{_NUM}]+条")
RE_TOC = re.compile(r"^目\s*录$")


def normalize(text: str) -> str:
    """清理 docx 中常见的转换残留空格。

    只删除半角空格：法规正文里用于排版分隔的是全角空格 U+3000
    （如 ``第一章\\u3000总\\u3000\\u3000则``），必须保留。
    """
    text = text.replace("\xa0", " ").replace("\u200b", "")
    return _SPURIOUS_SPACE.sub("", text).strip()


def read_paragraphs(path: str | Path) -> list[str]:
    """返回文档中所有非空段落的文本（已 normalize）。"""
    doc = docx.Document(str(path))
    out = []
    for p in doc.paragraphs:
        t = normalize(p.text)
        if t:
            out.append(t)
    return out


def read_tables(path: str | Path) -> list[list[list[str]]]:
    """返回文档中所有表格（三维列表：表 -> 行 -> 单元格）。"""
    doc = docx.Document(str(path))
    return [
        [[normalize(c.text) for c in row.cells] for row in table.rows]
        for table in doc.tables
    ]


# --------------------------------------------------------------------------- #
# 结构化解析
# --------------------------------------------------------------------------- #
def split_structure(paras: list[str]) -> dict:
    """把段落切成 章 / 节 / 条 三层结构。

    难点是文档开头有一份「目录」，条目与正文标题长得一模一样。
    做法：正文第一条必然是 ``第X条``，而它前面最后一个"章/节"标题
    就是正文的真正起点。
    """
    first_article = next(
        (i for i, t in enumerate(paras) if RE_ARTICLE.match(t)), None
    )

    # 正文起点：第一条之前最后一个章/节标题
    body_start = 0
    if first_article is not None:
        for i in range(first_article - 1, -1, -1):
            if RE_CHAPTER.match(paras[i]) or RE_SECTION.match(paras[i]):
                body_start = i
                break

    toc_start = next((i for i, t in enumerate(paras) if RE_TOC.match(t)), None)
    toc = paras[toc_start:body_start] if toc_start is not None and toc_start < body_start else []

    chapters: list[dict] = []
    ungrouped: list[dict] = []
    chapter: dict | None = None
    section: dict | None = None
    current: dict | None = None  # 当前正在累积的"条"

    def close_article():
        nonlocal current
        if current is not None:
            current["text"] = "\n".join(current.pop("_lines"))
        current = None

    for line in paras[body_start:]:
        if RE_CHAPTER.match(line):
            close_article()
            chapter = {"type": "chapter", "title": line, "sections": []}
            chapters.append(chapter)
            section = None
        elif RE_SECTION.match(line):
            close_article()
            section = {"type": "section", "title": line, "articles": []}
            if chapter is None:
                chapter = {"type": "chapter", "title": None, "sections": []}
                chapters.append(chapter)
            chapter["sections"].append(section)
        elif RE_ARTICLE.match(line):
            close_article()
            current = {"type": "article", "title": line, "_lines": [line]}
            if section is not None:
                section["articles"].append(current)
            elif chapter is not None:
                chapter.setdefault("articles", []).append(current)
            else:
                ungrouped.append(current)
        else:
            # 续行：并入当前"条"；没有当前条则视为章前说明
            if current is not None:
                current["_lines"].append(line)
            elif chapters or section:
                target = section if section is not None else chapter
                target.setdefault("preamble", []).append(line)
            else:
                ungrouped.append({"type": "text", "text": line})

    close_article()

    # 章下没有必要分节时，把 articles 提到章一级
    for ch in chapters:
        flat = []
        for sec in ch.get("sections", []):
            flat.extend(sec["articles"])
        if flat:
            ch["articles"] = flat

    return {
        "title": paras[0] if paras else "",
        "preamble": paras[1] if len(paras) > 1 and not RE_CHAPTER.match(paras[1]) else "",
        "toc": toc,
        "chapters": chapters,
        "ungrouped": ungrouped,
        "stats": {
            "paragraphs": len(paras),
            "chapters": len(chapters),
            "articles": sum(len(ch.get("articles", [])) for ch in chapters)
            + len(ungrouped),
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _preview(paras: list[str], n: int) -> None:
    print(f"（共 {len(paras)} 个非空段落，显示前 {min(n, len(paras))} 个）\n")
    for i, t in enumerate(paras[:n]):
        print(f"[{i:>4}] {t}")


def _resolve_dst(raw: str, path: Path, batch: bool, suffix: str) -> Path:
    """把 --text/--json 的取值解析成最终文件路径。

    规则：值为已存在的目录、以分隔符结尾、或（批量模式下的）无扩展名路径
    → 视为目录，输出 <目录>/<源文件名>.<suffix>；否则视为具体文件。
    """
    dst = Path(raw)
    is_dir = (
        dst.is_dir()
        or raw.endswith(("/", "\\"))
        or (batch and not dst.suffix)
    )
    return dst / (path.stem + suffix) if is_dir else dst


def _process(path: Path, args) -> None:
    paras = read_paragraphs(path)
    batch = args.all or len(args.files) > 1
    print(f"== {path.name} ==")
    print(f"   段落数: {len(paras)}  表格数: {len(docx.Document(str(path)).tables)}")

    if args.text:
        dst = _resolve_dst(args.text, path, batch, ".txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(paras), encoding="utf-8")
        print(f"   -> 文本已写入 {dst}")

    if args.json:
        data = split_structure(paras)
        dst = _resolve_dst(args.json, path, batch, ".json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        s = data["stats"]
        print(
            f"   -> JSON 已写入 {dst}"
            f"  (章 {s['chapters']} / 条 {s['articles']})"
        )

    if args.preview:
        _preview(paras, args.preview)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="读取 .docx 法规文件（基于 python-docx）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("files", nargs="*", help="一个或多个 .docx 文件")
    ap.add_argument(
        "--all",
        action="store_true",
        help="处理当前目录下所有 .docx（同时指定 --text/--json 时视为输出目录）",
    )
    ap.add_argument("--text", metavar="PATH", nargs="?", const="out.txt",
                    help="导出纯文本；批量时为输出目录")
    ap.add_argument("--json", metavar="PATH", nargs="?", const="out.json",
                    help="导出结构化 JSON；批量时为输出目录")
    ap.add_argument("--preview", type=int, nargs="?", const=30, default=None,
                    metavar="N", help="预览前 N 个段落（默认 30）")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.all:
        files = sorted(Path.cwd().glob("*.docx"))
        if not files:
            print("当前目录下没有 .docx 文件", file=sys.stderr)
            return 1
    elif args.files:
        files = [Path(f) for f in args.files]
    else:
        build_parser().print_help()
        return 1

    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"找不到文件: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    for f in files:
        _process(f, args)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
