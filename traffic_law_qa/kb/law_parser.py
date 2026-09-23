from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import config
from ..contracts import Article, Heading, LawDocument, Paragraph
from .docx_reader import DocxReader
from .text_reader import TextReader

READER_BY_SUFFIX: dict[str, Any] = {".docx": DocxReader, ".md": TextReader, ".txt": TextReader}


def reader_for(path: str | Path) -> Any:
    try:
        return READER_BY_SUFFIX[Path(path).suffix.lower()]()
    except KeyError:
        raise ValueError(
            f"{Path(path).name} 的后缀不在可读范围内（{'、'.join(sorted(READER_BY_SUFFIX))}）"
        ) from None

CN_NUM = "零一二三四五六七八九十百千"
CN_CLASS = f"[{CN_NUM}]"
RE_TOC_HEAD = re.compile(r"^目\s*录$")
RE_CHAPTER = re.compile(rf"^第({CN_CLASS})章[\s　]*(.*)$")
RE_SECTION = re.compile(rf"^第({CN_CLASS})节[\s　]*(.*)$")
RE_ARTICLE = re.compile(rf"^第({CN_CLASS}+)条(?:[\s　]*(.*))?$", re.S)
RE_ARTICLE_REF = re.compile(rf"第{CN_CLASS}+条")
RE_FILENAME_VERSION = re.compile(r"_(\d{4})(\d{2})(\d{2})$")

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MID_ASCII_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])")

CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}

LAW_ID_REGISTRY: dict[str, str] = {
    "中华人民共和国道路交通安全法": "road_traffic_safety_law",
    "中华人民共和国道路交通安全法实施条例": "road_traffic_safety_regulation",
    "深圳经济特区智能网联汽车管理条例": "sz_icv_regulation",
    "深圳经济特区道路交通安全违法行为处罚条例": "sz_traffic_penalty_regulation",
    "中华人民共和国道路运输条例": "road_transport_regulation",
    "机动车交通事故责任强制保险条例": "traffic_insurance_regulation",
}


def clean_text(text: str) -> str:
    return _MID_ASCII_SPACE.sub("", _ZERO_WIDTH.sub("", text)).strip()


def cn_to_int(cn: str) -> int:
    if not cn:
        raise ValueError("空的中文数字")
    section = number = 0
    for ch in cn:
        if ch in CN_DIGITS:
            number = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            unit = CN_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        else:
            raise ValueError(f"无法解析的中文数字：{cn}")
    total = section + number
    if total == 0 and cn != "零":
        raise ValueError(f"无法解析的中文数字：{cn}")
    return total


def _norm_title(raw: str) -> str:
    return re.sub(r"\s+", "", raw)


def sha1_of(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class LawParser:

    layer = "parse"
    input_desc = "list[Paragraph]"
    output_desc = "LawDocument"

    def __init__(self, law_ids: dict[str, str] | None = None) -> None:
        self.law_ids = dict(LAW_ID_REGISTRY if law_ids is None else law_ids)

    def parse(
        self,
        paragraphs: list[Paragraph],
        *,
        source_file: str,
        law_name: str | None = None,
        version: str | None = None,
    ) -> LawDocument:
        paras = [Paragraph(p.index, clean_text(p.text), p.style) for p in paragraphs]
        paras = [p for p in paras if p.text]
        if not paras:
            raise ValueError(f"{source_file} 没有可解析的段落")

        body_start, toc_lines = self._locate_body(paras)
        preamble = [p.text for p in paras[:body_start]]
        law_name = law_name or self._guess_law_name(preamble, source_file)
        version = version or self._guess_version(source_file)
        law_id = self.law_ids.get(law_name) or f"law_{hashlib.sha1(law_name.encode()).hexdigest()[:10]}"

        chapters: list[Heading] = []
        sections: list[Heading] = []
        articles: list[Article] = []
        leftovers: list[str] = []

        current_chapter: str | None = None
        current_section: str | None = None
        pending_no: str | None = None
        pending_index = 0
        buffer: list[str] = []

        def flush() -> None:
            nonlocal pending_no, buffer
            if pending_no is None or not buffer:
                buffer, pending_no = [], None
                return
            text = "\n".join(buffer)
            refs = tuple(
                dict.fromkeys(
                    m.group(0) for m in RE_ARTICLE_REF.finditer(text) if m.group(0) != pending_no
                )
            )
            articles.append(
                Article(
                    article_no=pending_no,
                    article_index=pending_index,
                    chapter=current_chapter,
                    section=current_section,
                    text=text,
                    paragraphs=tuple(buffer),
                    refs=refs,
                )
            )
            buffer, pending_no = [], None

        for para in paras[body_start:]:
            text = para.text

            chapter_match = RE_CHAPTER.match(text)
            if chapter_match:
                flush()
                chapter_no = f"第{chapter_match.group(1)}章"
                title = _norm_title(chapter_match.group(2))
                current_chapter = re.sub(r"\s+", " ", f"{chapter_no} {title}").strip()
                current_section = None
                chapters.append(Heading(no=chapter_no, title=title))
                continue

            section_match = RE_SECTION.match(text)
            if section_match:
                flush()
                section_no = f"第{section_match.group(1)}节"
                title = _norm_title(section_match.group(2))
                current_section = re.sub(r"\s+", " ", f"{section_no} {title}").strip()
                sections.append(Heading(no=section_no, title=title))
                continue

            article_match = RE_ARTICLE.match(text)
            if article_match:
                flush()
                pending_no = f"第{article_match.group(1)}条"
                pending_index = len(articles) + 1
                buffer = [text]
                continue

            if pending_no is not None:
                buffer.append(text)
            else:
                leftovers.append(text)

        flush()

        return LawDocument(
            law_id=law_id,
            law_name=law_name,
            version=version,
            source_file=source_file,
            citation=f"《{law_name}》({version})",
            articles=tuple(articles),
            chapters=tuple(chapters),
            sections=tuple(sections),
            toc=tuple(toc_lines),
            preamble=tuple(preamble),
            leftovers=tuple(leftovers),
        )

    def parse_file(self, path: str | Path, reader: Any | None = None) -> LawDocument:
        path = Path(path)
        paragraphs = (reader or reader_for(path)).read(path)
        return self.parse(paragraphs, source_file=path.name)

    @staticmethod
    def _locate_body(paras: list[Paragraph]) -> tuple[int, list[str]]:
        toc_start = next((i for i, p in enumerate(paras) if RE_TOC_HEAD.match(p.text)), None)

        if toc_start is None:
            first_article = next((i for i, p in enumerate(paras) if RE_ARTICLE.match(p.text)), 0)
            for i in range(first_article, -1, -1):
                if RE_CHAPTER.match(paras[i].text):
                    return i, []
            return first_article, []

        toc_lines: list[str] = []
        cursor = toc_start + 1
        while cursor < len(paras) and (
            RE_CHAPTER.match(paras[cursor].text) or RE_SECTION.match(paras[cursor].text)
        ):
            toc_lines.append(paras[cursor].text)
            cursor += 1

        head = cursor
        while head < len(paras) and not RE_ARTICLE.match(paras[head].text):
            head += 1

        for i in range(head, toc_start, -1):
            if RE_CHAPTER.match(paras[i].text):
                return i, toc_lines
        return head, toc_lines

    def _guess_law_name(self, preamble: list[str], source_file: str) -> str:
        stem = RE_FILENAME_VERSION.sub("", Path(source_file).stem).strip()
        if stem:
            return stem
        for line in preamble:
            if line and not line.startswith("（") and not RE_TOC_HEAD.match(line):
                return line
        return Path(source_file).stem

    @staticmethod
    def _guess_version(source_file: str) -> str:
        match = RE_FILENAME_VERSION.search(Path(source_file).stem)
        if not match:
            return "unknown"
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def render_markdown(law: LawDocument) -> str:
    lines = [f"# {law.law_name}", "", f"版本：{law.version} ｜ 来源：{law.source_file}", ""]
    if law.preamble:
        lines += [f"> {line}" for line in law.preamble]
        lines.append("")

    emitted: set[str] = set()
    for article in law.articles:
        for level, heading in ((2, article.chapter), (3, article.section)):
            if heading and heading not in emitted:
                emitted.add(heading)
                lines += ["", f"{'#' * level} {heading}", ""]
        lines += [f"#### {article.article_no}", "", article.text, ""]
    return "\n".join(lines).rstrip() + "\n"


class LawLibrary:

    input_desc = "LawDocument | law_id"
    output_desc = "parsed/*.json + text/*.md + manifest.json"

    def __init__(self, parsed_dir: Path | None = None, text_dir: Path | None = None) -> None:
        self.parsed_dir = Path(parsed_dir or config.PARSED_DIR)
        self.text_dir = Path(text_dir or config.TEXT_DIR)
        self.manifest_path = self.parsed_dir / "manifest.json"

    def path_of(self, law_id: str) -> Path:
        return self.parsed_dir / f"{law_id}.json"

    def save(self, law: LawDocument, *, write_markdown: bool = True) -> Path:
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_of(law.law_id)
        path.write_text(json.dumps(law.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if write_markdown:
            self.text_dir.mkdir(parents=True, exist_ok=True)
            (self.text_dir / f"{law.law_id}.md").write_text(render_markdown(law), encoding="utf-8")
        return path

    def load(self, law_id: str) -> LawDocument:
        path = self.path_of(law_id)
        if not path.exists():
            raise FileNotFoundError(f"未找到结构层产物：{path}（请先运行 parse 阶段）")
        return LawDocument.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_all(self) -> list[LawDocument]:
        ids = [item["law_id"] for item in self.manifest().get("laws", [])]
        if not ids:
            ids = sorted(p.stem for p in self.parsed_dir.glob("*.json") if p.name != "manifest.json")
        return [self.load(law_id) for law_id in ids]

    def manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"generated_at": None, "cross_check": False, "laws": []}

    def manifest_by_file(self) -> dict[str, dict]:
        return {item["file"]: item for item in self.manifest().get("laws", [])}

    def save_manifest(self, entries: list[dict]) -> dict:
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_dir": str(config.SOURCE_DIR),
            "cross_check": False,
            "laws": entries,
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def entry_of(law: LawDocument, source_path: Path, digest: str) -> dict:
        return {
            "law_id": law.law_id,
            "law_name": law.law_name,
            "version": law.version,
            "file": source_path.name,
            "sha1": digest,
            "articles": law.article_count,
            "chars": law.char_count,
            "chapters": len(law.chapters),
            "sections": len(law.sections),
            "leftovers": len(law.leftovers),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


class ParseStage:

    layer = "parse"
    input_desc = "docx 目录 (Path)"
    output_desc = "list[LawDocument] + parsed/*.json + text/*.md"

    def __init__(
        self,
        *,
        source_dir: Path | None = None,
        readers: dict[str, Any] | None = None,
        parser: LawParser | None = None,
        library: LawLibrary | None = None,
        verbose: bool = True,
    ) -> None:
        self.source_dir = Path(source_dir or config.SOURCE_DIR)
        self.readers = readers or {suffix: READER_BY_SUFFIX[suffix]() for suffix in config.SOURCE_SUFFIXES}
        self.parser = parser or LawParser()
        self.library = library or LawLibrary()
        self.verbose = verbose
        self.last_skipped: list[str] = []

    def run(self, *, force: bool = False, only: str | None = None) -> list[LawDocument]:
        cached_entries = self.library.manifest_by_file()
        laws: list[LawDocument] = []
        entries: list[dict] = []
        self.last_skipped = []

        for docx_path in config.source_files(self.source_dir):
            digest = sha1_of(docx_path)
            cached = cached_entries.get(docx_path.name)

            if cached and only and cached["law_id"] != only:
                entries.append(cached)
                continue

            if not force and cached and cached.get("sha1") == digest:
                self.last_skipped.append(docx_path.name)
                if self.verbose:
                    print(f"[parse] skip {docx_path.name}（sha1 未变，{cached['articles']} 条）")
                entries.append(cached)
                laws.append(self.library.load(cached["law_id"]))
                continue

            law = self.parser.parse_file(docx_path, self.readers[docx_path.suffix.lower()])
            if only and law.law_id != only:
                continue

            self.library.save(law)
            entries.append(LawLibrary.entry_of(law, docx_path, digest))
            laws.append(law)

            if self.verbose:
                flag = "!" if law.leftovers else "v"
                print(
                    f"[parse] {flag} {law.law_id:<30} {law.article_count:>4} 条 | "
                    f"{len(law.chapters)} 章 {len(law.sections)} 节 | {law.char_count} 字 | "
                    f"残留 {len(law.leftovers)}"
                )
                for item in law.leftovers[:3]:
                    print(f"         残留段落：{item[:60]}")

        self.library.save_manifest(entries)
        if self.verbose:
            total = sum(item["articles"] for item in entries)
            print(f"[parse] 完成：{len(entries)} 部法规 / {total} 条 → {self.library.parsed_dir}")
        return laws


if __name__ == "__main__":
    raise SystemExit("已收口：请用 python -m traffic_law_qa.pipeline parse（清单见 README「入口」）")
