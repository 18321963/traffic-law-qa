from __future__ import annotations

import re
from pathlib import Path

from .. import config
from ..contracts import Chunk, ChunkSet, LawDocument, ParentChunk
from .law_parser import LawLibrary

RE_ARTICLE_PREFIX = re.compile(r"^第[零一二三四五六七八九十百千]+条[\s　]*")
RE_SENTENCE_END = re.compile(r"[。；]")
RE_LIST_MARKER = re.compile(
    r"^\s*(?:[（(][一二三四五六七八九十百千\d]+[）)]"
    r"|[①-⑳]"
    r"|[一二三四五六七八九十]+[、.]"
    r"|\d+[.．、])"
)


class LawChunker:

    layer = "chunk"
    input_desc = "LawDocument"
    output_desc = "ChunkSet"

    def __init__(
        self,
        *,
        min_part_chars: int = 15,
        max_part_chars: int = 500,
        with_chapter_prefix: bool = True,
    ) -> None:
        self.min_part_chars = min_part_chars
        self.max_part_chars = max_part_chars
        self.with_chapter_prefix = with_chapter_prefix

    def chunk(self, law: LawDocument) -> ChunkSet:
        parents: list[ParentChunk] = []
        chunks: list[Chunk] = []

        for article in law.articles:
            parent_id = self.parent_id_of(law, article.article_index)
            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    law_id=law.law_id,
                    law_name=law.law_name,
                    version=law.version,
                    citation=law.citation,
                    article_no=article.article_no,
                    article_index=article.article_index,
                    chapter=article.chapter,
                    section=article.section,
                    text=article.text,
                    refs=article.refs,
                )
            )

            parts = self.split_parts(article.paragraphs or (article.text,))
            for part_index, part in enumerate(parts):
                chunks.append(
                    Chunk(
                        chunk_id=f"{parent_id}#p{part_index:02d}",
                        parent_id=parent_id,
                        law_id=law.law_id,
                        law_name=law.law_name,
                        version=law.version,
                        citation=law.citation,
                        article_no=article.article_no,
                        article_index=article.article_index,
                        part_index=part_index,
                        part_total=len(parts),
                        text=part,
                        embed_text=self.embed_text_of(law, article, part, part_index),
                        chapter=article.chapter,
                        section=article.section,
                        refs=article.refs,
                    )
                )

        return ChunkSet(
            parents=tuple(parents),
            chunks=tuple(chunks),
            stats={
                "law_id": law.law_id,
                "parents": len(parents),
                "chunks": len(chunks),
                "chars": sum(c.char_count for c in chunks),
            },
        )

    def chunk_all(self, laws: list[LawDocument]) -> ChunkSet:
        parents: list[ParentChunk] = []
        chunks: list[Chunk] = []
        for law in laws:
            part = self.chunk(law)
            parents.extend(part.parents)
            chunks.extend(part.chunks)

        stats = {
            "laws": len(laws),
            "parents": len(parents),
            "chunks": len(chunks),
            "chars": sum(c.char_count for c in chunks),
            "avg_chunk_chars": round(sum(c.char_count for c in chunks) / max(1, len(chunks)), 1),
            "min_part_chars": self.min_part_chars,
            "max_part_chars": self.max_part_chars,
        }
        return ChunkSet(parents=tuple(parents), chunks=tuple(chunks), stats=stats)

    @staticmethod
    def parent_id_of(law: LawDocument, article_index: int) -> str:
        return f"{law.law_id}@{law.version}#a{article_index:03d}"

    def embed_text_of(self, law: LawDocument, article, part: str, part_index: int) -> str:
        body = RE_ARTICLE_PREFIX.sub("", part) if part_index == 0 else part
        head = f"《{law.law_name}》"
        if self.with_chapter_prefix and article.chapter:
            head += f"{article.chapter} "
        return f"{head}{article.article_no}：{body}"

    def split_parts(self, paragraphs: tuple[str, ...]) -> list[str]:
        merged: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            is_list_item = bool(RE_LIST_MARKER.match(para))
            if merged and (is_list_item or len(para) < self.min_part_chars):
                merged[-1] = f"{merged[-1]}\n{para}"
            else:
                merged.append(para)

        parts: list[str] = []
        for para in merged:
            parts.extend(self._split_long(para))
        return parts or [""]

    def _split_long(self, text: str) -> list[str]:
        if len(text) <= self.max_part_chars:
            return [text]

        pieces: list[str] = []
        buffer = ""
        start = 0
        for match in RE_SENTENCE_END.finditer(text):
            end = match.end()
            if end - start >= self.max_part_chars:
                buffer += text[start:end]
                pieces.append(buffer.strip())
                buffer, start = "", end
        buffer += text[start:]
        if buffer.strip():
            pieces.append(buffer.strip())
        return pieces


class ChunkStage:

    layer = "chunk"
    input_desc = "list[LawDocument]"
    output_desc = "ChunkSet + chunks/chunks.jsonl + chunks/parents.jsonl"

    def __init__(
        self,
        *,
        library: LawLibrary | None = None,
        chunker: LawChunker | None = None,
        chunks_path: Path | None = None,
        parents_path: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.library = library or LawLibrary()
        self.chunker = chunker or LawChunker()
        self.chunks_path = Path(chunks_path or config.CHUNKS_PATH)
        self.parents_path = Path(parents_path or config.PARENTS_PATH)
        self.verbose = verbose

    def run(self, laws: list[LawDocument] | None = None) -> ChunkSet:
        laws = laws if laws is not None else self.library.load_all()
        chunk_set = self.chunker.chunk_all(laws)
        chunk_set.write(self.chunks_path, self.parents_path)
        if self.verbose:
            stats = chunk_set.stats
            print(
                f"[chunk] {stats['laws']} 部法规 | 父块 {stats['parents']} 条 | "
                f"子块 {stats['chunks']} 个 | 平均 {stats['avg_chunk_chars']} 字/块 → {self.chunks_path.parent}"
            )
        return chunk_set

    def load(self) -> ChunkSet:
        return ChunkSet.read(self.chunks_path, self.parents_path)


if __name__ == "__main__":
    raise SystemExit("已收口：请用 python -m traffic_law_qa.pipeline chunk（清单见 README「入口」）")
