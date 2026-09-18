"""Layer 3 · chunk：LawDocument → ChunkSet（条级父块 + 款级子块）。

    LawChunker.chunk : LawDocument → ChunkSet          （纯计算，不碰磁盘）
    ChunkStage       : list[LawDocument] → ChunkSet    （并写 chunks/*.jsonl）

为什么要父子块：
- 检索要细粒度（命中"第一百二十四条第二款"这种具体表述），
  但生成要完整上下文（只给一款会让 LLM 丢掉条内主体）。
- 所以子块（款）进索引，命中后回灌父块（整条）给 LLM。

切块规则（决定 demo 效果）：
1. 以条为父块；款为子块。
2. 列举项（「（一）…」）无条件并入前一款 —— 这类项脱离引出句后既搜不到也读不懂，
   实测占子块总数的 29%。
3. 短于 min_part_chars 的款同样并入前一款。
4. 长于 max_part_chars 的款按「。；」切分，父块不变。
5. 子块 text 存原文，embed_text 额外拼上「法名 + 章 + 条号」前缀，
   缓解"多部法规讲同一件事"（如道交法与深圳处罚条例）时的张冠李戴。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import config
from .contracts import Chunk, ChunkSet, LawDocument, ParentChunk
from .law_parser import LawLibrary

RE_ARTICLE_PREFIX = re.compile(r"^第[零一二三四五六七八九十百千]+条[\s　]*")
RE_SENTENCE_END = re.compile(r"[。；]")
# 列举项开头：（一）/1./①/一、
# 分几个片段拼是为了行宽（全角字符按两列算），拼接后与原表达式逐字符相同。
RE_LIST_MARKER = re.compile(
    r"^\s*(?:[（(][一二三四五六七八九十百千\d]+[）)]"
    r"|[①-⑳]"
    r"|[一二三四五六七八九十]+[、.]"
    r"|\d+[.．、])"
)


class LawChunker:
    """切块器：把结构化法条切成父子块。

    Input : LawDocument
    Output: ChunkSet
    """

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

    # -------------------------------------------------------------- 主接口
    def chunk(self, law: LawDocument) -> ChunkSet:
        """切一部法规。"""
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
        """切多部法规并合并成一个 ChunkSet。"""
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

    # -------------------------------------------------------------- 命名规则
    @staticmethod
    def parent_id_of(law: LawDocument, article_index: int) -> str:
        """父块 id：law_id@版本#条序号 —— 含版本，避免新旧版本串号。"""
        return f"{law.law_id}@{law.version}#a{article_index:03d}"

    def embed_text_of(self, law: LawDocument, article, part: str, part_index: int) -> str:
        """向量化用文本 = 《法名》章 条号：正文（第一条去掉重复的条号前缀）。"""
        body = RE_ARTICLE_PREFIX.sub("", part) if part_index == 0 else part
        head = f"《{law.law_name}》"
        if self.with_chapter_prefix and article.chapter:
            head += f"{article.chapter} "
        return f"{head}{article.article_no}：{body}"

    # -------------------------------------------------------------- 切分规则
    def split_parts(self, paragraphs: tuple[str, ...]) -> list[str]:
        """款级切分：合并列举项 → 合并过短款 → 切分过长款。"""
        merged: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            is_list_item = bool(RE_LIST_MARKER.match(para))
            if merged and (is_list_item or len(para) < self.min_part_chars):
                # 列举项（项）必须附在引出它的那一款上，否则「（一）兜售物品…」
                # 这种子块既搜不到也读不懂
                merged[-1] = f"{merged[-1]}\n{para}"
            else:
                merged.append(para)

        parts: list[str] = []
        for para in merged:
            parts.extend(self._split_long(para))
        return parts or [""]

    def _split_long(self, text: str) -> list[str]:
        """超长款按句末标点切分，保持在 max_part_chars 以内。"""
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
    """Layer 3 的落盘封装：结构层 → 检索层产物。

    Input : list[LawDocument]（缺省时自动从 parsed/ 读取）
    Output: ChunkSet（同时写 chunks/chunks.jsonl、chunks/parents.jsonl）
    """

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


# ------------------------------------------------------------------ 调试入口
def main(argv: list[str] | None = None) -> int:
    """python -m traffic_law_rag.chunker [--show 条号]"""
    args = list(sys.argv[1:] if argv is None else argv)
    chunk_set = ChunkStage().run()

    if "--show" in args:
        target = args[args.index("--show") + 1]
        for chunk in chunk_set.chunks:
            if chunk.article_no == target:
                print(f"\n[{chunk.chunk_id}] part {chunk.part_index + 1}/{chunk.part_total}")
                print(f"  embed_text: {chunk.embed_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
