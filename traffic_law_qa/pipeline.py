from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from . import config
from .contracts import (
    ChunkSet,
    IndexStats,
    LawDocument,
    Paragraph,
    PipelineReport,
    StageReport,
)
from .kb.chunker import ChunkStage
from .kb.indexer import Indexer
from .kb.law_parser import LawLibrary, LawParser, ParseStage, reader_for
from .kb.milvus_store import MilvusError, wait_until_ready
from .qa.generator import AnswerGenerator
from .qa.rag import LegalRAG


class RagPipeline:

    LAYERS: tuple[tuple[str, str, str, str], ...] = (
        ("read     ", "Docx/TextReader", "源文件 (Path)", "list[Paragraph]"),
        ("parse    ", "ParseStage", "docx 目录", "list[LawDocument]"),
        ("chunk    ", "ChunkStage", "list[LawDocument]", "ChunkSet"),
        ("index    ", "Indexer", "ChunkSet", "IndexStats"),
        ("rewrite  ", "QueryRewriter", "Query", "RewrittenQuery"),
        ("retrieve ", "HybridRetriever", "Query", "RetrievalResult"),
        ("generate ", "AnswerGenerator", "Question + RetrievalResult", "Answer"),
    )

    def __init__(
        self,
        *,
        readers: dict[str, Any] | None = None,
        parser: LawParser | None = None,
        library: LawLibrary | None = None,
        parse_stage: ParseStage | None = None,
        chunk_stage: ChunkStage | None = None,
        indexer: Indexer | None = None,
        generator: AnswerGenerator | None = None,
        verbose: bool = True,
    ) -> None:
        self.parser = parser or LawParser()
        self.library = library or LawLibrary()
        self.parse_stage = parse_stage or ParseStage(
            readers=readers, parser=self.parser, library=self.library, verbose=verbose
        )
        self.chunk_stage = chunk_stage or ChunkStage(library=self.library, verbose=verbose)
        self.indexer = indexer or Indexer(verbose=verbose)
        self.generator = generator or AnswerGenerator()
        self.verbose = verbose
        self._rag: LegalRAG | None = None
        self._rag_with_vector: bool | None = None
        self._index_stats: IndexStats | None = None

    def read(self, source_path) -> list[Paragraph]:
        return reader_for(source_path).read(source_path)

    def parse(self, *, force: bool = False, only: str | None = None) -> list[LawDocument]:
        return self.parse_stage.run(force=force, only=only)

    def chunk(self, laws: list[LawDocument] | None = None) -> ChunkSet:
        return self.chunk_stage.run(laws)

    def index(self, chunk_set: ChunkSet, *, with_vector: bool = True) -> IndexStats:
        self._index_stats = self.indexer.build(chunk_set, with_vector=with_vector)
        self._rag = None
        self._rag_with_vector = None
        return self._index_stats

    def rag_tool(self, *, with_vector: bool = True) -> LegalRAG:
        if self._rag is None or self._rag_with_vector != with_vector:
            self._rag = LegalRAG.load(with_vector=with_vector, generator=self.generator)
            self._rag_with_vector = with_vector
        return self._rag

    def build(self, *, force: bool = False, with_vector: bool = True) -> PipelineReport:
        stages: list[StageReport] = []

        stage_started = time.perf_counter()
        laws = self.parse_stage.run(force=force)
        skipped = bool(self.parse_stage.last_skipped)
        stages.append(
            StageReport(
                name="parse",
                input_desc=f"{len(config.source_files(self.parse_stage.source_dir))} 个源文件",
                output_desc=f"{len(laws)} 部 / {sum(law.article_count for law in laws)} 条",
                ok=True,
                elapsed_ms=(time.perf_counter() - stage_started) * 1000,
                detail=f"跳过 {len(self.parse_stage.last_skipped)} 个未变文件" if skipped else "全部重新解析",
                skipped=skipped and not force,
            )
        )

        stage_started = time.perf_counter()
        chunk_set = self.chunk_stage.run(laws)
        stages.append(
            StageReport(
                name="chunk",
                input_desc=f"{len(laws)} 部法规",
                output_desc=f"{len(chunk_set.parents)} 父块 / {len(chunk_set.chunks)} 子块",
                ok=True,
                elapsed_ms=(time.perf_counter() - stage_started) * 1000,
                detail=f"平均 {chunk_set.stats.get('avg_chunk_chars', 0)} 字/块",
            )
        )

        stage_started = time.perf_counter()
        stats = self.index(chunk_set, with_vector=with_vector)
        stages.append(
            StageReport(
                name="index",
                input_desc=f"{len(chunk_set.chunks)} 子块",
                output_desc=f"{stats.rows} 行 / 稠密 {stats.dense_rows} / BM25 {stats.sparse_rows}",
                ok=True,
                elapsed_ms=(time.perf_counter() - stage_started) * 1000,
                detail="；".join(self.indexer.notes) if self.indexer.notes else f"集合 {stats.collection}",
            )
        )

        return PipelineReport(
            stages=tuple(stages),
            law_count=len(laws),
            article_count=sum(law.article_count for law in laws),
            chunk_count=len(chunk_set.chunks),
            index=stats,
        )

    def layers(self) -> str:
        lines = ["RAG 管道各层：", ""]
        for index, (name, cls, input_desc, output_desc) in enumerate(self.LAYERS, start=1):
            lines.append(f"  {index}. {name} {cls:<18} {input_desc:<26} → {output_desc}")
        return "\n".join(lines)

    def status(self) -> str:
        manifest = self.library.manifest()
        stats = self.indexer.load_stats()
        lines = [
            f"知识库：{len(manifest.get('laws', []))} 部法规 / "
            f"{sum(item['articles'] for item in manifest.get('laws', []))} 条",
            f"清单生成时间：{manifest.get('generated_at')}",
        ]
        for item in manifest.get("laws", []):
            lines.append(
                f"  - {item['law_id']:<30} {item['articles']:>4} 条  {item['version']}  {item['law_name']}"
            )
        if stats:
            dense = (
                f"{stats.dense_rows} 行（{stats.embedding_dim} 维，{stats.embedding_model}）"
                if stats.vector_enabled
                else "未启用（纯 BM25）"
            )
            lines.append(
                f"索引：Milvus 集合 {stats.collection} | 共 {stats.rows} 行 | "
                f"BM25 稀疏 {stats.sparse_rows} 行 | 稠密 {dense} | 建于 {stats.built_at}"
            )
            for note in self.indexer.load_notes():
                lines.append(f"  提示：{note}")
        else:
            lines.append("索引：尚未构建（docker compose up -d 后运行 python -m traffic_law_qa.pipeline build）")

        milvus = config.milvus_config()
        try:
            lines.append(f"Milvus：{self.indexer.connect()} @ {milvus.uri}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"Milvus：连接失败（{milvus.uri}）—— {str(exc).splitlines()[0]}")
        lines.append(f"模型：LLM={self.generator.cfg.model} | Embedding={config.embed_config().model}")
        return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traffic_law_qa.pipeline",
        description='建库与查状态（问答走 python -m traffic_law_qa "问题"）',
    )
    subs = parser.add_subparsers(
        dest="command", required=True, metavar="{build,status,layers,docx,parse,chunk,index}"
    )

    build = subs.add_parser(
        "build",
        help="整条建库：parse → chunk → index（手工把法规放进 docx/ 后跑这个；"
        "HTTP 上传走 POST /documents + POST /reindex，不用手工跑）",
    )
    build.add_argument("--force", action="store_true", help="无视 sha1 增量门控，全部重新解析")
    build.add_argument("--no-vector", action="store_true", help="只建 BM25 字段，不算向量")

    subs.add_parser("status", help="看库内规模、索引快照与 Milvus 连接")
    subs.add_parser("layers", help="看七层管道各自的输入输出")

    docx = subs.add_parser("docx", help="单步：源文件 → 段落（不给路径就跑 SOURCE_DIR 全部）")
    docx.add_argument("paths", nargs="*", metavar="源文件", help="要预览的源文件路径")

    parse = subs.add_parser("parse", help="单步：段落 → 法→章→节→条")
    parse.add_argument("--force", action="store_true", help="无视 sha1 门控，全部重解析")
    parse.add_argument("--only", metavar="法id", default=None, help="只解析这一部（如 sz_icv_regulation）")

    chunk = subs.add_parser("chunk", help="单步：条文 → 父子块")
    chunk.add_argument("--show", metavar="条号", default=None, help="切完打印该条号的子块（如 第八条）")

    index = subs.add_parser("index", help="单步：块 → Milvus 集合（先等 Milvus 就绪）")
    index.add_argument("--no-vector", action="store_true", help="只建 BM25 字段，不算向量")
    index.add_argument("--query", metavar="词", default=None, help="建完用纯 BM25 试查一次（top-5）")
    return parser


def _cmd_docx(paths: list[str]) -> int:
    targets = [Path(a) for a in paths] if paths else config.source_files()

    for target in targets:
        paragraphs = reader_for(target).read(target)
        chars = sum(len(p.text) for p in paragraphs)
        print(f"\n=== {target.name} | 段落 {len(paragraphs)} | 字符 {chars}")
        for para in paragraphs[:12]:
            print(f"  [{para.index:>3}] style={para.style!s:<10} {para.text[:60]}")
    return 0


def _cmd_parse(*, force: bool, only: str | None) -> int:
    ParseStage().run(force=force, only=only)
    return 0


def _cmd_chunk(show: str | None) -> int:
    chunk_set = ChunkStage().run()

    if show:
        for chunk in chunk_set.chunks:
            if chunk.article_no == show:
                print(f"\n[{chunk.chunk_id}] part {chunk.part_index + 1}/{chunk.part_total}")
                print(f"  embed_text: {chunk.embed_text}")
    return 0


def _cmd_index(*, with_vector: bool, query: str | None) -> int:
    chunk_set = ChunkStage(verbose=False).load()
    indexer = Indexer()
    print(f"[index] Milvus 版本 {wait_until_ready(indexer.store)}")
    try:
        indexer.build(chunk_set, with_vector=with_vector)
    except MilvusError as exc:
        print(f"[index] 失败：{exc}")
        return 1

    if query:
        store = indexer.store
        print(f"\n仅 BM25 通道：{query}")
        for chunk_id, score in store.sparse_search(query, limit=5):
            print(f"  {score:8.3f}  {chunk_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()

    if not args:
        parser.print_help()
        return 2

    try:
        options = parser.parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    if options.command == "layers":
        print(RagPipeline().layers())
        return 0

    if options.command == "status":
        print(RagPipeline().status())
        return 0

    if options.command == "build":
        report = RagPipeline().build(force=options.force, with_vector=not options.no_vector)
        print("\n管道执行结果：")
        print(report.render())
        return 0

    if options.command == "docx":
        return _cmd_docx(options.paths)
    if options.command == "parse":
        return _cmd_parse(force=options.force, only=options.only)
    if options.command == "chunk":
        return _cmd_chunk(options.show)
    return _cmd_index(with_vector=not options.no_vector, query=options.query)


if __name__ == "__main__":
    raise SystemExit(main())
