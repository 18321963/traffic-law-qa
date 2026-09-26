from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .. import config
from ..contracts.disk import ChunkSet, IndexStats, LawDocument, Paragraph
from ..contracts.reports import PipelineReport, StageReport
from ..ports import IndexBuilder
from .chunker import ChunkStage
from .indexer import Indexer
from .parser import LawLibrary, LawParser, ParseStage, reader_for


class RagPipeline(IndexBuilder):

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
        verbose: bool = True,
    ) -> None:
        self.parser = parser or LawParser()
        self.library = library or LawLibrary()
        self.parse_stage = parse_stage or ParseStage(
            readers=readers, parser=self.parser, library=self.library, verbose=verbose
        )
        self.chunk_stage = chunk_stage or ChunkStage(library=self.library, verbose=verbose)
        self.indexer = indexer or Indexer(verbose=verbose)
        self.verbose = verbose
        self._index_stats: IndexStats | None = None

    def read(self, source_path) -> list[Paragraph]:
        return reader_for(source_path).read(source_path)

    def parse(self, *, force: bool = False, only: str | None = None) -> list[LawDocument]:
        return self.parse_stage.run(force=force, only=only)

    def chunk(self, laws: list[LawDocument] | None = None) -> ChunkSet:
        return self.chunk_stage.run(laws)

    def index(self, chunk_set: ChunkSet, *, with_vector: bool = True) -> IndexStats:
        self._index_stats = self.indexer.build(chunk_set, with_vector=with_vector)
        return self._index_stats

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
            lines.append("索引：尚未构建（docker compose up -d 后运行 python -m traffic_law_qa.cli.build build）")

        milvus = config.milvus_config()
        try:
            lines.append(f"Milvus：{self.indexer.connect()} @ {milvus.uri}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"Milvus：连接失败（{milvus.uri}）—— {str(exc).splitlines()[0]}")
        embed = config.embed_config()
        rerank = config.rerank_config()
        lines.append(
            f"模型：LLM={config.llm_config().model} | 嵌入={Path(embed.model).name} | "
            f"重排={Path(rerank.model).name if rerank.enabled else '关'}"
        )
        return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit("已收口：请用 python -m traffic_law_qa.cli.build（清单见 README「入口」）")
