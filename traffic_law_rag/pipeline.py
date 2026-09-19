"""交通法规 RAG 管道的端到端编排与命令行入口。

各层输入输出（也是 `python -m traffic_law_rag.pipeline layers` 打印的内容）：

| 层       | 类                 | 输入                            | 输出                        |
|----------|--------------------|---------------------------------|-----------------------------|
| read     | `DocxReader`       | docx 路径 `Path`                | `list[Paragraph]`           |
| parse    | `ParseStage`       | docx 目录                       | `list[LawDocument]`         |
| chunk    | `ChunkStage`       | `list[LawDocument]`             | `ChunkSet`                  |
| index    | `Indexer`          | `ChunkSet`                      | `IndexStats`                |
| rewrite  | `QueryRewriter`    | `Query`                         | `RewrittenQuery`            |
| retrieve | `HybridRetriever`  | `Query`                         | `RetrievalResult`           |
| generate | `AnswerGenerator`  | `Question` + `RetrievalResult`  | `Answer`                    |

命令行用法（只管知识库本身；**问答统一走 `python -m traffic_law_rag "问题"`** ——
那条路带一致性检查与自动重建，这里的 build/status/layers 都只管建库和查状态）：
    python -m traffic_law_rag.pipeline build  [--force] [--no-vector]
    python -m traffic_law_rag.pipeline layers | status
"""

from __future__ import annotations

import sys
import time

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
from .kb.docx_reader import DocxReader
from .kb.indexer import Indexer
from .kb.law_parser import LawLibrary, LawParser, ParseStage
from .qa.generator import AnswerGenerator
from .qa.rag import LegalRAG


class RagPipeline:
    """交通法规 RAG 管道：把七个层串起来，并负责每层的计时与报告。"""

    LAYERS: tuple[tuple[str, str, str, str], ...] = (
        ("read     ", "DocxReader", "docx 路径 (Path)", "list[Paragraph]"),
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
        reader: DocxReader | None = None,
        parser: LawParser | None = None,
        library: LawLibrary | None = None,
        parse_stage: ParseStage | None = None,
        chunk_stage: ChunkStage | None = None,
        indexer: Indexer | None = None,
        generator: AnswerGenerator | None = None,
        verbose: bool = True,
    ) -> None:
        self.reader = reader or DocxReader()
        self.parser = parser or LawParser()
        self.library = library or LawLibrary()
        self.parse_stage = parse_stage or ParseStage(
            reader=self.reader, parser=self.parser, library=self.library, verbose=verbose
        )
        self.chunk_stage = chunk_stage or ChunkStage(library=self.library, verbose=verbose)
        self.indexer = indexer or Indexer(verbose=verbose)
        self.generator = generator or AnswerGenerator()
        self.verbose = verbose
        self._rag: LegalRAG | None = None
        self._rag_with_vector: bool | None = None
        self._index_stats: IndexStats | None = None

    # ============================================================== 单层接口
    def read(self, docx_path) -> list[Paragraph]:
        """层 1：读单个 docx。"""
        return self.reader.read(docx_path)

    def parse(self, *, force: bool = False, only: str | None = None) -> list[LawDocument]:
        """层 2：docx 目录 → 结构层产物。"""
        return self.parse_stage.run(force=force, only=only)

    def chunk(self, laws: list[LawDocument] | None = None) -> ChunkSet:
        """层 3：结构层 → 父子块。"""
        return self.chunk_stage.run(laws)

    def index(self, chunk_set: ChunkSet, *, with_vector: bool = True) -> IndexStats:
        """层 4：父子块 → BM25 + 向量索引。"""
        self._index_stats = self.indexer.build(chunk_set, with_vector=with_vector)
        self._rag = None  # 索引变了，缓存的检索器作废
        self._rag_with_vector = None
        return self._index_stats

    def rag_tool(self, *, with_vector: bool = True) -> LegalRAG:
        """层 5+6 的门面（懒加载并缓存；向量开关变化时自动重建）。

        `generator=self.generator` 必须显式传：`LegalRAG.load()` 自己会 new 一个
        `AnswerGenerator`，不传就等于把管道持有的那一个（含测试注入的替身、
        含 `status()` 打印的模型名）静默丢掉。
        """
        if self._rag is None or self._rag_with_vector != with_vector:
            self._rag = LegalRAG.load(with_vector=with_vector, generator=self.generator)
            self._rag_with_vector = with_vector
        return self._rag

    # ============================================================== 全流程
    def build(self, *, force: bool = False, with_vector: bool = True) -> PipelineReport:
        """跑完 read → parse → chunk → index（read 体现在 parse 阶段内部）。"""
        stages: list[StageReport] = []

        # 1) read + 2) parse
        stage_started = time.perf_counter()
        laws = self.parse_stage.run(force=force)
        skipped = bool(self.parse_stage.last_skipped)
        stages.append(
            StageReport(
                name="parse",
                input_desc=f"{len(list(self.parse_stage.docx_dir.glob('*.docx')))} 个 docx",
                output_desc=f"{len(laws)} 部 / {sum(law.article_count for law in laws)} 条",
                ok=True,
                elapsed_ms=(time.perf_counter() - stage_started) * 1000,
                detail=f"跳过 {len(self.parse_stage.last_skipped)} 个未变文件" if skipped else "全部重新解析",
                skipped=skipped and not force,
            )
        )

        # 3) chunk
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

        # 4) index
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

    # ============================================================== 自述
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
            lines.append("索引：尚未构建（docker compose up -d 后运行 python -m traffic_law_rag.pipeline build）")

        milvus = config.milvus_config()
        try:
            lines.append(f"Milvus：{self.indexer.connect()} @ {milvus.uri}")
        except Exception as exc:  # noqa: BLE001 - 状态查询不该因连不上就崩
            lines.append(f"Milvus：连接失败（{milvus.uri}）—— {str(exc).splitlines()[0]}")
        lines.append(f"模型：LLM={self.generator.cfg.model} | Embedding={config.embed_config().model}")
        return "\n".join(lines)


# ================================================================== CLI
USAGE = __doc__


def _flag(args: list[str], name: str) -> bool:
    return name in args


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    # --help 必须在 command 推断之前拦掉。原来没有这个分支，而
    # `command = args[0] if ... not startswith("--") else "build"` 会把 `--help`
    # 当成「无子命令」→ 直接跑 build，也就是一次完整的重新切块 + 重新 embedding。
    # 查一次用法花掉一次 embedding 钱，这个坑踩过一次。
    if any(a in ("--help", "-h", "help") for a in args):
        print(USAGE)
        return 0

    command = args[0] if args and not args[0].startswith("--") else "build"
    pipeline = RagPipeline()

    if command == "layers":
        print(pipeline.layers())
        return 0

    if command == "status":
        print(pipeline.status())
        return 0

    if command == "build":
        report = pipeline.build(force=_flag(args, "--force"), with_vector=not _flag(args, "--no-vector"))
        print("\n管道执行结果：")
        print(report.render())
        return 0

    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
