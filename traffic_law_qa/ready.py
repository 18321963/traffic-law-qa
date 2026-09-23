from __future__ import annotations

from dataclasses import dataclass

from . import config
from .contracts import QaError

__all__ = ["ReadyState", "ensure_ready", "stale_reason"]


@dataclass(frozen=True)
class ReadyState:

    action: str
    reason: str
    rows: int
    laws: int
    articles: int
    parents: int
    dense: bool
    milvus: str

    def describe(self) -> str:
        verb = "复用已有索引" if self.action == "reuse" else "已重建索引"
        channels = "稠密+BM25" if self.dense else "纯 BM25"
        return (
            f"{verb}：{self.laws} 部法规 / {self.articles} 条 / {self.parents} 父块，"
            f"集合 {self.rows} 行（{channels}）｜{self.reason}｜Milvus {self.milvus}"
        )


def ensure_ready(*, with_vector: bool = True, rebuild: bool = False) -> ReadyState:
    from .kb.indexer import Indexer
    from .kb.milvus_store import MilvusStore

    docx_files = config.source_files()
    if not docx_files:
        raise QaError(f"知识库为空：{config.SOURCE_DIR} 下没有 {config.SOURCE_SUFFIXES} 文件")

    store = MilvusStore(verbose=False)
    version = _ping(store)
    stats = Indexer(verbose=False).load_stats()

    build_dense = config.embed_config().ready and (
        with_vector or bool(stats is not None and stats.vector_enabled)
    )

    reason = "指定了 rebuild=True" if rebuild else stale_reason(store, stats, want_dense=build_dense)
    if reason is not None:
        print(f"[qa] 索引需要重建（{reason}），开始建库；首次约 30~60 秒…")
        from .pipeline import RagPipeline

        RagPipeline(verbose=True).build(force=rebuild, with_vector=build_dense)
        stats = Indexer(verbose=False).load_stats()
        if stats is None:
            raise QaError(f"建库未写出索引快照：{config.INDEX_META_PATH}")

    laws, articles = _manifest_counts()
    return ReadyState(
        action="rebuild" if reason is not None else "reuse",
        reason=reason or "docx、本地产物与集合三者一致",
        rows=stats.rows,
        laws=laws,
        articles=articles,
        parents=articles,
        dense=stats.vector_enabled,
        milvus=version,
    )


def _ping(store) -> str:
    from .kb.milvus_store import MilvusError

    try:
        return store.ping()
    except MilvusError:
        raise QaError(
            f"Milvus 连不上（{store.uri}）→ 先执行：docker compose up -d --wait"
        ) from None


def stale_reason(store, stats, *, want_dense: bool) -> str | None:
    from .kb.law_parser import LawLibrary, sha1_of

    library = LawLibrary()
    entries = library.manifest().get("laws", [])
    if not entries:
        return "结构层产物缺失（parsed/manifest.json 为空）"

    for entry in entries:
        if not library.path_of(entry["law_id"]).exists():
            return f"缺少结构层产物 {entry['law_id']}.json"

    cached = {entry["file"]: entry for entry in entries}
    docx_files = config.source_files()
    if len(docx_files) != len(entries):
        return f"源文件数量（{len(docx_files)}）与清单（{len(entries)}）不一致"
    for path in docx_files:
        entry = cached.get(path.name)
        if entry is None:
            return f"新法规未入库：{path.name}"
        if entry.get("sha1") != sha1_of(path):
            return f"{path.name} 已改动（sha1 与清单不一致）"

    if not config.CHUNKS_PATH.exists() or not config.PARENTS_PATH.exists():
        return "检索层产物缺失（chunks/chunks.jsonl 或 parents.jsonl）"
    articles = sum(int(item.get("articles", 0)) for item in entries)
    parents_on_disk = _count_lines(config.PARENTS_PATH)
    if parents_on_disk != articles:
        return f"父块数（{parents_on_disk}）与条文数（{articles}）不一致，切块层产物已过期"

    if stats is None:
        return "索引快照缺失（index/index_meta.json）"
    if not store.has_collection():
        return f"Milvus 集合 {store.collection} 不存在"
    actual = store.count()
    if actual != stats.rows:
        return f"集合行数（{actual}）与索引快照（{stats.rows}）不一致"

    if want_dense and stats.vector_enabled:
        from .services.embedding import EmbeddingClient

        want_label = EmbeddingClient().model_label
        if stats.embedding_model != want_label:
            return (
                f"索引快照的向量模型（{stats.embedding_model}）"
                f"与当前配置（{want_label}）不一致"
            )

    if want_dense and not stats.vector_enabled:
        from .kb.indexer import Indexer
        from .services.embedding import EMBED_FAILED_NOTE_PREFIX

        notes = Indexer(verbose=False).load_notes()
        if any(note.startswith(EMBED_FAILED_NOTE_PREFIX) for note in notes):
            return None
        return "索引快照为纯 BM25，本次需要向量通道，重建以补上稠密向量字段"

    return None


def _manifest_counts() -> tuple[int, int]:
    from .kb.law_parser import LawLibrary

    entries = LawLibrary().manifest().get("laws", [])
    return len(entries), sum(int(item.get("articles", 0)) for item in entries)


def _count_lines(path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
