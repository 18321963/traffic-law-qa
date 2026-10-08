from __future__ import annotations

from rag_contracts import config
from rag_contracts.domain.disk import IndexStats
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.reports import WEIGHTS_MISMATCH, WEIGHTS_OK, WEIGHTS_UNKNOWN
from rag_contracts.domain.textutil import sha1_of
from rag_contracts.ports import Embedder, IndexStatus, VectorStore

from ..adapters.embedding import EMBED_FAILED_NOTE_PREFIX
from ..adapters.milvus import MilvusError
from .indexer import Indexer
from .parser import LawLibrary

__all__ = ["CorpusStatus"]


class CorpusStatus(IndexStatus):

    def __init__(
        self,
        *,
        store: VectorStore,
        indexer: Indexer,
        embedder: Embedder,
        library: LawLibrary,
    ) -> None:
        self._store = store
        self._indexer = indexer
        self._embedder = embedder
        self._library = library

    @property
    def uri(self) -> str:
        return self._store.uri

    def ping(self) -> str:
        try:
            return self._store.ping()
        except MilvusError:
            raise QaError(
                f"Milvus 连不上（{self._store.uri}）→ 先执行：{config.COMPOSE} up -d --wait"
            ) from None

    def has_collection(self) -> bool:
        return self._store.has_collection()

    def rows(self) -> int:
        return self._store.count()

    def snapshot(self) -> IndexStats | None:
        return self._indexer.load_stats()

    def notes(self) -> list[str]:
        return self._indexer.load_notes()

    def counts(self) -> tuple[int, int]:
        entries = self._manifest()
        return len(entries), sum(int(item.get("articles", 0)) for item in entries)

    def desired_embedding_label(self) -> str:
        return self._embedder.model_label

    def weights(self) -> str:
        stats = self.snapshot()
        recorded = stats.embedding_fingerprint if stats is not None else None
        if not recorded or not stats.vector_enabled:
            return WEIGHTS_UNKNOWN
        actual = self._embedder.fingerprint
        if not actual:
            return WEIGHTS_UNKNOWN
        return WEIGHTS_OK if actual == recorded else WEIGHTS_MISMATCH

    def blocked_reason(self) -> str | None:
        if self.weights() != WEIGHTS_MISMATCH:
            return None
        stats = self.snapshot()
        return (
            f"{config.embed_config().model} 的权重与索引快照对不上："
            f"快照记的是 {stats.embedding_fingerprint}，当前算出 {self._embedder.fingerprint}。"
            "指纹按「文件清单 + 大小 + mtime」算，防误换不防篡改；"
            "换了权重、或整目录拷贝导致 mtime 变了，就显式重建一次："
            "POST /reindex，或设 RAG_ALLOW_REBUILD=1 后重启。"
        )

    def stale_reason(self, *, want_dense: bool) -> str | None:
        entries = self._manifest()
        if not entries:
            return "结构层产物缺失（parsed/manifest.json 为空）"

        for entry in entries:
            if not self._library.path_of(entry["law_id"]).exists():
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

        stats = self.snapshot()
        if stats is None:
            return "索引快照缺失（index/index_meta.json）"
        if not self.has_collection():
            return f"Milvus 集合 {self._store.collection} 不存在"
        actual = self.rows()
        if actual != stats.rows:
            return f"集合行数（{actual}）与索引快照（{stats.rows}）不一致"

        if want_dense and stats.vector_enabled:
            want_label = self.desired_embedding_label()
            if stats.embedding_model != want_label:
                return (
                    f"索引快照的向量模型（{stats.embedding_model}）"
                    f"与当前配置（{want_label}）不一致"
                )

        if want_dense and not stats.vector_enabled:
            if any(note.startswith(EMBED_FAILED_NOTE_PREFIX) for note in self.notes()):
                return None
            return "索引快照为纯 BM25，本次需要向量通道，重建以补上稠密向量字段"

        return None

    def _manifest(self) -> list[dict]:
        return self._library.manifest().get("laws", [])


def _count_lines(path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
