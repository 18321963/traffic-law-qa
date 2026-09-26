from __future__ import annotations

from .. import config
from ..contracts.disk import IndexStats
from ..contracts.errors import QaError
from ..infra.embedding import EMBED_FAILED_NOTE_PREFIX
from ..infra.milvus import MilvusError
from ..ports import Embedder, IndexStatus, VectorStore
from ..textutil import sha1_of
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
        library: LawLibrary | None = None,
    ) -> None:
        self._store = store
        self._indexer = indexer
        self._embedder = embedder
        self._library = library or LawLibrary()

    @property
    def uri(self) -> str:
        return self._store.uri

    def ping(self) -> str:
        try:
            return self._store.ping()
        except MilvusError:
            raise QaError(
                f"Milvus 连不上（{self._store.uri}）→ 先执行：docker compose up -d --wait"
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
