from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..contracts import ChunkSet, IndexStats
from ..services.embedding import EMBED_FAILED_NOTE_PREFIX, EmbeddingClient
from .milvus_store import MilvusStore, row_of


class Indexer:

    layer = "index"
    input_desc = "ChunkSet"
    output_desc = "IndexStats + Milvus 集合（dense + BM25 sparse）"

    def __init__(
        self,
        *,
        store: MilvusStore | None = None,
        embedder: EmbeddingClient | None = None,
        meta_path: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.store = store or MilvusStore(verbose=verbose)
        self.embedder = embedder or EmbeddingClient()
        self.meta_path = Path(meta_path or config.INDEX_META_PATH)
        self.verbose = verbose
        self.notes: list[str] = []

    def build(self, chunk_set: ChunkSet, *, with_vector: bool = True) -> IndexStats:
        started = time.perf_counter()
        self.notes = []

        texts = [chunk.embed_text for chunk in chunk_set.chunks]
        vectors: list[list[float]] | None = None
        dim: int | None = None

        if with_vector and self.embedder.available:
            try:
                vectors = self.embedder.embed(texts)
                dim = len(vectors[0]) if vectors else None
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"{EMBED_FAILED_NOTE_PREFIX}，降级为纯 BM25：{exc}")
                vectors, dim = None, None
        elif with_vector:
            self.notes.append(self.embedder.unavailable_reason)
        else:
            self.notes.append("已按参数跳过稠密向量（--no-vector），只建 BM25 稀疏索引")

        schema = self.store.recreate(dim=dim)
        rows = [
            row_of(chunk, vectors[index] if vectors else None)
            for index, chunk in enumerate(chunk_set.chunks)
        ]
        written = self.store.insert(rows)
        rows_in_collection = self.store.count()

        built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        stats = IndexStats(
            collection=self.store.collection,
            uri=self.store.uri,
            rows=rows_in_collection,
            dense_rows=written if dim is not None else 0,
            sparse_rows=rows_in_collection,
            embedding_model=self.embedder.model_label if dim is not None else None,
            embedding_dim=dim,
            vector_enabled=dim is not None,
            built_at=built_at,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            schema=schema,
        )

        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(
                {"stats": stats.to_dict(), "notes": self.notes, "chunks": chunk_set.stats},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if self.verbose:
            dense = f"{stats.dense_rows} 行（{stats.embedding_dim} 维）" if stats.vector_enabled else "未启用"
            print(
                f"[index] Milvus {self.store.uri} → 集合 {stats.collection} | "
                f"稠密 {dense} | BM25 稀疏 {stats.sparse_rows} 行 | {stats.elapsed_ms:.0f}ms"
            )
            for note in self.notes:
                print(f"[index] 提示：{note}")
        return stats

    def load_stats(self) -> IndexStats | None:
        if not self.meta_path.exists():
            return None
        payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return IndexStats.from_dict(payload["stats"])

    def load_notes(self) -> list[str]:
        if not self.meta_path.exists():
            return []
        return list(json.loads(self.meta_path.read_text(encoding="utf-8")).get("notes", []))

    def connect(self) -> str:
        return self.store.ping()


if __name__ == "__main__":
    raise SystemExit("已收口：请用 python -m traffic_law_qa.pipeline index（清单见 README「入口」）")
