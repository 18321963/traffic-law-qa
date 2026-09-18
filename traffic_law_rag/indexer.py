"""Layer 4 · index：ChunkSet → Milvus 集合（稠密向量 + BM25 稀疏向量）。

    EmbeddingClient.embed : list[str] → list[list[float]]（OpenAI 兼容端点）
    Indexer.build         : ChunkSet → IndexStats

为什么这一层比之前薄了很多：
- 稀疏向量（BM25）由 Milvus 的 BM25 函数在服务端生成，我们只写原文；
- 倒排索引、df/postings、分词、落盘 bm25.json 全部不再需要；
- 融合也移到服务端（retriever 里用 hybrid_search + RRFRanker）。

没配 EMBED_API_KEY 时仍可建库：集合里只建 text / sparse 字段，纯 BM25 跑通整条管道。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .contracts import ChunkSet, IndexStats
from .milvus_store import MilvusError, MilvusStore, row_of

# 向量端点挂了时的降级标记，写进 index_meta.json 的 notes。
# api 层据此区分「没打算建稠密」和「想建但端点挂了」，避免每次调用都重试重建（会反复 drop 集合）。
EMBED_FAILED_NOTE_PREFIX = "向量化失败"


class EmbeddingClient:
    """OpenAI 兼容 /embeddings 端点封装（通义百炼 / 智谱 / OpenAI …）。

    Input : list[str]
    Output: list[list[float]]
    """

    input_desc = "list[str]"
    output_desc = "list[list[float]]"

    def __init__(self, cfg: config.EmbedConfig | None = None, *, retries: int = 3) -> None:
        self.cfg = cfg or config.embed_config()
        self.retries = retries
        self._client = None

    @property
    def available(self) -> bool:
        return self.cfg.ready

    @property
    def unavailable_reason(self) -> str:
        if not self.cfg.api_key:
            return "未配置 EMBED_API_KEY / LLM_API_KEY，只建 BM25 稀疏索引（纯关键词检索）"
        return ""

    @property
    def model_label(self) -> str:
        return f"{self.cfg.model}@{self.cfg.base_url}"

    def _ensure_client(self):
        if self._client is None:
            if not self.available:
                raise RuntimeError(self.unavailable_reason)
            from openai import OpenAI  # 延迟导入

            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """分批向量化，失败按指数退避重试。"""
        client = self._ensure_client()
        vectors: list[list[float]] = []
        batch = max(1, self.cfg.batch)
        for start in range(0, len(texts), batch):
            vectors.extend(self._embed_window(client, texts[start : start + batch]))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_window(self, client, window: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {}
        if self.cfg.dim:
            kwargs["dimensions"] = self.cfg.dim
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = client.embeddings.create(model=self.cfg.model, input=window, **kwargs)
                return [item.embedding for item in response.data]
            except Exception as exc:  # noqa: BLE001 - 网络/配额错误统一重试
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"向量化失败（{self.cfg.model}）：{last_error}") from last_error


class Indexer:
    """建索引阶段：父子块 → Milvus 集合。

    Input : ChunkSet
    Output: IndexStats（同时写 index/index_meta.json 快照）
    """

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

    # -------------------------------------------------------------- 建库
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
            except Exception as exc:  # noqa: BLE001 - 向量端点挂了也要能建库
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

    # -------------------------------------------------------------- 读取
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
        """探活；返回 Milvus 版本号。"""
        return self.store.ping()


# ------------------------------------------------------------------ 调试入口
def main(argv: list[str] | None = None) -> int:
    """python -m traffic_law_rag.indexer [--no-vector] [--query 词]"""
    import sys

    from .chunker import ChunkStage
    from .milvus_store import wait_until_ready

    args = list(sys.argv[1:] if argv is None else argv)
    chunk_set = ChunkStage(verbose=False).load()
    indexer = Indexer()
    print(f"[index] Milvus 版本 {wait_until_ready(indexer.store)}")
    try:
        indexer.build(chunk_set, with_vector="--no-vector" not in args)
    except MilvusError as exc:
        print(f"[index] 失败：{exc}")
        return 1

    if "--query" in args:
        query = args[args.index("--query") + 1]
        store = indexer.store
        print(f"\n仅 BM25 通道：{query}")
        for chunk_id, score in store.sparse_search(query, limit=5):
            print(f"  {score:8.3f}  {chunk_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
