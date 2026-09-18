"""Layer 4 · index：ChunkSet → 可检索索引（BM25 倒排 + 向量库）。

    Indexer.build : ChunkSet → IndexStats   （并写 index/bm25.json、index/chroma、index/index_meta.json）

为什么是混合检索：
- 法规问答里大量查询带**条号或专有名词**（"第十九条""智能网联汽车道路测试"），
  BM25 命中率远高于向量；
- 而口语化提问（"喝了酒开车会怎么样"）向量更稳。
两条通道用 RRF 融合（在 retriever 层），任一条缺失都能降级运行。
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .contracts import ChunkSet, IndexStats

# 领域词：jieba 默认词典对法规术语切不准，这里补几个高频词
DOMAIN_WORDS = (
    "智能网联汽车",
    "非机动车",
    "机动车",
    "道路交通安全",
    "交通事故",
    "交通警察",
    "公安机关交通管理部门",
    "违法行为",
    "记分",
    "暂扣",
    "吊销",
    "扣留",
    "罚款",
    "追究刑事责任",
)


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
            return "未配置 EMBED_API_KEY / LLM_API_KEY，向量检索已跳过"
        return ""

    @property
    def model_label(self) -> str:
        return f"{self.cfg.model}@{self.cfg.base_url}"

    def _ensure_client(self):
        if self._client is None:
            if not self.available:
                raise RuntimeError(self.unavailable_reason)
            from openai import OpenAI  # 延迟导入：BM25 单通道模式无需 openai

            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """分批向量化，失败按指数退避重试。"""
        client = self._ensure_client()
        vectors: list[list[float]] = []
        batch = max(1, self.cfg.batch)
        for start in range(0, len(texts), batch):
            window = texts[start : start + batch]
            vectors.extend(self._embed_window(client, window))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_window(self, client, window: list[str]) -> list[list[float]]:
        kwargs = {}
        if self.cfg.dim:
            kwargs["dimensions"] = self.cfg.dim
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = client.embeddings.create(model=self.cfg.model, input=window, **kwargs)
                return [item.embedding for item in response.data]
            except Exception as exc:  # noqa: BLE001 - 需要把网络/配额错误统一重试
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"向量化失败（{self.cfg.model}）：{last_error}") from last_error


class BM25Index:
    """BM25 倒排索引（Okapi），纯 Python 实现，可存 JSON 便于排查。

    Input : [(doc_id, text)]
    Output: [(doc_id, score)]
    """

    input_desc = "list[(doc_id, text)]"
    output_desc = "list[(doc_id, score)]"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, dict[int, int]] = {}
        self.df: dict[str, int] = {}
        self.avgdl = 0.0
        self.built_at: str | None = None
        self._tokenizer = None

    # -------------------------------------------------------------- 分词
    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            import jieba  # 延迟导入

            jieba.setLogLevel(20)
            for word in DOMAIN_WORDS:
                jieba.add_word(word)
            self._tokenizer = jieba
        return self._tokenizer

    def tokenize(self, text: str) -> list[str]:
        jieba = self._ensure_tokenizer()
        return [tok for tok in jieba.lcut(text) if tok.strip() and tok not in "　 \t\n"]

    def tokenize_query(self, text: str) -> list[str]:
        """查询侧用搜索引擎模式切词，提升长术语的召回。"""
        jieba = self._ensure_tokenizer()
        return [tok for tok in jieba.lcut_for_search(text) if tok.strip() and tok not in "　 \t\n"]

    # -------------------------------------------------------------- 构建
    def build(self, docs: list[tuple[str, str]]) -> None:
        self.doc_ids = [doc_id for doc_id, _ in docs]
        self.doc_len = []
        self.postings = {}
        df_counter: Counter[str] = Counter()

        for index, (_, text) in enumerate(docs):
            tokens = self.tokenize(text)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, {})[index] = tf
                df_counter[term] += 1

        self.df = dict(df_counter)
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # -------------------------------------------------------------- 检索
    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        scores: dict[int, float] = {}
        total = len(self.doc_ids)
        for term in set(self.tokenize_query(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for doc_index, tf in postings.items():
                doc_len = self.doc_len[doc_index] or 1
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * tf * (self.k1 + 1) / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.doc_ids[index], score) for index, score in ranked]

    # -------------------------------------------------------------- 持久化
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "df": self.df,
            "postings": self.postings,
            "avgdl": self.avgdl,
            "built_at": self.built_at,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        if not path.exists():
            raise FileNotFoundError(f"缺少 BM25 索引：{path}（请先运行建索引阶段）")
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        index.doc_ids = payload["doc_ids"]
        index.doc_len = payload["doc_len"]
        index.df = payload["df"]
        index.postings = {term: {int(k): v for k, v in postings.items()} for term, postings in payload["postings"].items()}
        index.avgdl = payload.get("avgdl", 0.0)
        index.built_at = payload.get("built_at")
        return index


class VectorIndex:
    """Chroma 持久化向量库。

    Input : list[Chunk]（建库）/ str（查询）
    Output: int（建库条数）/ list[(chunk_id, score)]（查询）
    """

    input_desc = "list[Chunk] | str"
    output_desc = "int | list[(chunk_id, score)]"

    def __init__(
        self,
        *,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION
        self.embedder = embedder or EmbeddingClient()
        self._client = None
        self._collection = None
        self.dim: int | None = None

    @property
    def available(self) -> bool:
        return self.embedder.available

    def _ensure_collection(self, *, recreate: bool = False):
        if self._collection is not None and not recreate:
            return self._collection

        import chromadb  # 延迟导入

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        if recreate:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:  # noqa: BLE001 - 集合不存在时忽略
                pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # 向量由我们自己算，禁用 chroma 默认 ONNX 模型
        )
        return self._collection

    # -------------------------------------------------------------- 构建
    def build(self, chunks, *, batch_size: int = 0) -> int:
        """重建集合（幂等）。返回写入条数。"""
        collection = self._ensure_collection(recreate=True)
        texts = [chunk.embed_text for chunk in chunks]
        # chroma 的 embeddings 形参用了 numpy 类型标法，实际接受 list[list[float]]；用 Any 避免误报
        vectors: list[Any] = self.embedder.embed(texts)
        if vectors:
            self.dim = len(vectors[0])

        step = batch_size or max(1, self.embedder.cfg.batch)
        for start in range(0, len(chunks), step):
            window = chunks[start : start + step]
            collection.add(
                ids=[chunk.chunk_id for chunk in window],
                embeddings=vectors[start : start + step],
                documents=[chunk.embed_text for chunk in window],
                metadatas=[_metadata_of(chunk) for chunk in window],
            )
        return len(chunks)

    def count(self) -> int:
        try:
            return self._ensure_collection().count()
        except Exception:  # noqa: BLE001 - 索引不存在时按 0 处理
            return 0

    # -------------------------------------------------------------- 检索
    def search(self, query: str, top_k: int = 20, law_filter: tuple[str, ...] = ()) -> list[tuple[str, float]]:
        collection = self._ensure_collection()
        where: Any = {"law_id": {"$in": list(law_filter)}} if law_filter else None
        response = collection.query(
            query_embeddings=[self.embedder.embed_one(query)],
            n_results=top_k,
            where=where,
            include=["distances"],
        )
        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        # cosine 距离 → 相似度
        return [(chunk_id, 1.0 - float(distance)) for chunk_id, distance in zip(ids, distances)]


def _metadata_of(chunk) -> dict:
    """Chroma 的 metadata 只接受 str/int/float/bool，None 一律转空串。"""
    return {
        "law_id": chunk.law_id,
        "law_name": chunk.law_name,
        "version": chunk.version,
        "citation": chunk.citation,
        "chapter": chunk.chapter or "",
        "section": chunk.section or "",
        "article_no": chunk.article_no,
        "article_index": chunk.article_index,
        "part_index": chunk.part_index,
        "part_total": chunk.part_total,
        "parent_id": chunk.parent_id,
    }


class Indexer:
    """建索引阶段：父子块 → BM25 + 向量索引。

    Input : ChunkSet
    Output: IndexStats（并落盘 index/）
    """

    layer = "index"
    input_desc = "ChunkSet"
    output_desc = "IndexStats + index/bm25.json + index/chroma/"

    def __init__(
        self,
        *,
        bm25_path: Path | None = None,
        chroma_dir: Path | None = None,
        collection_name: str | None = None,
        embedder: EmbeddingClient | None = None,
        verbose: bool = True,
    ) -> None:
        self.bm25_path = Path(bm25_path or config.BM25_PATH)
        self.chroma_dir = Path(chroma_dir or config.CHROMA_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION
        self.embedder = embedder or EmbeddingClient()
        self.verbose = verbose
        self.notes: list[str] = []

    def build(self, chunk_set: ChunkSet, *, with_vector: bool = True) -> IndexStats:
        started = time.perf_counter()
        self.notes = []

        bm25 = BM25Index()
        bm25.build([(chunk.chunk_id, chunk.embed_text) for chunk in chunk_set.chunks])
        bm25.save(self.bm25_path)

        vector_count = 0
        embedding_model: str | None = None
        embedding_dim: int | None = None
        vector_enabled = False

        if with_vector and self.embedder.available:
            index = VectorIndex(
                persist_dir=self.chroma_dir,
                collection_name=self.collection_name,
                embedder=self.embedder,
            )
            try:
                vector_count = index.build(chunk_set.chunks)
                embedding_model = self.embedder.model_label
                embedding_dim = index.dim
                vector_enabled = True
            except Exception as exc:  # noqa: BLE001 - 向量不可用时降级为纯 BM25
                self.notes.append(f"向量索引构建失败，已降级为纯 BM25：{exc}")
        elif with_vector:
            self.notes.append(self.embedder.unavailable_reason)
        else:
            self.notes.append("已按参数跳过向量索引（--no-vector）")

        stats = IndexStats(
            collection=self.collection_name,
            chunk_count=len(chunk_set.chunks),
            bm25_docs=len(bm25.doc_ids),
            vector_count=vector_count,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            vector_enabled=vector_enabled,
            built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

        config.INDEX_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.INDEX_META_PATH.write_text(
            json.dumps({"stats": stats.to_dict(), "notes": self.notes, "chunks": chunk_set.stats},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if self.verbose:
            print(
                f"[index] BM25 {stats.bm25_docs} 块 | 向量 {stats.vector_count} 块"
                f"{'' if vector_enabled else '（未启用）'} | {stats.elapsed_ms:.0f}ms → {self.bm25_path.parent}"
            )
            for note in self.notes:
                print(f"[index] 提示：{note}")
        return stats

    # -------------------------------------------------------------- 读取
    def load_bm25(self) -> BM25Index:
        return BM25Index.load(self.bm25_path)

    def load_vector_index(self) -> VectorIndex:
        return VectorIndex(
            persist_dir=self.chroma_dir,
            collection_name=self.collection_name,
            embedder=self.embedder,
        )

    def load_stats(self) -> IndexStats | None:
        if not config.INDEX_META_PATH.exists():
            return None
        payload = json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))
        return IndexStats.from_dict(payload["stats"])


# ------------------------------------------------------------------ 调试入口
def main(argv: list[str] | None = None) -> int:
    """python -m tools.indexer [--no-vector] [--query 问题]"""
    from .chunker import ChunkStage

    args = list(sys.argv[1:] if argv is None else argv)
    chunk_set = ChunkStage(verbose=False).load()
    indexer = Indexer()
    indexer.build(chunk_set, with_vector="--no-vector" not in args)

    if "--query" in args:
        query = args[args.index("--query") + 1]
        bm25 = indexer.load_bm25()
        print(f"\nBM25 检索：{query}")
        for chunk_id, score in bm25.search(query, top_k=5):
            print(f"  {score:8.3f}  {chunk_id}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
