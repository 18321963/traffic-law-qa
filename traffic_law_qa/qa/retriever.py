from __future__ import annotations

import time
from collections import defaultdict

from .. import config
from ..contracts import (
    Chunk,
    ChunkSet,
    CorpusStats,
    ParentChunk,
    Query,
    RetrievalResult,
    RetrievedArticle,
)
from ..kb.milvus_store import MilvusStore
from .query_rewriter import QueryRewriter


class HybridRetriever:

    layer = "retrieve"
    input_desc = "Query（内部先经 QueryRewriter 改写）"
    output_desc = "RetrievalResult"

    def __init__(
        self,
        *,
        store: MilvusStore,
        parents: dict[str, ParentChunk],
        chunks: dict[str, Chunk],
        embedder=None,
        cfg: config.RetrieveConfig | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._store = store
        self.parents = parents
        self._chunks = chunks
        self.embedder = embedder
        self.cfg = cfg or config.retrieve_config()
        self._rewriter = rewriter or QueryRewriter(
            law_names=tuple(dict.fromkeys(p.law_name for p in parents.values()))
        )

    @classmethod
    def load(
        cls,
        *,
        with_vector: bool = True,
        chunk_set: ChunkSet | None = None,
        cfg: config.RetrieveConfig | None = None,
        store: MilvusStore | None = None,
        embedder=None,
    ) -> "HybridRetriever":
        from ..kb.chunker import ChunkStage
        from ..services.embedding import EmbeddingClient

        chunk_set = chunk_set or ChunkStage(verbose=False).load()
        store = store or MilvusStore(verbose=False)
        embedder = embedder if embedder is not None else EmbeddingClient()

        return cls(
            store=store,
            parents=chunk_set.parent_map(),
            chunks={chunk.chunk_id: chunk for chunk in chunk_set.chunks},
            embedder=embedder if with_vector else None,
            cfg=cfg,
        )

    def warm(self, *, probe: str = "预热") -> float | None:
        if self.embedder is None:
            return None
        started = time.perf_counter()
        try:
            self.embedder.embed_query(probe)
        except Exception:  # noqa: BLE001
            return None
        return (time.perf_counter() - started) * 1000

    def expand(self, text: str) -> str:
        return self._rewriter.rewrite(Query(text=text)).expanded

    def stats(self) -> CorpusStats:
        try:
            dense: bool | None = self._store.has_dense_field()
        except Exception:  # noqa: BLE001
            dense = None
        return CorpusStats(
            articles=len(self.parents),
            chunks=len(self._chunks),
            dense=dense,
            collection=self._store.collection,
        )

    def retrieve(self, query: Query) -> RetrievalResult:
        query = query.normalized()
        started = time.perf_counter()
        notes: list[str] = []

        rewritten = self._rewriter.rewrite(query)
        alias_note = self._rewriter.describe_aliases(rewritten)
        if alias_note:
            notes.append(alias_note)
        hints = set(rewritten.law_hints)
        if hints:
            notes.append(
                f"法名线索：{'、'.join(rewritten.law_hints)}（命中法规的分数 ×{self.cfg.law_hint_boost}）"
            )
        search_text = rewritten.expanded
        filter_expr = MilvusStore.law_filter_expr(query.law_filter)

        dense_vector: list[float] | None = None
        if query.use_vector:
            if not self._store.has_dense_field():
                notes.append("集合未建稠密向量字段（纯 BM25 模式），本次只用 BM25 召回")
            elif self.embedder is None:
                notes.append("本次禁用了稠密向量，只用 BM25 召回")
            else:
                try:
                    dense_vector = self.embedder.embed_query(search_text)
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"查询向量化失败，本次只用 BM25：{exc}")

        if not query.use_bm25 and dense_vector is not None:
            notes.append("已禁用 BM25 通道，本次仅用稠密向量")
            hits = self._store.dense_search(
                dense_vector, limit=query.candidates, filter_expr=filter_expr
            )
        else:
            hits = self._store.hybrid_search(
                query_text=search_text,
                dense_vector=dense_vector,
                limit=query.candidates,
                candidates=query.candidates,
                filter_expr=filter_expr,
                rrf_k=self.cfg.rrf_k,
            )

        ranks, raw_scores = self._channel_detail(query, search_text, dense_vector, filter_expr, notes)
        articles = self._group_to_articles(hits, ranks, raw_scores, query, hints)

        return RetrievalResult(
            query=query.text,
            articles=tuple(articles),
            used_vector=dense_vector is not None,
            used_bm25=query.use_bm25,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            notes=tuple(notes),
        )

    def search(self, text: str, top_k: int | None = None, **kwargs) -> RetrievalResult:
        return self.retrieve(
            Query(
                text=text,
                top_k=top_k or self.cfg.top_k,
                candidates=self.cfg.candidates,
                **kwargs,
            )
        )

    def _channel_detail(
        self,
        query: Query,
        search_text: str,
        dense_vector: list[float] | None,
        filter_expr: str | None,
        notes: list[str],
    ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        raw_scores: dict[str, dict[str, float]] = defaultdict(dict)
        if not query.channel_debug:
            return ranks, raw_scores

        channels: list[tuple[str, list[tuple[str, float]]]] = []
        if dense_vector is not None:
            channels.append(
                ("dense", self._store.dense_search(dense_vector, limit=query.candidates, filter_expr=filter_expr))
            )
        if query.use_bm25:
            channels.append(
                ("bm25", self._store.sparse_search(search_text, limit=query.candidates, filter_expr=filter_expr))
            )
        if not channels:
            notes.append("通道明细不可用：既没有稠密通道也没有 BM25 通道")

        for name, hits in channels:
            for rank, (chunk_id, score) in enumerate(hits, start=1):
                ranks[chunk_id][name] = rank
                raw_scores[chunk_id][name] = score
        return ranks, raw_scores

    def _group_to_articles(
        self,
        hits: list[tuple[str, float]],
        ranks: dict[str, dict[str, int]],
        raw_scores: dict[str, dict[str, float]],
        query: Query,
        hints: set[str],
    ) -> list[RetrievedArticle]:
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for chunk_id, score in hits:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            grouped[chunk.parent_id].append((chunk_id, score))

        articles: list[RetrievedArticle] = []
        for parent_id, group in grouped.items():
            parent = self.parents.get(parent_id)
            if parent is None:
                continue
            if query.law_filter and parent.law_id not in query.law_filter:
                continue

            best_id, best_score = max(group, key=lambda item: item[1])
            hint = next((h for h in hints if h in parent.law_name), None)
            if hint:
                best_score *= self.cfg.law_hint_boost

            articles.append(
                RetrievedArticle(
                    article=parent,
                    score=best_score,
                    vector_rank=ranks.get(best_id, {}).get("dense"),
                    bm25_rank=ranks.get(best_id, {}).get("bm25"),
                    vector_score=raw_scores.get(best_id, {}).get("dense"),
                    bm25_score=raw_scores.get(best_id, {}).get("bm25"),
                    hit_chunks=tuple(sorted(chunk_id for chunk_id, _ in group)),
                    law_hint=hint,
                )
            )

        articles.sort(key=lambda item: (item.score, len(item.hit_chunks)), reverse=True)
        return articles[: query.top_k]
