"""Layer 5 · retrieve：Query → RetrievalResult（管道对外的检索工具）。

    HybridRetriever.retrieve : Query → RetrievalResult

Milvus 版本做了三件事，其余保持原样：
1. **融合下推到服务端**：`hybrid_search` + `RRFRanker(k)` 就是我们要的 RRF，
   客户端不再自己拼排名；稠密（COSINE）+ BM25 稀疏双路召回。
2. **查询改写保留**：口语词 → 法条用语（醉驾→醉酒驾驶）、法名线索加成，
   这两处是实测纠正过召回的关键，与存储引擎无关。
3. **父块回灌保留**：在款级子块上检索，命中后按 parent_id 聚合回整条法条。

`Query.channel_debug=True` 时会额外跑两次单通道检索，用来看某条法条是被
向量捞到的还是被 BM25 捞到的（不开启则只有融合分）。
"""

from __future__ import annotations

import time
from collections import defaultdict

from . import config
from .contracts import (
    Chunk,
    ChunkSet,
    ParentChunk,
    Query,
    RetrievalResult,
    RetrievedArticle,
)
from .milvus_store import MilvusStore
from .query_rewriter import QueryRewriter


class HybridRetriever:
    """混合检索器（Milvus 双路召回 + 父块回灌 + 查询改写）。

    Input : Query
    Output: RetrievalResult
    """

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
        self.store = store
        self.parents = parents
        self.chunks = chunks
        self.embedder = embedder
        self.cfg = cfg or config.retrieve_config()
        self.rewriter = rewriter or QueryRewriter(
            law_names=tuple(dict.fromkeys(p.law_name for p in parents.values()))
        )

    # -------------------------------------------------------------- 装配
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
        """从 Milvus + 本地块文件装配检索器。"""
        from .chunker import ChunkStage
        from .indexer import EmbeddingClient

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

    # -------------------------------------------------------------- 主接口
    def retrieve(self, query: Query) -> RetrievalResult:
        query = query.normalized()
        started = time.perf_counter()
        notes: list[str] = []

        rewritten = self.rewriter.rewrite(query)
        alias_note = self.rewriter.describe_aliases(rewritten)
        if alias_note:
            notes.append(alias_note)
        hints = set(rewritten.law_hints)
        if hints:
            notes.append(
                f"法名线索：{'、'.join(rewritten.law_hints)}（命中法规的分数 ×{self.cfg.law_hint_boost}）"
            )
        search_text = rewritten.expanded
        filter_expr = MilvusStore.law_filter_expr(query.law_filter)

        # 稠密通道：没配 embedding 或集合里没建稠密字段时自动只走 BM25
        dense_vector: list[float] | None = None
        if query.use_vector:
            if not self.store.has_dense_field():
                notes.append("集合未建稠密向量字段（纯 BM25 模式），本次只用 BM25 召回")
            elif self.embedder is None:
                notes.append("本次禁用了稠密向量，只用 BM25 召回")
            else:
                try:
                    dense_vector = self.embedder.embed_one(search_text)
                except Exception as exc:  # noqa: BLE001 - 向量端点是外部依赖
                    notes.append(f"查询向量化失败，本次只用 BM25：{exc}")

        if not query.use_bm25 and dense_vector is not None:
            notes.append("已禁用 BM25 通道，本次仅用稠密向量")
            hits = self.store.dense_search(
                dense_vector, limit=query.candidates, filter_expr=filter_expr
            )
        else:
            hits = self.store.hybrid_search(
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
        """文本直入的检索接口（供管道与外部调用）。"""
        return self.retrieve(
            Query(
                text=text,
                top_k=top_k or self.cfg.top_k,
                candidates=self.cfg.candidates,
                **kwargs,
            )
        )

    # -------------------------------------------------------------- 通道明细
    def _channel_detail(
        self,
        query: Query,
        search_text: str,
        dense_vector: list[float] | None,
        filter_expr: str | None,
        notes: list[str],
    ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
        """可选：额外跑两次单通道检索，拿到每条命中在两路里的排名与原始分。"""
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        raw_scores: dict[str, dict[str, float]] = defaultdict(dict)
        if not query.channel_debug:
            return ranks, raw_scores

        channels: list[tuple[str, list[tuple[str, float]]]] = []
        if dense_vector is not None:
            channels.append(
                ("dense", self.store.dense_search(dense_vector, limit=query.candidates, filter_expr=filter_expr))
            )
        if query.use_bm25:
            channels.append(
                ("bm25", self.store.sparse_search(search_text, limit=query.candidates, filter_expr=filter_expr))
            )
        if not channels:
            notes.append("通道明细不可用：既没有稠密通道也没有 BM25 通道")

        for name, hits in channels:
            for rank, (chunk_id, score) in enumerate(hits, start=1):
                ranks[chunk_id][name] = rank
                raw_scores[chunk_id][name] = score
        return ranks, raw_scores

    # -------------------------------------------------------------- 父块聚合
    def _group_to_articles(
        self,
        hits: list[tuple[str, float]],
        ranks: dict[str, dict[str, int]],
        raw_scores: dict[str, dict[str, float]],
        query: Query,
        hints: set[str],
    ) -> list[RetrievedArticle]:
        """按 parent_id 聚合子块分数，取每条法条的最佳命中作为它的分数。

        额外做法名加成：查询里出现"深圳""智能网联汽车"这类法名片段时，
        对应法规的条文会被加权 —— 否则问深圳的事很容易被国家法律的一般条款盖过。
        """
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for chunk_id, score in hits:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:  # 集合与本地块文件不同步
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
