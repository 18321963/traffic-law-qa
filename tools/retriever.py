"""Layer 5 · retrieve：Query → RetrievalResult（这就是"RAG 里的 R"，也是管道对外的检索工具）。

    HybridRetriever.retrieve : Query → RetrievalResult

两个关键设计：
1. **RRF 融合**（Reciprocal Rank Fusion）：向量分数与 BM25 分数不同量纲，不能直接相加；
   RRF 只用排名，天然免疫量纲问题，且任一条通道缺失都能继续工作。
2. **父块回灌**：子在款级子块上检索，命中后按 parent_id 聚合，返回整条法条给 LLM。
   否则模型只会看到"（一）兜售物品、散发广告或者乞讨；"这种半句。
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
from .indexer import BM25Index, VectorIndex
from .query_rewriter import QueryRewriter


class HybridRetriever:
    """混合检索器（查询改写 + BM25 + 向量，RRF 融合 + 父块回灌）。

    Input : Query
    Output: RetrievalResult
    """

    layer = "retrieve"
    input_desc = "Query（内部先经 QueryRewriter 改写）"
    output_desc = "RetrievalResult"

    def __init__(
        self,
        *,
        bm25: BM25Index,
        parents: dict[str, ParentChunk],
        chunks: dict[str, Chunk],
        vector: VectorIndex | None = None,
        cfg: config.RetrieveConfig | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.bm25 = bm25
        self.parents = parents
        self.chunks = chunks
        self.vector = vector
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
    ) -> "HybridRetriever":
        """从磁盘装配检索器（索引 + 块）。"""
        from .chunker import ChunkStage
        from .indexer import Indexer

        chunk_set = chunk_set or ChunkStage(verbose=False).load()
        indexer = Indexer(verbose=False)
        bm25 = indexer.load_bm25()

        vector = None
        if with_vector:
            candidate = indexer.load_vector_index()
            if candidate.available and candidate.count() > 0:
                vector = candidate

        return cls(
            bm25=bm25,
            parents=chunk_set.parent_map(),
            chunks={chunk.chunk_id: chunk for chunk in chunk_set.chunks},
            vector=vector,
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

        use_vector = bool(query.use_vector and self.vector is not None)
        if query.use_vector and self.vector is None:
            notes.append("向量索引未启用，本次仅使用 BM25 召回")
        use_bm25 = bool(query.use_bm25)

        vector_hits: list[tuple[str, float]] = []
        bm25_hits: list[tuple[str, float]] = []

        if use_vector and self.vector is not None:
            vector_hits = self.vector.search(search_text, query.candidates, query.law_filter)
        if use_bm25:
            bm25_hits = self.bm25.search(search_text, query.candidates)

        fused, ranks, raw_scores = self._fuse(vector_hits, bm25_hits)
        articles = self._group_to_articles(fused, ranks, raw_scores, query, hints)

        return RetrievalResult(
            query=query.text,
            articles=tuple(articles),
            used_vector=use_vector,
            used_bm25=use_bm25,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            notes=tuple(notes),
        )

    # -------------------------------------------------------------- 融合
    def _fuse(
        self,
        vector_hits: list[tuple[str, float]],
        bm25_hits: list[tuple[str, float]],
    ) -> tuple[dict[str, float], dict[str, dict[str, int]], dict[str, dict[str, float]]]:
        """RRF：score = Σ 通道权重 / (k + 该通道内排名)。"""
        fused: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        raw_scores: dict[str, dict[str, float]] = defaultdict(dict)

        channels = (
            ("vector", vector_hits, self.cfg.vector_weight),
            ("bm25", bm25_hits, self.cfg.bm25_weight),
        )
        for name, hits, weight in channels:
            for rank, (chunk_id, score) in enumerate(hits, start=1):
                if chunk_id not in self.chunks:  # 索引与块文件不同步
                    continue
                fused[chunk_id] += weight / (self.cfg.rrf_k + rank)
                ranks[chunk_id][name] = rank
                raw_scores[chunk_id][name] = score
        return dict(fused), ranks, raw_scores

    def _group_to_articles(
        self,
        fused: dict[str, float],
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
        for chunk_id, score in fused.items():
            grouped[self.chunks[chunk_id].parent_id].append((chunk_id, score))

        articles: list[RetrievedArticle] = []
        for parent_id, hits in grouped.items():
            parent = self.parents.get(parent_id)
            if parent is None:
                continue
            if query.law_filter and parent.law_id not in query.law_filter:
                continue

            best_id, best_score = max(hits, key=lambda item: item[1])
            hint = next((h for h in hints if h in parent.law_name), None)
            if hint:
                best_score *= self.cfg.law_hint_boost
            hit_ids = tuple(sorted(chunk_id for chunk_id, _ in hits))
            articles.append(
                RetrievedArticle(
                    article=parent,
                    score=best_score,
                    vector_rank=ranks[best_id].get("vector"),
                    bm25_rank=ranks[best_id].get("bm25"),
                    vector_score=raw_scores[best_id].get("vector"),
                    bm25_score=raw_scores[best_id].get("bm25"),
                    hit_chunks=hit_ids,
                    law_hint=hint,
                )
            )

        articles.sort(key=lambda item: (item.score, len(item.hit_chunks)), reverse=True)
        return articles[: query.top_k]

    # -------------------------------------------------------------- 便捷入口
    def search(self, text: str, top_k: int | None = None, **kwargs) -> RetrievalResult:
        """文本直入的检索接口（供管道与外部调用）。"""
        return self.retrieve(
            Query(text=text, top_k=top_k or self.cfg.top_k, candidates=self.cfg.candidates, **kwargs)
        )
