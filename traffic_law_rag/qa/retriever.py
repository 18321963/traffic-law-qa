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

**这一层是 `LegalRAG` 的实现细节，不是门面。** `store` / `chunks` / `rewriter`
都是私有的：上层要什么能力，由 `LegalRAG` 以方法的形式给出（`expand` 如此、
`stats` 如此），不要伸手进来取零件。构造参数仍然叫 `store=` / `chunks=` ——
那是**注入点**（测试要塞替身），与「构造之后谁能看见」是两回事。
`parents` 与 `embedder` 是例外，保持公开：前者是语料本身、门面会原样再暴露，
后者有测试在断言预热前后是同一个对象。
"""

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
        self._store = store
        self.parents = parents
        self._chunks = chunks
        self.embedder = embedder
        self.cfg = cfg or config.retrieve_config()
        self._rewriter = rewriter or QueryRewriter(
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
        from ..kb.chunker import ChunkStage
        from ..kb.indexer import EmbeddingClient

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
        """预热稠密通道，返回耗时毫秒；没得预热或失败时返回 None。

        **为什么需要它**：embedding 端点的首次调用要建 TCP + TLS 连接，实测
        1.5~3 秒，之后的调用只要 ~150ms（换没见过的文本也一样，所以不是缓存）。
        这个开销被惰性付掉时，是算在**第一个调用者**头上的 —— 放进 HTTP 服务里
        就是「第一个用户特别慢」，而 `load()` 本身只要 39ms。预热的唯一作用
        是把它从用户请求挪到启动阶段。命令行一次性调用不需要它：早晚都要付。

        **失败不是错**：检索层本来就有「向量端点不可用就退回 BM25」的降级路径
        （见 `retrieve()` 里 embed_query 的 except）。预热同理 —— 这里吞掉异常，
        真正的错误会在第一次真实检索时带着上下文报出来。
        """
        if self.embedder is None:
            return None
        started = time.perf_counter()
        try:
            self.embedder.embed_query(probe)
        except Exception:  # noqa: BLE001 - 外部依赖不可用；降级路径在 retrieve() 里
            return None
        return (time.perf_counter() - started) * 1000

    # -------------------------------------------------------------- 门面支撑
    def expand(self, text: str) -> str:
        """复算改写器的口语对齐，返回**真正会被拿去检索的那串词**。

        **为什么需要它**：摘要在法条原文里开一个窗口，窗口按什么词定位决定了
        模型看得见什么。原话里的口语词在法条里压根不出现 —— 实测「深圳开车玩手机」
        与第十三条的唯一字面重叠只有「罚款」二字，窗口于是停在开头，而真正说明
        这条与问题有关的那一项「（七）手动操作移动电话、电子设备」落在窗口之外。
        用对齐后的词（…拨打接听手持电话、移动电话、电子设备…）才能把窗口移到第七项。

        确实是把刚算过的东西又算了一遍。可以接受，是因为 `QueryRewriter.rewrite`
        是**纯函数**（查表 + 最长公共子串，无状态、无随机），复算结果与检索时那次
        逐字相同。**若将来改写器变成有状态或依赖外部输入，这里必须改成由
        `retrieve()` 回传**，不能继续复算。

        **为什么放在这一层**：它就是改写器的产物，出去也只有这一个用途。
        从前由 `agent_tools.expand_query` 用 `getattr(retriever, "rewriter")` 摸进来取，
        于是「改写器改名」这种事不会报错、只会让窗口**静默**退回到修好之前的位置。
        放在这里，改名的代价是一次 `AttributeError`，不是一次性能静默劣化。
        """
        return self._rewriter.rewrite(Query(text=text)).expanded

    def stats(self) -> CorpusStats:
        """语料规模与通道状态。**唯一一处**吞 Milvus 连接的异常。

        从前 `rag.describe()` 与 `agent.describe()` 各写一遍同样的 try/except，
        于是「连不上时显示什么」有两个真源。

        `collection` 只读配置、不查库，所以放在 try 外面：连不上 Milvus 时
        它照样是准的，不该跟着一起变成「未知」。

        **不缓存探测结果**：缓存 `None` 就等于「Milvus 没起过一次，这一辈子
        都说它没起」，而这条路径常在服务启动时被调用来做健康检查。
        """
        try:
            dense: bool | None = self._store.has_dense_field()
        except Exception:  # noqa: BLE001 - 外部依赖不可用，状态未知不是错误
            dense = None
        return CorpusStats(
            articles=len(self.parents),
            chunks=len(self._chunks),
            dense=dense,
            collection=self._store.collection,
        )

    # -------------------------------------------------------------- 主接口
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

        # 稠密通道：没配 embedding 或集合里没建稠密字段时自动只走 BM25
        dense_vector: list[float] | None = None
        if query.use_vector:
            if not self._store.has_dense_field():
                notes.append("集合未建稠密向量字段（纯 BM25 模式），本次只用 BM25 召回")
            elif self.embedder is None:
                notes.append("本次禁用了稠密向量，只用 BM25 召回")
            else:
                try:
                    # 查询前缀只加在**这一条路径**上：search_text 在下面还以
                    # query_text=search_text 喂给 BM25，往它身上拼前缀会把关键词
                    # 通道的查询串一起污染。embed_query 内部处理，两路因此各得其宜。
                    dense_vector = self.embedder.embed_query(search_text)
                except Exception as exc:  # noqa: BLE001 - 向量端点是外部依赖
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
            chunk = self._chunks.get(chunk_id)
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
