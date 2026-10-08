from __future__ import annotations

from eval.corpus import EvalCase
from eval.harness import CaseResult, EvalReport
from rag_contracts import config
from rag_contracts.domain.disk import Chunk, ParentChunk
from rag_contracts.domain.retrieval import Query
from rag_service.query.retriever import HybridRetriever

LAW_NAME = "中华人民共和国道路交通安全法"
CITATION = f"《{LAW_NAME}》"
QUESTION = "饮酒后驾驶营运机动车怎么处罚"
POOL = 20
CUTOFF = 6


def _case() -> EvalCase:
    return EvalCase(
        question=QUESTION,
        gold_ids=(f"{LAW_NAME}#第一条",),
        gold_citations=(f"{CITATION}第一条",),
        gold_laws=(LAW_NAME,),
    )


def _row(
    rank: int | None,
    *,
    in_dense: bool | None = True,
    in_bm25: bool | None = True,
) -> CaseResult:
    return CaseResult(
        case=_case(),
        hit_ids=(),
        hit_citations=(),
        rank=rank,
        used_vector=in_dense is not None,
        in_dense=in_dense,
        in_bm25=in_bm25,
    )


def _report(*rows: CaseResult) -> EvalReport:
    return EvalReport(
        results=rows,
        source="内存题集",
        top_k=CUTOFF,
        pool=POOL,
        used_vector=True,
        elapsed_ms=1000.0,
    )


def test_pool_recall_counts_the_gold_that_hit_at_6_leaves_out() -> None:
    report = _report(_row(1), _row(9, in_bm25=False), _row(None, in_dense=False, in_bm25=False))

    assert report.hit_at(1) == 1 / 3
    assert report.hit_at(6) == 1 / 3
    assert report.pool_hits() == 2
    assert report.pool_recall() == 2 / 3
    assert report.mrr() == 1 / 3
    assert all(report.hit_at(k) <= report.pool_recall() for k in (1, 3, 6))


def test_mrr_is_locked_at_the_top_k_cutoff() -> None:
    report = _report(_row(7), _row(20))

    assert report.pool_recall() == 1.0
    assert report.hit_at(6) == 0.0
    assert report.mrr() == 0.0, "池子放大后第 7 名往后的名次混进了 MRR，与已发布口径不可比"


def test_channel_recall_skips_a_channel_that_never_ran() -> None:
    report = _report(_row(1, in_dense=None, in_bm25=True), _row(None, in_dense=None, in_bm25=False))

    assert report.channel_recall("dense") == (0, 0)
    assert report.channel_recall("bm25") == (1, 2)


def test_the_report_always_carries_the_three_lines() -> None:
    text = _report(_row(1, in_bm25=True), _row(9, in_bm25=False)).render()

    assert f"池子 {POOL} 截到 top_k={CUTOFF}" in text
    assert "池子召回：" in text
    assert "单通道召回：BM25 1/2" in text
    assert "hit@6" in text and "MRR@6" in text


def test_dense_line_says_off_when_the_channel_never_ran() -> None:
    text = _report(_row(1, in_dense=None, in_bm25=True)).render()

    assert "稠密 未开" in text, "没开稠密那趟报的是 0%，读起来像「开了但一条没召回」"


def test_the_dict_separates_the_pool_from_the_cutoff() -> None:
    payload = _report(_row(1, in_bm25=True), _row(None, in_dense=False, in_bm25=False)).to_dict()

    assert payload["top_k"] == CUTOFF
    assert payload["pool"] == POOL
    assert payload["overall"]["pool_recall"] == 0.5
    assert payload["channel_recall"]["bm25"] == {"hits": 1, "cases": 2}
    assert payload["channel_recall"]["dense"] == {"hits": 1, "cases": 2}
    assert _report(_row(1, in_dense=None)).to_dict()["channel_recall"]["dense"] is None


def test_show_misses_counts_pool_only_hits_as_misses() -> None:
    text = _report(_row(1), _row(8)).render(show_misses=5)

    assert "未命中 1 题" in text, "捞进池子但排在 6 名之外的那题，在未命中清单里被漏掉了"


def test_an_empty_report_does_not_divide_by_zero() -> None:
    report = _report()

    assert report.hit_at(6) == 0.0
    assert report.mrr() == 0.0
    assert report.pool_recall() == 0.0
    assert report.channel_recall("bm25") == (0, 0)
    assert "（无）" in report.render()


ARTICLE_A = ParentChunk(
    parent_id=f"{LAW_NAME}#第一条",
    law_id="road_traffic_safety",
    law_name=LAW_NAME,
    version="2021",
    citation=CITATION,
    article_no="第一条",
    article_index=1,
    chapter=None,
    section=None,
    text="第一条的正文。",
)
ARTICLE_B = ParentChunk(
    parent_id=f"{LAW_NAME}#第二条",
    law_id="road_traffic_safety",
    law_name=LAW_NAME,
    version="2021",
    citation=CITATION,
    article_no="第二条",
    article_index=2,
    chapter=None,
    section=None,
    text="第二条的正文。",
)
ARTICLES = (ARTICLE_A, ARTICLE_B)


def _part(article: ParentChunk, part_index: int, chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_id=article.parent_id,
        law_id=article.law_id,
        law_name=article.law_name,
        version=article.version,
        citation=article.citation,
        article_no=article.article_no,
        article_index=article.article_index,
        part_index=part_index,
        part_total=2,
        text=f"{article.article_no}的第 {part_index + 1} 段。",
        embed_text=f"{article.article_no}的第 {part_index + 1} 段。",
    )


CHUNKS = {
    chunk.chunk_id: chunk
    for chunk in (_part(ARTICLE_A, 0, "a1"), _part(ARTICLE_A, 1, "a2"), _part(ARTICLE_B, 0, "b1"))
}


class _ChannelStore:
    collection = "测试集合"

    def __init__(self, *, fused, dense=(), sparse=()) -> None:
        self.fused = list(fused)
        self.dense = list(dense)
        self.sparse = list(sparse)
        self.calls: list[str] = []

    def has_dense_field(self) -> bool:
        return True

    def hybrid_search(self, **_kwargs):
        self.calls.append("hybrid")
        return list(self.fused)

    def dense_search(self, vector, *, limit, filter_expr=None):
        self.calls.append("dense")
        return list(self.dense)

    def sparse_search(self, text, *, limit, filter_expr=None):
        self.calls.append("sparse")
        return list(self.sparse)

    def law_filter(self, law_ids) -> str | None:
        return None


class _StubEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


def _retriever(store: _ChannelStore) -> HybridRetriever:
    return HybridRetriever(
        store=store,
        parents={article.parent_id: article for article in ARTICLES},
        chunks=CHUNKS,
        embedder=_StubEmbedder(),
        reranker=None,
        cfg=config.RetrieveConfig(top_k=CUTOFF, candidates=POOL, rrf_k=60, law_hint_boost=1.0),
    )


def test_a_channel_rank_is_the_articles_best_chunk_in_that_channel() -> None:
    store = _ChannelStore(
        fused=[("a2", 0.30), ("a1", 0.20)],
        dense=[("a1", 0.91)],
        sparse=[("a2", 7.5), ("a1", 6.0)],
    )

    result = _retriever(store).retrieve(
        Query(text=QUESTION, top_k=CUTOFF, candidates=POOL, channel_debug=True)
    )

    hit = next(item for item in result.articles if item.article is ARTICLE_A)
    assert hit.score == 0.30, "融合分被通道名次改写了"
    assert (hit.vector_rank, hit.vector_score) == (1, 0.91), (
        "稠密那路取的是组里融合分最高子块的名次 —— 它没进稠密，于是 gold 被误记成没被稠密召回"
    )
    assert (hit.bm25_rank, hit.bm25_score) == (1, 7.5)
    assert hit.hit_chunks == ("a1", "a2")


def test_channel_detail_only_runs_when_asked() -> None:
    store = _ChannelStore(fused=[("a1", 0.30)], dense=[("a1", 0.91)], sparse=[("a1", 7.5)])

    result = _retriever(store).retrieve(Query(text=QUESTION, top_k=CUTOFF, candidates=POOL))

    assert store.calls == ["hybrid"], "没要通道明细，却多跑了两遍单通道检索"
    assert result.articles[0].vector_rank is None


def test_equal_fusion_scores_are_still_broken_by_the_number_of_hit_chunks() -> None:
    store = _ChannelStore(fused=[("b1", 0.30), ("a2", 0.30), ("a1", 0.30)])

    result = _retriever(store).retrieve(Query(text=QUESTION, top_k=CUTOFF, candidates=POOL))

    assert [hit.article.article_no for hit in result.articles] == ["第一条", "第二条"], (
        "并列时「命中子块多」的那条没排在前面"
    )
    assert result.articles[0].hit_chunks == ("a1", "a2")
