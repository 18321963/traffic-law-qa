from __future__ import annotations

import pytest

from traffic_law_qa import config
from traffic_law_qa.contracts.disk import Chunk, ParentChunk
from traffic_law_qa.contracts.retrieval import Query
from traffic_law_qa.infra.embedding import LocalEmbedder
from traffic_law_qa.infra.reranking import LocalReranker
from traffic_law_qa.search.retriever import HybridRetriever

LAW_NAME = "中华人民共和国道路交通安全法"
CITATION = f"《{LAW_NAME}》"
QUESTION = "饮酒后驾驶营运机动车怎么处罚"
INDEXES = (1, 5, 9, 13, 20)
TOP_K = 3


def _article(index: int) -> ParentChunk:
    return ParentChunk(
        parent_id=f"{LAW_NAME}#第{index}条",
        law_id="road_traffic_safety",
        law_name=LAW_NAME,
        version="2021",
        citation=CITATION,
        article_no=f"第{index}条",
        article_index=index,
        chapter=None,
        section=None,
        text=f"第{index}条的正文：饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证。",
    )


ARTICLES = [_article(index) for index in INDEXES]
BY_TEXT = {article.text: article for article in ARTICLES}


def _chunk(article: ParentChunk) -> Chunk:
    return Chunk(
        chunk_id=f"c{article.article_index}",
        parent_id=article.parent_id,
        law_id=article.law_id,
        law_name=article.law_name,
        version=article.version,
        citation=article.citation,
        article_no=article.article_no,
        article_index=article.article_index,
        part_index=0,
        part_total=1,
        text=article.text,
        embed_text=article.text,
    )


class _Store:
    collection = "测试集合"

    def has_dense_field(self) -> bool:
        return False

    def hybrid_search(self, **_kwargs):
        return [(f"c{index}", 0.5 - index * 0.001) for index in INDEXES]

    def law_filter(self, law_ids) -> str | None:
        return None


class _StubReranker:
    model_label = "stub-reranker"

    def __init__(self, scores=None, *, reason: str = "", top_n: int = 0) -> None:
        self.cfg = config.RerankConfig(top_n=top_n)
        self._scores = scores or {}
        self._reason = reason
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def available(self) -> bool:
        return not self._reason

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    @property
    def top_n(self) -> int:
        return self.cfg.top_n

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, tuple(texts)))
        return [self._scores.get(text, 0.0) for text in texts]


class _BoomReranker(_StubReranker):
    def score(self, query: str, texts: list[str]) -> list[float]:
        raise RuntimeError("CUDA out of memory")


class _StubEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.0, 1.0]


def _retriever(reranker=None, embedder=None) -> HybridRetriever:
    return HybridRetriever(
        store=_Store(),
        parents={article.parent_id: article for article in ARTICLES},
        chunks={chunk.chunk_id: chunk for chunk in map(_chunk, ARTICLES)},
        embedder=embedder,
        reranker=reranker,
        cfg=config.RetrieveConfig(top_k=TOP_K, candidates=len(INDEXES), rrf_k=60, law_hint_boost=1.0),
    )


def _ask(reranker=None, top_k: int = TOP_K):
    return _retriever(reranker).retrieve(Query(text=QUESTION, top_k=top_k, candidates=len(INDEXES)))


def _nos(result) -> list[str]:
    return [hit.article.article_no for hit in result.articles]


def test_rerank_decides_the_order_and_becomes_the_only_score() -> None:
    scores = {article.text: float(article.article_index) for article in ARTICLES}

    result = _ask(_StubReranker(scores))

    assert _nos(result) == ["第20条", "第13条", "第9条"], "重排分没决定顺序（截断在重排之后）"
    assert [hit.score for hit in result.articles] == [20.0, 13.0, 9.0]
    assert "RRF" not in result.render()
    top = result.articles[0]
    assert f"1. {top.score:.4f}  {top.citation}" in result.render()


def test_rerank_scores_the_whole_pool_with_the_raw_question() -> None:
    reranker = _StubReranker({article.text: 0.0 for article in ARTICLES})

    _ask(reranker)

    query_text, texts = reranker.calls[0]
    assert len(texts) == len(ARTICLES), "融合池在重排前就被 top_k 截断了"
    assert query_text == QUESTION, "喂给交叉编码器的是改写后的关键词串，不是问句原话"


def test_top_n_leaves_the_tail_on_the_fusion_score() -> None:
    scores = {ARTICLES[0].text: 0.1, ARTICLES[1].text: 0.9}

    result = _ask(_StubReranker(scores, top_n=2), top_k=len(INDEXES))

    tail = [0.5 - index * 0.001 for index in (9, 13, 20)]

    assert _nos(result) == ["第5条", "第1条", "第9条", "第13条", "第20条"]
    assert [hit.score for hit in result.articles] == [0.9, 0.1, *tail]


def test_missing_reranker_keeps_the_fusion_order() -> None:
    baseline = _ask()

    for reranker in (_StubReranker(reason="本地没有重排权重 x，按融合顺序返回"), _BoomReranker()):
        result = _ask(reranker)
        assert _nos(result) == _nos(baseline)
        assert [hit.score for hit in result.articles] == [hit.score for hit in baseline.articles]

    assert any("未重排" in note for note in _ask(_StubReranker(reason="关了")).notes)
    assert any("重排失败" in note and "CUDA out of memory" in note for note in _ask(_BoomReranker()).notes)


def test_warm_pulls_both_models_in() -> None:
    reranker = _StubReranker()
    embedder = _StubEmbedder()

    elapsed = _retriever(reranker, embedder).warm(probe="预热")

    assert elapsed is not None
    assert embedder.queries == ["预热"]
    assert reranker.calls == [("预热", ("预热",))], "预热没顺带载重排模型，第一次问答要现等它"


def test_warm_skips_a_reranker_that_has_no_weights() -> None:
    reranker = _StubReranker(reason="本地没有重排权重 x")

    assert _retriever(reranker, _StubEmbedder()).warm() is not None
    assert reranker.calls == []


def test_warm_still_refuses_when_there_is_no_embedder() -> None:
    assert _retriever(_StubReranker()).warm() is None


def test_reranker_without_weights_refuses_instead_of_scoring() -> None:
    reranker = LocalReranker(config.RerankConfig(model="/no/such/dir"))

    assert reranker.available is False
    assert "本地没有重排权重" in reranker.unavailable_reason
    with pytest.raises(RuntimeError):
        reranker.score(QUESTION, ["正文"])


def test_embedding_client_without_weights_refuses_instead_of_loading_torch() -> None:
    client = LocalEmbedder(config.EmbedConfig(model="/no/such/dir"))

    assert client.available is False
    assert "本地没有权重目录" in client.unavailable_reason
    with pytest.raises(RuntimeError):
        client.embed(["正文"])


def test_weights_ready_accepts_a_local_dir_or_a_repo_id(tmp_path) -> None:
    weights = tmp_path / "bge-reranker-v2-m3"
    weights.mkdir()

    assert config.RerankConfig(model=str(weights)).ready is True
    assert config.RerankConfig(enabled=False, model=str(weights)).ready is False
    assert config.RerankConfig(model="BAAI/bge-reranker-v2-m3").ready is True
    assert config.RerankConfig(model=str(tmp_path / "还没下")).ready is False
    assert config.RerankConfig(model="").ready is False


def test_embedding_label_names_the_weights_dir_without_loading_it() -> None:
    assert LocalEmbedder(config.EmbedConfig(model="/x/y/bge-m3")).model_label == "bge-m3"
    assert LocalEmbedder(config.EmbedConfig(model="/x/y/bge-reranker-v2-m3")).model_label == (
        "bge-reranker-v2-m3"
    )
