"""`LegalRAG` 门面：参数透传、默认值、自述。

**这个文件之前不存在** —— `LegalRAG` 是上层唯一的检索入口，却一条测试都没有。
于是「门面把参数透错了」这类错误只能靠人肉看代码发现，而它恰恰是最不值得靠人眼
盯的一类改动：`law_filter` 默认写成 `None` 就是一个必炸的 `TypeError`，
却没有任何一条既有测试会红。

全部离线：替身检索器只记录收到了什么，不连 Milvus、不发网络请求。
"""

from __future__ import annotations

import pytest

from traffic_law_qa import config
from traffic_law_qa.contracts import CorpusStats, Question, RetrievalResult
from traffic_law_qa.qa.generator import AnswerGenerator
from traffic_law_qa.qa.rag import LegalRAG

FAKE_LLM_CFG = config.LLMConfig(base_url="http://fake", api_key="fake", model="fake-model")

# 夹具的实例条数，**刻意不等于 `RAG_TOP_K` 的默认值**（理由见下面 `rag` 夹具）。
INSTANCE_TOP_K = 4


class RecordingRetriever:
    """记录每一次调用收到的参数；不实现检索，也不提供任何私有属性。"""

    def __init__(self, *, dense: bool | None = True, boom: bool = False) -> None:
        self.calls: list[dict] = []
        self.parents = {"p1": object(), "p2": object()}
        self._dense = dense
        self._boom = boom
        self.stats_calls = 0

    def search(self, text: str, top_k: int | None = None, **kwargs) -> RetrievalResult:
        self.calls.append({"text": text, "top_k": top_k, **kwargs})
        return RetrievalResult(query=text, articles=(), used_vector=False, used_bm25=True, elapsed_ms=1.0)

    def expand(self, text: str) -> str:
        return f"<展开>{text}"

    def stats(self) -> CorpusStats:
        self.stats_calls += 1
        return CorpusStats(articles=2, chunks=3, dense=self._dense, collection="c")


@pytest.fixture
def generator() -> AnswerGenerator:
    return AnswerGenerator(FAKE_LLM_CFG)


@pytest.fixture
def retriever() -> RecordingRetriever:
    return RecordingRetriever()


@pytest.fixture
def rag(retriever, generator) -> LegalRAG:
    """实例条数**刻意不等于配置默认值** —— 两个数撞在一起，`top_k or` 少写前半截也看不出来。

    `LegalRAG.__init__` 那行有两种写法：`top_k or config.retrieve_config().top_k` 与
    `config.retrieve_config().top_k`。它们只在**传了 `top_k` 且它不等于配置默认值**时
    才分得开。本文件第一版传的是 6，而 `RAG_TOP_K` 的默认值也是 6 —— 「实例默认值」与
    「配置默认值」逐位相同，于是删掉 `top_k or` 之后整个文件照样全绿，而线上
    `LegalRAG.load(top_k=3)` / `--top-k 3` 会静默失效，检索仍按 6 条走。

    `test_top_k_falls_back_to_the_instance_default` 断言里的那个 4 就是它，
    **不是**随手挑的数。下面这句守卫钉住这个差：谁把常量改回配置默认值，本文件当场红。
    """
    assert INSTANCE_TOP_K != config.retrieve_config().top_k, "实例条数又和配置默认值撞了"
    return LegalRAG(retriever, generator, top_k=INSTANCE_TOP_K)


# ================================================================== 参数透传
def test_search_forwards_top_k_and_channel_debug(rag, retriever):
    rag.search("醉驾怎么处罚", top_k=3, channel_debug=True)
    assert retriever.calls == [
        {"text": "醉驾怎么处罚", "top_k": 3, "channel_debug": True, "law_filter": ()}
    ]


def test_ask_forwards_channel_debug(rag, retriever):
    """`ask()` 也得把 `channel_debug` 透下去 —— 上面那条只钉住了 `search()`。

    `ask()` 是**另一个入口**，`api.py:242` 的 debug 模式走的正是它（`rag.py:104`
    写着这段来历：这个参数不在 `ask` 上的时候，api 只能自己 `search()` 之后再伸手调
    生成器，一条绕过门面的路）。门面收下参数却不往下传时，`search` 那条测试照样绿 ——
    参数在门面上「能收」、在门面里「没用」，而 `?debug=1` 从此静默返回不带分通道明细的
    结果，调用方拿到的字段还在（`channel_debug` 只控制要不要多跑那两次单通道检索），
    只是全空 —— 没有异常，只有一份看起来正常、实际什么都没说的响应。
    """
    rag.ask("醉驾怎么处罚", channel_debug=True)
    assert retriever.calls[0]["channel_debug"] is True


def test_top_k_falls_back_to_the_instance_default(rag, retriever):
    """不传 `top_k` 时用**实例**那个（构造时给的），不是配置默认值 —— 两者刻意不同值。"""
    rag.search("醉驾怎么处罚")
    assert retriever.calls[0]["top_k"] == INSTANCE_TOP_K


def test_default_law_filter_is_an_empty_tuple_not_none(rag, retriever):
    """`law_filter` 的默认值必须是 `()`。

    写 `None` 是一个**必炸**的默认值：`MilvusStore.law_filter_expr` 直接迭代入参
    （`for law_id in law_ids`），`Query.law_filter` 本身也是 `()`。
    这里除了断言透传值，还把它真的喂给那个函数 —— 只有真跑一遍才算验证过。
    """
    from traffic_law_qa.kb.milvus_store import MilvusStore

    rag.search("醉驾怎么处罚")
    forwarded = retriever.calls[0]["law_filter"]
    assert forwarded == ()
    assert MilvusStore.law_filter_expr(forwarded) is None, "空过滤器应当等于「不过滤」"


def test_law_filter_reaches_the_retriever(rag, retriever):
    rag.search("深圳的事", law_filter=("sz_traffic_penalty_regulation",))
    assert retriever.calls[0]["law_filter"] == ("sz_traffic_penalty_regulation",)


def test_ask_passes_the_whole_question_object_through(rag, retriever):
    """`ask` 用 `search` 的结果去生成，而 `search` 收到的是问题原文与生效的 top_k。"""
    answer = rag.ask(Question(text="醉驾怎么处罚", top_k=3))
    assert retriever.calls[0] == {
        "text": "醉驾怎么处罚",
        "top_k": 3,
        "channel_debug": False,
        "law_filter": (),
    }
    assert answer.question == "醉驾怎么处罚"


def test_ask_uses_the_question_top_k_over_the_instance_default(rag, retriever):
    """`Question.top_k` 优先于实例默认值，实例默认值优先于配置默认值。

    末一档（实例 vs 配置）**在本文件里没有任何一条能测**：只有构造时不给 `top_k`
    才走得到它，而那时两者本来就该是同一个数。它由 `LegalRAG.load()` 那条路覆盖 ——
    `test_cli_args.py` 钉的是 `--top-k` 有没有被传进去，不是这行 `or`。
    """
    rag.ask("醉驾怎么处罚")
    assert retriever.calls[0]["top_k"] == INSTANCE_TOP_K
    rag.ask(Question(text="醉驾怎么处罚", top_k=2))
    assert retriever.calls[1]["top_k"] == 2


# ================================================================== 转发
def test_expand_delegates_to_the_retriever(rag):
    """摘要窗口靠它定位 —— 门面必须是转发，不是另算一遍。"""
    assert rag.expand("深圳开车玩手机") == "<展开>深圳开车玩手机"


def test_parents_is_the_retrievers_corpus(rag, retriever):
    assert rag.parents is retriever.parents


# ================================================================== stats 与自述
def test_stats_reports_the_three_states(generator):
    assert LegalRAG(RecordingRetriever(), generator).stats().dense is True
    assert LegalRAG(RecordingRetriever(dense=False), generator).stats().dense is False
    assert LegalRAG(RecordingRetriever(dense=None), generator).stats().dense is None


def test_describe_says_unknown_when_the_store_is_unreachable(generator):
    """连不上 Milvus 时只说「未知」，不抛异常 —— 而且 search 照样能用。"""
    rag = LegalRAG(RecordingRetriever(dense=None), generator)
    text = rag.describe()
    assert "未知（Milvus 未连接）" in text
    assert "2 条法条 / 3 个子块" in text
    assert "集合 c" in text
    rag.search("醉驾怎么处罚")   # 自述失败不该让检索也坏掉


def test_describe_renders_the_enabled_and_disabled_states(generator):
    assert "已启用" in LegalRAG(RecordingRetriever(), generator).describe()
    assert "未启用（仅 BM25）" in LegalRAG(RecordingRetriever(dense=False), generator).describe()


def test_stats_does_not_cache_the_unknown_state(generator):
    """探测结果**不缓存**。

    缓存 `None` 就等于「Milvus 没起过一次，这一辈子都说它没起」，
    而这条路径常在服务启动时被健康检查调用 —— 那之后 Milvus 起来了也翻不了身。
    """
    retriever = RecordingRetriever(dense=None)
    rag = LegalRAG(retriever, generator)
    assert rag.stats().dense is None
    retriever._dense = True
    assert rag.stats().dense is True
    assert retriever.stats_calls == 2, "两次调用都该真的去探"
