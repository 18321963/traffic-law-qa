"""generate 层：强制引用式生成的两条路径（同步 / 流式）。

这个文件里最要紧的一条是 **`test_流式与非流式发同一份提示词`**：
流式是后加的，若它自己另写一份 messages 构造，同一个问题在 `/qa` 与
`/qa/stream` 下会给出不一样的答案 —— 而这种偏差只在对比两个入口时才看得出来。

用假 LLM 客户端驱动，不连任何端点。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traffic_law_qa.config import LLMConfig
from traffic_law_qa.contracts import Question, RetrievalResult, RetrievedArticle
from traffic_law_qa.qa.generator import (
    EMPTY_RETRIEVAL_ANSWER,
    SYSTEM_PROMPT,
    UNAVAILABLE_ANSWER,
    AnswerGenerator,
)


# ------------------------------------------------------------------ 假客户端
def _chunk(text: str | None = None, *, usage: tuple[int, int, int] | None = None):
    choices = [] if text is None else [SimpleNamespace(delta=SimpleNamespace(content=text))]
    usage_obj = None if usage is None else SimpleNamespace(
        prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=usage[2]
    )
    return SimpleNamespace(choices=choices, usage=usage_obj)


class _FakeCompletions:
    def __init__(self, chunks=(), error: Exception | None = None) -> None:
        self.chunks = list(chunks)
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return iter(self.chunks)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="（模型答案）"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


class _FakeClient:
    def __init__(self, chunks=(), error: Exception | None = None) -> None:
        self.completions = _FakeCompletions(chunks, error)
        self.chat = SimpleNamespace(completions=self.completions)


def _generator(*, api_key: str = "test-key", retries: int = 2) -> AnswerGenerator:
    cfg = LLMConfig(base_url="http://fake", api_key=api_key, model="qwen-plus", temperature=0.0)
    return AnswerGenerator(cfg=cfg, retries=retries)


@pytest.fixture
def retrieval(chunk_set) -> RetrievalResult:
    articles = tuple(
        RetrievedArticle(
            article=parent,
            score=1.0 - index * 0.1,
            vector_rank=index + 1,
            bm25_rank=index + 1,
            vector_score=None,
            bm25_score=None,
            hit_chunks=(),
            law_hint=None,
        )
        for index, parent in enumerate(chunk_set.parents[:2])
    )
    return RetrievalResult(
        query="醉驾怎么处罚", articles=articles, used_vector=True, used_bm25=True,
        elapsed_ms=1.0, notes=(),
    )


EMPTY = RetrievalResult(
    query="库外问题", articles=(), used_vector=False, used_bm25=False, elapsed_ms=0.0, notes=()
)
QUESTION = Question(text="醉驾怎么处罚")


def _deltas(gen: AnswerGenerator, retrieval: RetrievalResult, question=QUESTION) -> tuple[str, dict]:
    """跑一遍流式，返回（拼起来的文本, usage）。"""
    parts, usage = [], {}
    for kind, payload in gen.stream(question, retrieval):
        if kind == "delta":
            parts.append(payload)
        elif kind == "usage":
            usage = payload
    return "".join(parts), usage


# ------------------------------------------------------------------ 头条断言
def test_流式与非流式发同一份提示词(retrieval):
    """两条路径必须构造出逐字相同的 messages —— 否则同一个问题在
    `/qa` 与 `/qa/stream` 两个入口会得到不同的答案。"""
    gen = _generator()
    gen._client = client = _FakeClient([_chunk("答案")])

    gen.generate(QUESTION, retrieval)
    _deltas(gen, retrieval)

    assert len(client.completions.calls) == 2
    plain, streamed = client.completions.calls

    assert plain["messages"] == streamed["messages"]
    assert plain["model"] == streamed["model"] == "qwen-plus"
    assert plain["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert streamed["stream"] is True
    assert "stream" not in plain


def test_提示词里带上依据(retrieval):
    gen = _generator()
    gen._client = client = _FakeClient([_chunk("答案")])

    gen.generate(QUESTION, retrieval)
    user = client.completions.calls[0]["messages"][-1]["content"]

    assert QUESTION.text in user
    assert "依据（共 2 条）" in user
    assert "【依据1】" in user and "【依据2】" in user


# ------------------------------------------------------------------ 流式
def test_流式拼接结果完整(retrieval):
    gen = _generator()
    gen._client = _FakeClient([_chunk("依据"), _chunk("[依据1]"), _chunk("应处拘役。")])

    text, _ = _deltas(gen, retrieval)

    assert text == "依据[依据1]应处拘役。"


def test_跳过空delta(retrieval):
    """OpenAI 兼容端点的首块常常只有 role 没有 content —— 不能因此吐出 None。"""
    gen = _generator()
    gen._client = _FakeClient([_chunk(None), _chunk("正文")])

    text, _ = _deltas(gen, retrieval)

    assert text == "正文"


def test_末块usage被读出(retrieval):
    gen = _generator()
    gen._client = _FakeClient([_chunk("正文"), _chunk(usage=(11, 22, 33))])

    _, usage = _deltas(gen, retrieval)

    assert usage == {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}


def test_端点不给usage时如实返回空(retrieval):
    """流式下 usage 只在末块出现，且不是所有兼容端点都会给（DashScope 就不给）。"""
    gen = _generator()
    gen._client = _FakeClient([_chunk("正文")])

    _, usage = _deltas(gen, retrieval)

    assert usage == {}


# ------------------------------------------------------------------ 兜底路径
def test_检索为空时流式不调模型():
    gen = _generator()
    gen._client = client = _FakeClient([_chunk("不该被调用")])

    text, usage = _deltas(gen, EMPTY)

    assert client.completions.calls == [], "检索为空时不该调用 LLM"
    assert text == EMPTY_RETRIEVAL_ANSWER
    assert usage == {}


def test_检索为空时两条路径文案一致():
    """同一句拒答在两个入口里措辞不同，用户会以为是两种不同的失败。"""
    gen = _generator()
    gen._client = _FakeClient()

    assert _deltas(gen, EMPTY)[0] == gen.generate(QUESTION, EMPTY).text


def test_未配置LLM时两条路径文案一致(retrieval):
    gen = _generator(api_key="")
    gen._client = client = _FakeClient()

    assert _deltas(gen, retrieval)[0] == UNAVAILABLE_ANSWER
    assert gen.generate(QUESTION, retrieval).text == UNAVAILABLE_ANSWER
    assert client.completions.calls == []


def test_未配置LLM时仍给出召回的法条(retrieval):
    """没配 key 也要让用户看到检索到了什么，而不是一片空白。"""
    answer = _generator(api_key="").generate(QUESTION, retrieval)

    assert [e.citation for e in answer.evidences] == [hit.citation for hit in retrieval.articles]
    assert answer.model == "(unavailable)"


# ------------------------------------------------------------------ 失败语义
def test_流式首字节前失败抛可读错误(retrieval):
    gen = _generator()
    gen._client = _FakeClient(error=RuntimeError("connection reset"))

    with pytest.raises(RuntimeError, match="调用 qwen-plus 失败"):
        next(gen.stream(QUESTION, retrieval))


def test_流式不重试(retrieval):
    """已经吐出去的 token 收不回来，重试会得到拼接了两次的答案。"""
    gen = _generator(retries=3)
    gen._client = client = _FakeClient(error=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        next(gen.stream(QUESTION, retrieval))

    assert len(client.completions.calls) == 1


def test_同步路径仍然重试(retrieval):
    """非流式没有「已经吐出去」的问题，重试仍然是对的。"""
    gen = _generator(retries=3)
    gen._client = client = _FakeClient(error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="调用 qwen-plus 失败"):
        gen.generate(QUESTION, retrieval)

    assert len(client.completions.calls) == 3
