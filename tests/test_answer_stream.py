from __future__ import annotations

import json
from dataclasses import replace

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：流式端点验收跳过")
pytest.importorskip("httpx", reason="客户端要 httpx")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api_contracts import RagClient  # noqa: E402
from rag_contracts import config  # noqa: E402
from rag_contracts.domain.answer import Answer, Question, drain  # noqa: E402
from rag_contracts.domain.disk import ParentChunk  # noqa: E402
from rag_contracts.domain.errors import QaError  # noqa: E402
from rag_contracts.domain.retrieval import (  # noqa: E402
    MaterialPassage,
    RetrievalResult,
    RetrievedArticle,
    WebFinding,
)
from rag_contracts.observability import langfuse as lt  # noqa: E402
from rag_contracts.observability.tracer import Tracer  # noqa: E402
from rag_contracts.ports import TRUNCATED_FINISH_REASON  # noqa: E402
from rag_service.api import app as server  # noqa: E402
from rag_service.api.runtime import Runtime  # noqa: E402
from rag_service.indexing.readiness import ReadyState  # noqa: E402
from rag_service.query.generator import AnswerGenerator  # noqa: E402

LAW_NAME = "中华人民共和国道路交通安全法"
ARTICLE_TEXT = "醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证。"
NORMAL_DELTAS = ["醉驾", "按第九十一条", "处罚。"]

PARENT = ParentChunk(
    parent_id=f"{LAW_NAME}#第九十一条",
    law_id="road_traffic_safety",
    law_name=LAW_NAME,
    version="2021",
    citation=f"《{LAW_NAME}》",
    article_no="第九十一条",
    article_index=91,
    chapter="第七章 法律责任",
    section=None,
    text=ARTICLE_TEXT,
)

WEB_FINDING = WebFinding(
    label="[时效1]",
    title="醉驾处罚最新口径",
    url="https://example.com/stream",
    snippet="示例片段",
)

MATERIAL = MaterialPassage(
    label="[材料1]",
    doc_id="d0",
    display_name="车辆管理规定.md",
    index=0,
    text="培训费用按每人每年一千二百元包干。",
)


def _retrieval(question: str, *, empty: bool = False, used_vector: bool = False) -> RetrievalResult:
    articles = () if empty else (RetrievedArticle(article=PARENT, score=0.9),)
    return RetrievalResult(
        query=question,
        articles=articles,
        used_vector=used_vector,
        used_bm25=not empty,
        elapsed_ms=1.0,
    )


def _answer(question: str = "醉驾怎么处罚", text: str = "醉驾按第九十一条处罚。[依据1]") -> Answer:
    return Answer(
        question=question,
        text=text,
        evidences=(),
        model="stub-model",
        elapsed_ms=1.0,
        usage={"total_tokens": 9},
        retrieval=_retrieval(question),
    )


def _zeroed(answer: Answer) -> Answer:
    return replace(answer, elapsed_ms=0.0)


class FakeLLM:

    def __init__(
        self,
        deltas: list[str],
        *,
        usage: dict | None = None,
        finish: str = "stop",
        model: str = "内存模型",
    ) -> None:
        self._deltas = list(deltas)
        self.usage = usage if usage is not None else {"total_tokens": 7}
        self.finish = finish
        self.model_name = model
        self.available = True
        self.chats = 0
        self.streams = 0
        self.prompts: list[list[dict]] = []

    def chat(self, messages, *, tools=None, temperature=None, name="llm.chat"):
        self.chats += 1
        self.prompts.append(list(messages))
        return {"role": "assistant", "content": "".join(self._deltas)}, dict(self.usage), self.finish

    def stream(self, messages, *, temperature=None):
        self.streams += 1
        self.prompts.append(list(messages))
        for piece in self._deltas:
            yield "delta", piece
        yield "finish_reason", self.finish
        yield "usage", dict(self.usage)


def _generator(llm=None, *, api_key: str = "stub") -> AnswerGenerator:
    return AnswerGenerator(
        config.LLMConfig(base_url="http://stub", api_key=api_key, model="内存模型"),
        llm=llm if llm is not None else FakeLLM(NORMAL_DELTAS),
    )


def test_drain_calls_back_in_frame_order_and_returns_the_tail() -> None:
    seen: list[str] = []
    tail = _answer()
    frames = [("delta", "甲"), ("usage", {"total_tokens": 1}), ("delta", "乙"), ("answer", tail)]

    assert drain(frames, seen.append) is tail
    assert seen == ["甲", "乙"]


def test_drain_ignores_every_kind_it_does_not_know() -> None:
    seen: list[str] = []
    tail = _answer()
    frames = [
        ("usage", {}),
        ("finish_reason", "stop"),
        ("delta", "甲"),
        ("未来帧", 1),
        ("answer", tail),
        ("delta", "尾后"),
    ]

    assert drain(frames, seen.append) is tail
    assert seen == ["甲", "尾后"]


def test_drain_keeps_the_last_answer_frame() -> None:
    first, second = _answer(text="一稿"), _answer(text="二稿")

    assert drain([("answer", first), ("answer", second)]) == second


def test_drain_without_a_tail_frame_raises() -> None:
    with pytest.raises(QaError):
        drain([("delta", "半句"), ("usage", {})])
    with pytest.raises(QaError):
        drain([])


def test_stream_tail_matches_generate_on_the_normal_route() -> None:
    retrieval = _retrieval("醉驾怎么处罚")
    question = Question(text="醉驾怎么处罚")
    llm = FakeLLM(NORMAL_DELTAS, usage={"total_tokens": 7})

    frames = list(_generator(llm).stream(question, retrieval))

    assert [kind for kind, _ in frames] == [
        "delta",
        "delta",
        "delta",
        "finish_reason",
        "usage",
        "answer",
    ]
    tail = frames[-1][1]
    assert "".join(text for kind, text in frames if kind == "delta") == tail.text
    assert tail.text == "醉驾按第九十一条处罚。"
    assert tail.model == "内存模型" and tail.usage == {"total_tokens": 7}
    expected = _generator(FakeLLM(NORMAL_DELTAS, usage={"total_tokens": 7})).generate(
        question, retrieval
    )
    assert _zeroed(tail) == _zeroed(expected)


def test_stream_tail_matches_generate_on_the_empty_retrieval_route() -> None:
    retrieval = _retrieval("没有命中的问题", empty=True)
    question = Question(text="没有命中的问题")
    llm = FakeLLM(["不该被用到"])
    generator = _generator(llm)

    frames = list(generator.stream(question, retrieval))

    assert [kind for kind, _ in frames] == ["delta", "usage", "answer"]
    tail = frames[-1][1]
    assert "".join(text for kind, text in frames if kind == "delta") == tail.text
    assert tail.model == "(skip)" and tail.usage == {}
    assert _zeroed(tail) == _zeroed(generator.generate(question, retrieval))
    assert llm.chats == 0 and llm.streams == 0, "空检索这条路不该碰模型"


def test_stream_tail_matches_generate_when_the_llm_is_not_configured() -> None:
    retrieval = _retrieval("醉驾怎么处罚")
    question = Question(text="醉驾怎么处罚")
    llm = FakeLLM(["不该被用到"])
    generator = _generator(llm, api_key="")

    frames = list(generator.stream(question, retrieval))

    assert [kind for kind, _ in frames] == ["delta", "usage", "answer"]
    tail = frames[-1][1]
    assert "".join(text for kind, text in frames if kind == "delta") == tail.text
    assert tail.model == "(unavailable)"
    expected = generator.generate(question, retrieval)
    assert _zeroed(tail) == _zeroed(expected)
    assert expected.notes and "未配置" in expected.notes[0]
    assert llm.chats == 0 and llm.streams == 0, "未配置这条路不该碰模型"


def test_stream_marks_a_truncated_answer_like_generate_does() -> None:
    retrieval = _retrieval("醉驾怎么处罚")
    question = Question(text="醉驾怎么处罚")
    llm = FakeLLM(NORMAL_DELTAS, finish=TRUNCATED_FINISH_REASON)

    frames = list(_generator(llm).stream(question, retrieval))

    tail = frames[-1][1]
    expected = _generator(FakeLLM(NORMAL_DELTAS, finish=TRUNCATED_FINISH_REASON)).generate(
        question, retrieval
    )
    assert any("截断" in note for note in tail.notes)
    assert any("截断" in note for note in expected.notes)
    assert _zeroed(tail) == _zeroed(expected)


def test_stream_feeds_timeliness_and_materials_into_the_prompt_like_generate() -> None:
    retrieval = _retrieval("醉驾怎么处罚")
    question = Question(text="醉驾怎么处罚")
    timeliness = (WEB_FINDING,)
    materials = (MATERIAL,)
    streaming_llm = FakeLLM(NORMAL_DELTAS)
    plain_llm = FakeLLM(NORMAL_DELTAS)

    frames = list(
        _generator(streaming_llm).stream(
            question, retrieval, timeliness=timeliness, materials=materials
        )
    )
    tail = frames[-1][1]
    expected = _generator(plain_llm).generate(
        question, retrieval, timeliness=timeliness, materials=materials
    )

    assert streaming_llm.prompts == plain_llm.prompts, "同输入的提示词没对上，流式这条路的接线是断的"
    prompt = streaming_llm.prompts[-1][-1]["content"]
    assert WEB_FINDING.title in prompt and MATERIAL.text in prompt
    assert _zeroed(tail) == _zeroed(expected)


STREAM_DELTAS = ["醉驾按", "第九十一条", "处罚。[依据1]"]


class StubRag:

    def __init__(self) -> None:
        self.top_k = 6
        self.llm_ready = True
        self.model_name = "stub-model"
        self.streamed: list[tuple] = []
        self.answers = 0
        self.deltas = list(STREAM_DELTAS)
        self.fail: Exception | None = None
        self.tail = True

    def _answer(self, question, retrieval, timeliness, materials) -> Answer:
        return Answer(
            question=question.text,
            text="".join(self.deltas),
            evidences=(),
            model="stub-model",
            elapsed_ms=1.0,
            retrieval=retrieval,
            timeliness=tuple(timeliness),
            materials=tuple(materials),
        )

    def answer(self, question, retrieval, *, timeliness=(), materials=()):
        self.answers += 1
        return self._answer(question, retrieval, timeliness, materials)

    def stream(self, question, retrieval, *, timeliness=(), materials=()):
        self.streamed.append((question, retrieval, tuple(timeliness), tuple(materials)))
        if self.fail is not None:
            raise self.fail
        for piece in self.deltas:
            yield "delta", piece
        if self.tail:
            yield "answer", self._answer(question, retrieval, timeliness, materials)


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(lt, "from_env", lambda: Tracer())
    rt = Runtime(
        ready=ReadyState(
            action="reuse",
            reason="",
            rows=812,
            laws=6,
            articles=508,
            parents=508,
            dense=True,
            milvus="v2.6.24",
        ),
        rag=StubRag(),
        boot_ms=0.0,
    )
    server.app.state.rt = rt
    server.app.state.boot_error = None
    return TestClient(server.app), rt


def _events(body: str) -> list[tuple[str, str]]:
    return [
        (
            block.splitlines()[0].removeprefix("event: "),
            block.splitlines()[1].removeprefix("data: "),
        )
        for block in body.strip().split("\n\n")
    ]


def _body(question: str = "醉驾怎么处罚", *, history=(), used_vector: bool = False, **extra) -> dict:
    return {
        "question": {"text": question, "history": [list(turn) for turn in history]},
        "retrieval": _retrieval(question, used_vector=used_vector).to_dict(),
        **extra,
    }


def test_the_stream_endpoint_merges_deltas_into_the_done_payload(service) -> None:
    client, rt = service

    resp = client.post("/answer/stream", json=_body())

    assert resp.status_code == 200
    events = _events(resp.text)
    assert [name for name, _ in events] == ["delta", "delta", "delta", "done"]
    joined = "".join(json.loads(payload)["text"] for name, payload in events if name == "delta")
    done = json.loads(events[-1][1])
    assert joined == done["answer"] == "".join(rt.rag.deltas)
    assert "request_ms" in done and "usage" in done and "retrieval" in done


def test_answer_and_answer_stream_return_the_same_payload(service) -> None:
    client, _ = service

    plain = client.post("/answer", json=_body()).json()
    streamed = json.loads(_events(client.post("/answer/stream", json=_body()).text)[-1][1])

    for payload in (plain, streamed):
        payload.pop("request_ms")
        payload.pop("elapsed_ms")
    assert streamed == plain


def test_the_stream_endpoint_parses_the_same_request_as_answer(service) -> None:
    client, rt = service
    body = _body(
        "培训费能报多少",
        history=[("user", "你好")],
        used_vector=True,
        timeliness=[WEB_FINDING.to_dict()],
        materials=[MATERIAL.to_dict()],
    )

    resp = client.post("/answer/stream", json=body)

    assert resp.status_code == 200
    question, retrieval, timeliness, materials = rt.rag.streamed[-1]
    assert question.text == "培训费能报多少"
    assert question.history == (("user", "你好"),)
    assert retrieval.query == "培训费能报多少"
    assert [item.title for item in timeliness] == [WEB_FINDING.title]
    assert [item.text for item in materials] == [MATERIAL.text]
    assert rt.dense_live is True


def test_a_failing_stream_ends_with_an_error_frame_and_no_done(service) -> None:
    client, rt = service
    rt.rag.fail = RuntimeError("调用 stub-model 失败：连接超时")

    resp = client.post("/answer/stream", json=_body())

    assert resp.status_code == 200
    events = _events(resp.text)
    assert [name for name, _ in events] == ["error"]
    assert json.loads(events[-1][1])["message"] == "调用 stub-model 失败：连接超时"


def test_a_stream_without_the_tail_frame_ends_with_an_error_and_no_done(service) -> None:
    client, rt = service
    rt.rag.tail = False

    resp = client.post("/answer/stream", json=_body())

    assert resp.status_code == 200
    events = _events(resp.text)
    assert [name for name, _ in events] == ["delta", "delta", "delta", "error"]
    message = json.loads(events[-1][1])["message"]
    assert message and "answer" in message


def test_a_missing_runtime_still_gets_the_503_before_anything_streams(monkeypatch) -> None:
    monkeypatch.setattr(server.app.state, "rt", None, raising=False)
    monkeypatch.setattr(
        server.app.state, "boot_error", "知识库为空：docx 下没有源文件", raising=False
    )

    resp = TestClient(server.app).post("/answer/stream", json=_body())

    assert resp.status_code == 503
    assert "知识库为空" in resp.json()["detail"]


def _client(handler) -> RagClient:
    return RagClient(
        "http://stub",
        client=httpx.Client(base_url="http://stub", transport=httpx.MockTransport(handler)),
    )


def _sse(*events: tuple[str, dict]) -> str:
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        for name, payload in events
    )


def test_answer_stream_relays_deltas_and_parses_the_done_frame() -> None:
    answer = _answer()
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["path"] = request.url.path
        sent["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            text=_sse(
                ("delta", {"text": "醉驾按"}),
                ("delta", {"text": "第九十一条处罚。"}),
                ("done", dict(answer.to_dict(), request_ms=12.34)),
            ),
            headers={"content-type": "text/event-stream"},
        )

    frames = list(_client(handler).answer_stream("醉驾怎么处罚", _retrieval("醉驾怎么处罚")))

    assert frames[0] == ("delta", "醉驾按")
    assert frames[1] == ("delta", "第九十一条处罚。")
    kind, got = frames[2]
    assert kind == "answer" and isinstance(got, Answer)
    assert got.text == answer.text
    assert sent["path"] == "/answer/stream"
    assert sent["body"]["question"]["text"] == "醉驾怎么处罚"


def test_an_error_frame_comes_back_as_a_qa_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_sse(("delta", {"text": "半句"}), ("error", {"message": "检索结果为空"})),
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(QaError) as excinfo:
        list(_client(handler).answer_stream("q", _retrieval("q")))

    assert str(excinfo.value) == "检索结果为空"


def test_a_stream_that_ends_without_done_comes_back_as_a_qa_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_sse(("delta", {"text": "半句"})),
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(QaError) as excinfo:
        list(_client(handler).answer_stream("q", _retrieval("q")))

    assert "done" in str(excinfo.value)


def test_a_non_200_stream_response_comes_back_as_a_qa_error_with_the_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "服务未就绪"})

    with pytest.raises(QaError) as excinfo:
        list(_client(handler).answer_stream("q", _retrieval("q")))

    assert "503" in str(excinfo.value) and "服务未就绪" in str(excinfo.value)
