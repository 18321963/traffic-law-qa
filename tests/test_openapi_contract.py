from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：契约验收跳过")
pytest.importorskip("httpx", reason="客户端要 httpx")

import httpx  # noqa: E402

from api_contracts import RagClient  # noqa: E402
from rag_contracts.domain.answer import Question  # noqa: E402
from rag_contracts.domain.errors import QaError, QaTimeout  # noqa: E402
from rag_contracts.domain.retrieval import RetrievalResult  # noqa: E402
from rag_service.api.app import app  # noqa: E402

SPEC_PATH = Path(__file__).resolve().parent.parent / "api_contracts" / "openapi.json"


def _record(seen: list[tuple[str, str]]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/answer/stream":
            return httpx.Response(
                200,
                text='event: done\ndata: {"answer": "好"}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"ok": True})

    return handler


def _stub(handler) -> RagClient:
    return RagClient(
        "http://stub",
        client=httpx.Client(base_url="http://stub", transport=httpx.MockTransport(handler)),
    )


def test_the_committed_spec_is_what_the_app_serves():
    assert json.loads(SPEC_PATH.read_text(encoding="utf-8")) == app.openapi()


def test_the_committed_spec_is_canonically_formatted():
    raw = SPEC_PATH.read_text(encoding="utf-8")
    assert raw == json.dumps(json.loads(raw), ensure_ascii=False, indent=2) + "\n"


def test_the_client_only_calls_operations_the_spec_declares():
    seen: list[tuple[str, str]] = []
    client = _stub(_record(seen))

    client.health()
    client.laws()
    client.qa("醉驾怎么处罚")
    client.qa("醉驾怎么处罚", mode="ask", top_k=6, pool=50, debug=True, law_filter=["a"])
    client.article(article_no="第九十条", law_name="道路交通安全法")
    client.article(text="《道路交通安全法》第九十条")
    client.materials("培训费", ["d0"], top_k=3)
    client.search("醉驾怎么处罚", 3, channel_debug=True, law_filter=["road"], candidates=20)
    client.ask("醉驾怎么处罚", 3)
    client.get_article("第九十条", "道路交通安全法")
    client.search_materials("培训费", ["d0"], top_k=3)
    list(
        client.answer_stream(
            "醉驾怎么处罚",
            RetrievalResult(
                query="q", articles=(), used_vector=False, used_bm25=False, elapsed_ms=1.0
            ),
        )
    )

    assert seen == [
        ("GET", "/health"),
        ("GET", "/laws"),
        ("POST", "/qa"),
        ("POST", "/qa"),
        ("POST", "/articles/lookup"),
        ("POST", "/articles/lookup"),
        ("POST", "/materials/search"),
        ("POST", "/qa"),
        ("POST", "/qa"),
        ("POST", "/answer"),
        ("POST", "/articles/lookup"),
        ("POST", "/materials/search"),
        ("POST", "/answer/stream"),
    ]
    spec = app.openapi()
    for method, path in seen:
        assert path in spec["paths"], f"{method} {path} 不在契约里"
        assert method.lower() in spec["paths"][path], f"{method} {path} 不在契约里"


def test_the_client_sends_only_what_the_spec_knows_about():
    schema = app.openapi()["components"]["schemas"]
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content or b"{}"))
        return httpx.Response(200, json={"ok": True})

    client = _stub(handler)
    client.qa("q", mode="search", top_k=3, pool=50, debug=True, law_filter=["a"])
    client.article(article_no="90", law_name="法", text="……")
    client.materials("培训费", ["d0"], top_k=3)
    client.answer(
        Question(text="q", history=(("user", "你好"),)),
        RetrievalResult(query="q", articles=(), used_vector=False, used_bm25=False, elapsed_ms=1.0),
    )

    sent = dict(
        zip(
            ("QaRequest", "ArticleLookup", "MaterialSearch", "AnswerRequest"), bodies, strict=True
        )
    )
    for name, body in sent.items():
        declared = set(schema[name]["properties"])
        assert set(body) <= declared, f"{name} 里有契约没声明的字段：{set(body) - declared}"

    asked = set(sent["AnswerRequest"]["question"])
    declared_question = set(schema["QuestionBody"]["properties"])
    assert asked <= declared_question, (
        f"QuestionBody 里有契约没声明的字段：{asked - declared_question}"
    )


def test_the_qa_contract_keeps_no_agent_knobs():
    properties = app.openapi()["components"]["schemas"]["QaRequest"]["properties"]

    assert set(properties) == {"question", "mode", "top_k", "pool", "debug", "law_filter"}, (
        "doc_ids（会话材料）与 mode=agent（agent 循环）都随 agent 服务出包了："
        "它们在 agent 服务的 /qa 上，不在这个契约里"
    )
    assert properties["mode"]["enum"] == ["ask", "search"]


def test_a_refusal_comes_back_as_a_qa_error_with_the_detail():
    client = _stub(lambda request: httpx.Response(503, json={"detail": "知识库为空：docx 下没有源文件"}))

    with pytest.raises(QaError) as excinfo:
        client.laws()

    assert "503" in str(excinfo.value) and "知识库为空" in str(excinfo.value)


def test_an_unreachable_service_comes_back_as_a_qa_error_too():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(QaError) as excinfo:
        _stub(handler).health()

    assert "connection refused" in str(excinfo.value)


def test_a_slow_service_comes_back_as_the_timeout_subclass():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(QaTimeout) as excinfo:
        _stub(handler).qa("醉驾怎么处罚", timeout=2.5)

    assert "超时（2.5 秒）" in str(excinfo.value)
    assert "连不上" not in str(excinfo.value)
