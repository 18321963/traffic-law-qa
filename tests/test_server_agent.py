from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：服务层验收跳过")
pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")

from fastapi.testclient import TestClient  # noqa: E402

from traffic_law_qa import main as server  # noqa: E402
from traffic_law_qa.agents import langfuse_tracer as lt  # noqa: E402
from traffic_law_qa.agents.graph import AgentRunner  # noqa: E402
from traffic_law_qa.contracts import Answer, RetrievalResult, Review  # noqa: E402
from traffic_law_qa.obs import Tracer  # noqa: E402
from traffic_law_qa.ready import ReadyState  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEAVY = ("langgraph", "pymilvus", "openai", "langfuse", "langchain_core")


def _retrieval(question: str, *, used_vector: bool = False) -> RetrievalResult:
    return RetrievalResult(
        query=question, articles=(), used_vector=used_vector, used_bm25=True, elapsed_ms=1.0
    )


def _answer(question: str) -> Answer:
    return Answer(
        question=question,
        text="醉驾按…处罚。[依据1]",
        evidences=(),
        model="stub",
        elapsed_ms=1.0,
        retrieval=_retrieval(question, used_vector=True),
        review=Review(
            score=1.0,
            threshold=0.6,
            total=1,
            supported=1,
            unsupported=(),
            original_text="…",
            model="review-stub",
            passed=True,
        ),
    )


class StubRag:

    def __init__(self) -> None:
        self.top_k = 6
        self.llm_ready = True
        self.searches: list[tuple[str, int | None]] = []
        self.answers = 0

    def search(self, question: str, top_k: int | None = None) -> RetrievalResult:
        self.searches.append((question, top_k))
        return _retrieval(question)

    def answer(self, question, retrieval) -> Answer:
        self.answers += 1
        return _answer(question.text)


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(lt, "from_env", lambda: Tracer())
    calls: dict = {}

    def fake_graph(self):
        calls["graph"] = calls.get("graph", 0) + 1

    def fake_ask(self, question, *, material_ids=()):
        calls["question"] = question
        calls["material_ids"] = list(material_ids)
        return _answer(question.text)

    monkeypatch.setattr(AgentRunner, "graph", fake_graph)
    monkeypatch.setattr(AgentRunner, "ask", fake_ask)

    rt = server.Runtime(
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
        agent_extra=True,
    )
    server.app.state.rt = rt
    server.app.state.boot_error = None
    return TestClient(server.app), rt, calls


def test_mode_agent_goes_through_the_runner(service):
    client, rt, calls = service
    resp = client.post("/qa", json={"question": "醉驾怎么处罚", "mode": "agent", "top_k": 3})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["mode"] == "agent"
    assert payload["answer"] == "醉驾按…处罚。[依据1]"
    assert payload["review"] is not None
    assert calls["question"].text == "醉驾怎么处罚"
    assert calls["question"].top_k == 3
    assert rt.rag.searches == []
    assert rt.dense_live is True
    assert calls["graph"] == 1


def test_doc_ids_reach_the_runner(service):
    client, rt, calls = service
    resp = client.post(
        "/qa", json={"question": "q", "mode": "agent", "doc_ids": ["d1", "d2"]}
    )
    assert resp.status_code == 200
    assert calls["material_ids"] == ["d1", "d2"]
    client.post("/qa", json={"question": "q", "mode": "agent"})
    assert calls["material_ids"] == []


def test_agent_is_assembled_once_and_review_is_not_the_main_model(service):
    client, rt, calls = service
    for _ in range(3):
        assert client.post("/qa", json={"question": "q", "mode": "agent"}).status_code == 200
    assert calls["graph"] == 1
    assert rt.agent is not None
    assert rt.agent.review_llm is not rt.agent.llm


def test_modes_echo_and_keep_their_own_shapes(service):
    client, rt, _ = service
    search = client.post("/qa", json={"question": "q", "mode": "search"}).json()
    assert search["mode"] == "search" and "query" in search
    assert rt.rag.searches == [("q", None)] and rt.rag.answers == 0
    ask = client.post("/qa", json={"question": "q"}).json()
    assert ask["mode"] == "ask" and "answer" in ask
    assert rt.rag.answers == 1


def test_mode_agent_without_the_extra_is_503_one_line(service):
    client, rt, _ = service
    rt.agent_extra = False
    resp = client.post("/qa", json={"question": "q", "mode": "agent"})
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "langgraph" in detail and "\n" not in detail
    assert rt.agent is None


def test_mode_agent_without_llm_key_is_503(service):
    client, rt, _ = service
    rt.rag.llm_ready = False
    resp = client.post("/qa", json={"question": "q", "mode": "agent"})
    assert resp.status_code == 503
    assert "LLM_API_KEY" in resp.json()["detail"]


def test_stream_rejects_mode_agent(service):
    client, _, _ = service
    resp = client.post("/qa/stream", json={"question": "q", "mode": "agent"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "agent" in detail and "\n" not in detail


def test_runtime_error_reaches_the_client_as_one_line(service):
    client, rt, _ = service

    def boom(question, top_k=None):
        raise RuntimeError("调用 stub-model 失败：连接超时")

    rt.rag.search = boom
    resp = client.post("/qa", json={"question": "q"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "服务内部错误：调用 stub-model 失败：连接超时"


def test_health_reports_agent_ready_without_assembling(service):
    client, rt, calls = service
    body = client.get("/health").json()
    assert body["agent_ready"] is True and body["version"] == server.app.version
    assert rt.agent is None and "graph" not in calls
    rt.agent_extra = False
    assert client.get("/health").json()["agent_ready"] is False


def test_importing_main_keeps_heavy_deps_out():
    code = (
        "import sys, traffic_law_qa.main;"
        f"print(','.join(sorted({{m.split('.')[0] for m in sys.modules}} & set({HEAVY!r}))))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "", f"import traffic_law_qa.main 拉起了：{proc.stdout.strip()}"
