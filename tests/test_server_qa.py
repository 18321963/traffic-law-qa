from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：服务层验收跳过")
pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")

from fastapi.testclient import TestClient  # noqa: E402

from rag_contracts.domain.answer import Answer, Review  # noqa: E402
from rag_contracts.domain.reports import WEIGHTS_UNKNOWN  # noqa: E402
from rag_contracts.domain.retrieval import RetrievalResult  # noqa: E402
from rag_contracts.observability import langfuse as lt  # noqa: E402
from rag_contracts.observability.tracer import Tracer  # noqa: E402
from rag_contracts.ports import TRUNCATED_FINISH_REASON  # noqa: E402
from rag_service.api import app as server  # noqa: E402
from rag_service.api.runtime import Runtime  # noqa: E402
from rag_service.indexing.readiness import ReadyState  # noqa: E402

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
        self.model_name = "stub-model"
        self.searches: list[tuple[str, int | None]] = []
        self.streams: list[str] = []
        self.answers = 0
        self.finish = "stop"

    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        channel_debug: bool = False,
        law_filter: tuple[str, ...] = (),
        candidates: int | None = None,
    ) -> RetrievalResult:
        self.searches.append((question, top_k))
        return _retrieval(question)

    def answer(self, question, retrieval) -> Answer:
        self.answers += 1
        return _answer(question.text)

    def stream(self, question, retrieval):
        self.streams.append(question.text)
        yield "delta", "半句"
        yield "finish_reason", self.finish
        yield "usage", {"total_tokens": 5}


def _events(body: str) -> list[tuple[str, str]]:
    return [
        (
            block.splitlines()[0].removeprefix("event: "),
            block.splitlines()[1].removeprefix("data: "),
        )
        for block in body.strip().split("\n\n")
    ]


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


def test_modes_echo_and_keep_their_own_shapes(service):
    client, rt = service
    search = client.post("/qa", json={"question": "q", "mode": "search"}).json()
    assert search["mode"] == "search" and "query" in search
    assert rt.rag.searches == [("q", None)] and rt.rag.answers == 0
    ask = client.post("/qa", json={"question": "q"}).json()
    assert ask["mode"] == "ask" and "answer" in ask
    assert rt.rag.answers == 1


def test_the_rag_service_no_longer_runs_the_agent_loop(service):
    client, rt = service

    resp = client.post("/qa", json={"question": "q", "mode": "agent"})
    assert resp.status_code == 422, "mode=agent 随 agent 服务出包了，这个入口不该还认它"
    assert rt.rag.searches == [] and rt.rag.answers == 0
    assert client.post("/qa/stream", json={"question": "q", "mode": "agent"}).status_code == 422


def test_a_blank_question_is_refused_before_any_work(service):
    client, rt = service
    for path in ("/qa", "/qa/stream"):
        resp = client.post(path, json={"question": " "})
        assert resp.status_code == 422, path
        assert "空白" in resp.text, path
    blank_answer = client.post(
        "/answer",
        json={
            "question": {"text": "\n\t"},
            "retrieval": {
                "query": "q",
                "articles": [],
                "used_vector": False,
                "used_bm25": False,
                "elapsed_ms": 1.0,
            },
        },
    )
    assert blank_answer.status_code == 422
    assert "空白" in blank_answer.text
    assert rt.rag.searches == [] and rt.rag.streams == [] and rt.rag.answers == 0


def test_stream_reports_a_truncated_linear_answer(service):
    client, rt = service
    rt.rag.finish = TRUNCATED_FINISH_REASON
    resp = client.post("/qa/stream", json={"question": "q"})
    assert resp.status_code == 200
    events = _events(resp.text)
    assert [name for name, _ in events] == ["evidence", "delta", "done"]
    done = json.loads(events[-1][1])
    assert done["truncated"] is True
    assert done["answer"] == "半句" and done["usage"] == {"total_tokens": 5}


def test_stream_reports_a_complete_linear_answer(service):
    client, _ = service
    resp = client.post("/qa/stream", json={"question": "q"})
    assert resp.status_code == 200
    assert json.loads(_events(resp.text)[-1][1])["truncated"] is False


def test_runtime_error_reaches_the_client_as_one_line(service):
    client, rt = service

    def boom(question, top_k=None, *, channel_debug=False, law_filter=(), candidates=None):
        raise RuntimeError("调用 stub-model 失败：连接超时")

    rt.rag.search = boom
    resp = client.post("/qa", json={"question": "q"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "服务内部错误：调用 stub-model 失败：连接超时"


def test_health_says_degraded_when_the_fingerprint_could_not_be_checked(service):
    client, rt = service
    assert client.get("/health").json()["degraded"] is False
    assert "agent_ready" not in client.get("/health").json(), (
        "agent 就绪与否是 agent 服务自己的 /health 报的，rag 这边不该再替它答"
    )

    rt.ready = replace(rt.ready, weights=WEIGHTS_UNKNOWN)

    body = client.get("/health").json()
    assert body["weights"] == WEIGHTS_UNKNOWN and body["degraded"] is True


def test_importing_the_api_app_keeps_heavy_deps_out():
    code = (
        "import sys, rag_service.api.app;"
        f"print(','.join(sorted({{m.split('.')[0] for m in sys.modules}} & set({HEAVY!r}))))"
    )
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "", f"import rag_service.api.app 拉起了：{proc.stdout.strip()}"
