from __future__ import annotations

import json
import threading
import time

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：并发闸验收跳过")
pytest.importorskip("httpx", reason="客户端要 httpx")

from fastapi.testclient import TestClient  # noqa: E402

from rag_contracts.domain.retrieval import RetrievalResult  # noqa: E402
from rag_contracts.observability import langfuse as lt  # noqa: E402
from rag_contracts.observability.tracer import Tracer  # noqa: E402
from rag_service.api import app as server  # noqa: E402
from rag_service.api import gate as gate_mod  # noqa: E402
from rag_service.api.gate import QUEUE_TIMEOUT_DETAIL, RETRY_AFTER_SECONDS  # noqa: E402
from rag_service.api.runtime import Runtime  # noqa: E402
from rag_service.indexing.readiness import ReadyState  # noqa: E402

QUESTION = "醉驾怎么处罚"
STUB_WAIT_SECONDS = 10.0


class SlowRag:

    def __init__(self) -> None:
        self.top_k = 6
        self.llm_ready = True
        self.model_name = "stub-model"
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_search = True
        self.block_stream = False
        self.entries = 0

    def _retrieval(self, question: str) -> RetrievalResult:
        return RetrievalResult(
            query=question, articles=(), used_vector=False, used_bm25=True, elapsed_ms=1.0
        )

    def search(self, question, *, top_k=None, channel_debug=False, law_filter=(), candidates=None):
        self.entries += 1
        if self.block_search:
            self.entered.set()
            self.release.wait(STUB_WAIT_SECONDS)
        return self._retrieval(question)

    def search_materials(self, query, doc_ids, top_k=6):
        self.entries += 1
        if self.block_search:
            self.entered.set()
            self.release.wait(STUB_WAIT_SECONDS)
        return [], ""

    def laws(self) -> list:
        return []

    def stream(self, question, retrieval, *, timeliness=(), materials=()):
        yield "delta", "甲"
        if self.block_stream:
            self.entered.set()
            self.release.wait(STUB_WAIT_SECONDS)
        yield "delta", "乙"


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
        rag=SlowRag(),
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


def _occupy(rt) -> tuple[threading.Thread, dict]:
    holder = TestClient(server.app)
    result: dict = {}

    def hold() -> None:
        result["resp"] = holder.post("/qa", json={"question": "谁占着闸", "mode": "search"})

    thread = threading.Thread(target=hold)
    thread.start()
    assert rt.rag.entered.wait(STUB_WAIT_SECONDS), "占位请求没进到桩里"
    return thread, result


def test_a_queued_request_times_out_with_503_and_retry_after(monkeypatch, service) -> None:
    client, rt = service
    monkeypatch.setenv("RAG_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_QUEUE_TIMEOUT", "0.3")
    thread, result = _occupy(rt)
    try:
        started = time.perf_counter()
        resp = client.post("/qa", json={"question": "排队的", "mode": "search"})
        elapsed = time.perf_counter() - started

        assert resp.status_code == 503
        assert resp.json() == {"detail": QUEUE_TIMEOUT_DETAIL}
        assert "排队超时" in resp.json()["detail"]
        assert resp.headers["Retry-After"] == RETRY_AFTER_SECONDS
        assert elapsed >= 0.25

        assert client.get("/health").status_code == 200
        assert client.get("/laws").status_code == 200
    finally:
        rt.rag.release.set()
        thread.join(timeout=STUB_WAIT_SECONDS)

    assert not thread.is_alive()
    assert result["resp"].status_code == 200
    assert client.post("/qa", json={"question": "放行后的", "mode": "search"}).status_code == 200


def test_a_queued_stream_gets_an_error_frame_not_a_503(monkeypatch, service) -> None:
    client, rt = service
    monkeypatch.setenv("RAG_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_QUEUE_TIMEOUT", "0.3")
    thread, _result = _occupy(rt)
    try:
        resp = client.post("/qa/stream", json={"question": "排队的流"})

        assert resp.status_code == 200
        events = _events(resp.text)
        assert [name for name, _ in events] == ["error"]
        assert json.loads(events[0][1])["message"] == QUEUE_TIMEOUT_DETAIL
    finally:
        rt.rag.release.set()
        thread.join(timeout=STUB_WAIT_SECONDS)


def test_a_queued_answer_stream_gets_an_error_frame(monkeypatch, service) -> None:
    client, rt = service
    monkeypatch.setenv("RAG_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_QUEUE_TIMEOUT", "0.3")
    body = {
        "question": {"text": QUESTION, "history": []},
        "retrieval": RetrievalResult(
            query=QUESTION, articles=(), used_vector=False, used_bm25=False, elapsed_ms=0.0
        ).to_dict(),
    }
    thread, _result = _occupy(rt)
    try:
        resp = client.post("/answer/stream", json=body)

        assert resp.status_code == 200
        events = _events(resp.text)
        assert [name for name, _ in events] == ["error"]
        assert json.loads(events[0][1])["message"] == QUEUE_TIMEOUT_DETAIL
    finally:
        rt.rag.release.set()
        thread.join(timeout=STUB_WAIT_SECONDS)


def test_materials_search_shares_the_same_gate(monkeypatch, service) -> None:
    client, rt = service
    monkeypatch.setenv("RAG_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_QUEUE_TIMEOUT", "0.3")
    thread, _result = _occupy(rt)
    try:
        resp = client.post("/materials/search", json={"query": "培训费"})

        assert resp.status_code == 503
        assert resp.json() == {"detail": QUEUE_TIMEOUT_DETAIL}
        assert resp.headers["Retry-After"] == RETRY_AFTER_SECONDS
    finally:
        rt.rag.release.set()
        thread.join(timeout=STUB_WAIT_SECONDS)


def test_zero_concurrency_switches_the_gate_off(monkeypatch, service) -> None:
    client, rt = service
    monkeypatch.setenv("RAG_MAX_CONCURRENCY", "0")
    monkeypatch.setenv("RAG_QUEUE_TIMEOUT", "0.2")
    thread, result = _occupy(rt)
    second: dict = {}

    def follow() -> None:
        second["resp"] = TestClient(server.app).post(
            "/qa", json={"question": "关闸后的第二发", "mode": "search"}
        )

    follower = threading.Thread(target=follow)
    follower.start()
    try:
        time.sleep(0.5)
        assert follower.is_alive(), "闸关时第二发不该被拦"
    finally:
        rt.rag.release.set()
        thread.join(timeout=STUB_WAIT_SECONDS)
        follower.join(timeout=STUB_WAIT_SECONDS)

    assert result["resp"].status_code == 200
    assert second["resp"].status_code == 200
    assert rt.rag.entries == 2


def test_a_closed_stream_generator_releases_the_gate(monkeypatch, service) -> None:
    client, rt = service
    monkeypatch.setenv("RAG_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_QUEUE_TIMEOUT", "0.3")
    rt.rag.block_search = False
    rt.rag.block_stream = True

    events = server._sse_events(rt, server.QaRequest(question="半路断开"))
    assert next(events).startswith("event: evidence")
    assert next(events).startswith("event: delta")

    sem = gate_mod._state["sem"]
    assert sem.acquire(timeout=0.1) is False
    events.close()
    assert sem.acquire(timeout=0.5) is True
    sem.release()

    follow = client.post("/qa", json={"question": "断开后的", "mode": "search"})
    assert follow.status_code == 200
