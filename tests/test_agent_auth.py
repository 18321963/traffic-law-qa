from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：agent 侧 HTTP 验收跳过")
pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")

from fastapi.testclient import TestClient  # noqa: E402

from agent_service.api import app as server  # noqa: E402
from agent_service.api.auth import keys_match  # noqa: E402

QUESTION = "在深圳，醉酒驾驶机动车怎么处罚？"


class _StubRagService:
    def health(self) -> dict:
        return {"status": "ok"}


class _StubRunner:
    def __init__(self) -> None:
        self.rag = _StubRagService()
        self.stream_calls = 0

    def ask_payload(self, question, *, material_ids, session_id):
        return "ok", {"question": question.text, "answer": "答案正文", "session_id": session_id}

    def stream(self, question, *, material_ids, session_id):
        self.stream_calls += 1
        return iter(())

    def resume(self, session_id, value):
        return "ok", {"answer": "答案正文", "session_id": session_id}

    def resume_stream(self, session_id, value):
        self.stream_calls += 1
        return iter(())


@pytest.fixture
def wire():
    runner = _StubRunner()
    server.app.state.runner = runner
    server.app.state.boot_error = None
    return TestClient(server.app), runner


def test_without_keys_configured_requests_pass_untouched(wire, monkeypatch):
    http, _runner = wire
    monkeypatch.delenv("AGENT_API_KEYS", raising=False)
    monkeypatch.delenv("AGENT_RATE_LIMIT_RPM", raising=False)
    assert http.post("/qa", json={"question": QUESTION}).status_code == 200
    assert http.post("/qa/stream", json={"question": QUESTION}).status_code == 200


def test_a_missing_key_is_rejected(wire, monkeypatch):
    http, _runner = wire
    monkeypatch.setenv("AGENT_API_KEYS", "alpha-key,beta-key")
    resp = http.post("/qa", json={"question": QUESTION})
    assert resp.status_code == 401
    assert resp.json() == {"detail": "API key 缺失或无效"}


def test_a_wrong_or_prefix_key_is_rejected(wire, monkeypatch):
    http, _runner = wire
    monkeypatch.setenv("AGENT_API_KEYS", "alpha-key,beta-key")
    for bad in ("nope", "alpha-ke", "beta-key-extra"):
        resp = http.post("/qa", json={"question": QUESTION}, headers={"X-API-Key": bad})
        assert resp.status_code == 401, bad
        assert resp.json()["detail"] == "API key 缺失或无效"


def test_the_right_key_reaches_the_endpoint(wire, monkeypatch):
    http, _runner = wire
    monkeypatch.setenv("AGENT_API_KEYS", "gamma-key,delta-key")
    resp = http.post("/qa", json={"question": QUESTION}, headers={"X-API-Key": "delta-key"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "答案正文"


def test_health_is_exempt_from_the_key(wire, monkeypatch):
    http, _runner = wire
    monkeypatch.setenv("AGENT_API_KEYS", "epsilon-key")
    assert http.get("/health").status_code == 200
    assert http.get("/laws").status_code == 404


def test_the_rate_limit_counts_per_key_within_a_minute(wire, monkeypatch):
    http, _runner = wire
    monkeypatch.setenv("AGENT_API_KEYS", "limit-key")
    monkeypatch.setenv("AGENT_RATE_LIMIT_RPM", "3")
    headers = {"X-API-Key": "limit-key"}
    codes = [
        http.post("/qa", json={"question": QUESTION}, headers=headers).status_code
        for _ in range(5)
    ]
    assert codes == [200, 200, 200, 429, 429]
    blocked = http.post("/qa", json={"question": QUESTION}, headers=headers)
    assert blocked.headers["Retry-After"].isdigit()
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60
    assert blocked.json()["detail"] == "请求过于频繁：每分钟上限 3 次"


def test_the_limit_does_not_leak_across_keys(wire, monkeypatch):
    http, _runner = wire
    monkeypatch.setenv("AGENT_API_KEYS", "busy-key,calm-key")
    monkeypatch.setenv("AGENT_RATE_LIMIT_RPM", "2")
    busy = {"X-API-Key": "busy-key"}
    for _ in range(2):
        assert http.post("/qa", json={"question": QUESTION}, headers=busy).status_code == 200
    assert http.post("/qa", json={"question": QUESTION}, headers=busy).status_code == 429
    calm = http.post("/qa", json={"question": QUESTION}, headers={"X-API-Key": "calm-key"})
    assert calm.status_code == 200


def test_zero_rpm_switches_the_limit_off(wire, monkeypatch):
    http, _runner = wire
    monkeypatch.setenv("AGENT_API_KEYS", "free-key")
    monkeypatch.setenv("AGENT_RATE_LIMIT_RPM", "0")
    headers = {"X-API-Key": "free-key"}
    for _ in range(5):
        assert http.post("/qa", json={"question": QUESTION}, headers=headers).status_code == 200


def test_the_stream_is_guarded_before_its_generator_starts(wire, monkeypatch):
    http, runner = wire
    monkeypatch.setenv("AGENT_API_KEYS", "stream-key")
    blocked = http.post("/qa/stream", json={"question": QUESTION})
    assert blocked.status_code == 401
    assert runner.stream_calls == 0
    allowed = http.post(
        "/qa/stream", json={"question": QUESTION}, headers={"X-API-Key": "stream-key"}
    )
    assert allowed.status_code == 200
    assert runner.stream_calls == 1


def test_a_non_ascii_presented_key_is_a_clean_mismatch() -> None:
    assert keys_match("秘钥", ("alpha-key",)) is False
    assert keys_match("秘钥", ("秘钥",)) is True


def test_a_non_ascii_key_header_is_refused_not_a_crash(wire, monkeypatch):
    http, _runner = wire
    monkeypatch.setenv("AGENT_API_KEYS", "alpha-key")

    resp = http.post(
        "/qa",
        json={"question": QUESTION},
        headers={"X-API-Key": b"\xe7\xa7\x98\xe9\x92\xa5"},
    )

    assert resp.status_code == 401
    assert resp.json() == {"detail": "API key 缺失或无效"}
