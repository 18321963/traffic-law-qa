from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：agent 侧 HTTP 验收跳过")
pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")

from fastapi.testclient import TestClient  # noqa: E402

from agent_service.api import app as server  # noqa: E402
from rag_contracts import config  # noqa: E402

QUESTION = "在深圳，醉酒驾驶机动车怎么处罚？"
DEV_ORIGIN = "http://localhost:5173"
EVIL_ORIGIN = "http://evil.example"
DEFAULT_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


class _StubRagService:
    def health(self) -> dict:
        return {"status": "ok"}


class _StubRunner:
    def __init__(self) -> None:
        self.rag = _StubRagService()

    def ask_payload(self, question, *, material_ids, session_id):
        return "ok", {"question": question.text, "answer": "答案正文", "session_id": session_id}


@pytest.fixture
def cors_app(monkeypatch):
    def build(origins: str | None = None) -> TestClient:
        if origins is None:
            monkeypatch.delenv("CORS_ORIGINS", raising=False)
        else:
            monkeypatch.setenv("CORS_ORIGINS", origins)
        module = importlib.reload(server)
        module.app.state.runner = _StubRunner()
        module.app.state.boot_error = None
        return TestClient(module.app)

    yield build
    monkeypatch.undo()
    importlib.reload(server)


def _preflight(client, origin, *, method="POST", headers="content-type,X-API-Key"):
    request_headers = {"Origin": origin, "Access-Control-Request-Method": method}
    if headers is not None:
        request_headers["Access-Control-Request-Headers"] = headers
    return client.options("/qa", headers=request_headers)


def test_config_defaults_to_the_local_frontend_origins(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert config.cors_origins() == DEFAULT_ORIGINS
    monkeypatch.setenv("CORS_ORIGINS", "  ")
    assert config.cors_origins() == DEFAULT_ORIGINS


def test_config_reads_a_comma_separated_override(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://front.example:8080, http://other.example ,")
    assert config.cors_origins() == ("http://front.example:8080", "http://other.example")


def test_preflight_from_a_dev_origin_passes(cors_app, monkeypatch):
    monkeypatch.delenv("AGENT_API_KEYS", raising=False)
    http = cors_app()
    resp = _preflight(http, DEV_ORIGIN)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == DEV_ORIGIN
    assert "post" in resp.headers["access-control-allow-methods"].lower()
    assert "x-api-key" in resp.headers["access-control-allow-headers"].lower()


def test_preflight_from_an_unlisted_origin_is_refused(cors_app):
    http = cors_app()
    resp = _preflight(http, EVIL_ORIGIN)
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


def test_the_key_guard_sits_inside_the_cors_layer(cors_app, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEYS", "alpha-key")
    http = cors_app()
    preflight = _preflight(http, DEV_ORIGIN)
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == DEV_ORIGIN
    unauthorized = http.post("/qa", json={"question": QUESTION}, headers={"Origin": DEV_ORIGIN})
    assert unauthorized.status_code == 401
    assert unauthorized.headers["access-control-allow-origin"] == DEV_ORIGIN
    authorized = http.post(
        "/qa",
        json={"question": QUESTION},
        headers={"Origin": DEV_ORIGIN, "X-API-Key": "alpha-key"},
    )
    assert authorized.status_code == 200
    assert authorized.headers["access-control-allow-origin"] == DEV_ORIGIN


def test_a_custom_origin_list_reaches_the_middleware(cors_app):
    http = cors_app("http://front.example:8080")
    allowed = _preflight(http, "http://front.example:8080", headers=None)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://front.example:8080"
    foreign = _preflight(http, DEV_ORIGIN, headers=None)
    assert foreign.status_code == 400


def test_actual_responses_carry_the_allow_origin_header(cors_app, monkeypatch):
    monkeypatch.delenv("AGENT_API_KEYS", raising=False)
    http = cors_app()
    resp = http.post("/qa", json={"question": QUESTION}, headers={"Origin": DEV_ORIGIN})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == DEV_ORIGIN
    assert "origin" in resp.headers.get("vary", "").lower()
