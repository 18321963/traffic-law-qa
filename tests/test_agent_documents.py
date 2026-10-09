from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：agent 侧材料代理验收跳过")
pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")

from fastapi.testclient import TestClient  # noqa: E402

from agent_service.api import app as server  # noqa: E402
from agent_service.api import budget  # noqa: E402
from api_contracts import RagClient  # noqa: E402
from rag_contracts import config  # noqa: E402
from rag_contracts.domain.errors import QaError  # noqa: E402
from rag_service.adapters.sqlite import MODE_LABELS, init_database  # noqa: E402
from rag_service.api.routes import MISSING_DOC_NOTE, UNKNOWN_MODE_NOTE  # noqa: E402
from rag_service.indexing.ingest import read_material  # noqa: E402

QUESTION = "在深圳，醉酒驾驶机动车怎么处罚？"
BOOT_ERROR = "桩：agent 图没有起来"
MATERIAL_MD = """# 某某公司车辆管理规定

第一条 本公司车辆保险由行政部统一办理，驾驶员不得自行指定保险公司。

第十条 培训费用按每人每年一千二百元包干，超出部分由所在部门承担。
"""


@pytest.fixture
def wire(tmp_path, monkeypatch, agent_rag_server):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "documents.db")
    monkeypatch.setattr(config, "SOURCE_DIR", tmp_path / "docx")
    monkeypatch.setattr(config, "PDF_DIR", tmp_path / "pdf")
    (tmp_path / "pdf").mkdir()
    init_database()

    rag_app, _rt = agent_rag_server
    server.app.state.client = RagClient("http://testserver", client=TestClient(rag_app))
    server.app.state.runner = None
    server.app.state.boot_error = BOOT_ERROR
    return TestClient(server.app)


def _upload(http, text: str, name: str = "车辆管理规定.md", mode: str | None = None, headers=None):
    data = {} if mode is None else {"mode": mode}
    return http.post(
        "/documents",
        files={"file": (name, text.encode("utf-8"), "text/markdown")},
        data=data,
        headers=headers,
    )


def test_the_material_lifecycle_runs_end_to_end_through_the_agent(wire) -> None:
    http = wire

    accepted = _upload(http, MATERIAL_MD, mode="session")

    assert accepted.status_code == 200
    body = accepted.json()
    assert body["accepted"] is True and body["mode"] == "session"
    assert body["display_name"] == "车辆管理规定.md" and body["paragraphs"] == 3

    listed = http.get("/documents").json()
    assert listed["count"] == 1
    assert [row["doc_id"] for row in listed["documents"]] == [body["doc_id"]]

    removed = http.delete(f"/documents/{body['doc_id']}")
    assert removed.status_code == 200
    assert removed.json()["removed"] == body["doc_id"]
    assert http.get("/documents").json()["count"] == 0


def test_a_rejected_upload_keeps_the_rag_receipt_and_its_400(wire) -> None:
    http = wire

    resp = _upload(http, "", "空文件.md", mode="session")

    assert resp.status_code == 400, "业务的 400 不许被翻成 503"
    body = resp.json()
    assert body["accepted"] is False and body["mode"] == "session"
    _paragraphs, note = read_material(b"", "空文件.md")
    assert body["note"] == note, "回执被代理层改写了，不是上传判据那一份"


def test_the_default_mode_is_left_to_the_rag_side_as_session(wire) -> None:
    http = wire

    resp = _upload(http, MATERIAL_MD)

    assert resp.status_code == 200
    assert resp.json()["mode"] == "session"


def test_the_mode_really_travels_to_the_rag_side(wire) -> None:
    http = wire

    session = _upload(http, MATERIAL_MD, mode="session")
    assert session.status_code == 200
    assert session.json()["mode"] == "session" and "file" not in session.json()

    permanent = _upload(http, MATERIAL_MD, mode="permanent")
    assert permanent.status_code == 200
    assert permanent.json()["mode"] == "permanent"
    assert permanent.json()["law_name"] == "车辆管理规定"
    assert permanent.json()["file"].endswith(".md")


def test_an_unknown_mode_is_refused_with_the_rag_sides_detail(wire) -> None:
    http = wire

    resp = _upload(http, MATERIAL_MD, mode="temporary")

    assert resp.status_code == 400
    assert resp.json()["detail"] == UNKNOWN_MODE_NOTE.format(modes="、".join(MODE_LABELS))


def test_deleting_an_unknown_document_keeps_the_rag_404(wire) -> None:
    http = wire

    resp = http.delete("/documents/nope")

    assert resp.status_code == 404, "404 不许被翻成 503"
    assert resp.json()["detail"] == MISSING_DOC_NOTE.format(doc_id="nope")


def test_the_material_face_does_not_need_the_agent_graph(wire) -> None:
    http = wire

    blocked = http.post("/qa", json={"question": QUESTION})
    assert blocked.status_code == 503 and blocked.json()["detail"] == BOOT_ERROR

    assert _upload(http, MATERIAL_MD).status_code == 200


def test_the_material_face_is_behind_the_same_key(wire, monkeypatch) -> None:
    http = wire
    monkeypatch.setenv("AGENT_API_KEYS", "docs-key")
    headers = {"X-API-Key": "docs-key"}

    assert _upload(http, MATERIAL_MD).status_code == 401
    assert http.get("/documents").status_code == 401
    assert http.delete("/documents/x").status_code == 401

    accepted = _upload(http, MATERIAL_MD, mode="session", headers=headers)
    assert accepted.status_code == 200
    doc_id = accepted.json()["doc_id"]
    assert http.get("/documents", headers=headers).json()["count"] == 1
    assert http.delete(f"/documents/{doc_id}", headers=headers).status_code == 200


def test_the_daily_budget_gate_does_not_block_material_uploads(wire, monkeypatch) -> None:
    http = wire
    server.app.state.runner = object()
    monkeypatch.setattr(budget, "exhausted", lambda *_args, **_kwargs: (10, 10))

    assert http.post("/qa", json={"question": QUESTION}).status_code == 429
    assert _upload(http, MATERIAL_MD).status_code == 200


def test_a_broken_rag_transport_turns_into_503(wire, monkeypatch) -> None:
    http = wire
    client = server.app.state.client

    def _boom(*_args, **_kwargs):
        raise QaError("POST /documents 连不上 http://testserver：Connection refused")

    monkeypatch.setattr(client, "upload_document", _boom)

    resp = _upload(http, MATERIAL_MD)

    assert resp.status_code == 503
    assert "连不上" in resp.json()["detail"]
