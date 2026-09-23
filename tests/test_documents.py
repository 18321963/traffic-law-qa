from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：上传端点验收跳过")

from fastapi.testclient import TestClient  # noqa: E402

from traffic_law_qa import config  # noqa: E402
from traffic_law_qa import main as server
from traffic_law_qa.agents import graph as graph_mod  # noqa: E402
from traffic_law_qa.agents.review import cited_labels  # noqa: E402
from traffic_law_qa.agents.trace import render_trace  # noqa: E402
from traffic_law_qa.contracts import ParentChunk, RetrievalResult, RetrievedArticle  # noqa: E402
from traffic_law_qa.database import init_db  # noqa: E402
from traffic_law_qa.database.models import STATUS_READY, STATUS_REJECTED  # noqa: E402
from traffic_law_qa.obs import Tracer  # noqa: E402
from traffic_law_qa.qa.generator import MATERIAL_HEADER, AnswerGenerator  # noqa: E402
from traffic_law_qa.ready import ReadyState  # noqa: E402
from traffic_law_qa.services.ingest import dry_run  # noqa: E402

LAW_NAME = "中华人民共和国道路交通安全法"
ARTICLE = ParentChunk(
    parent_id=f"{LAW_NAME}#第九十一条",
    law_id="road_traffic_safety",
    law_name=LAW_NAME,
    version="2021",
    citation=f"《{LAW_NAME}》",
    article_no="第九十一条",
    article_index=91,
    chapter="第七章 法律责任",
    section=None,
    text="饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证，并处一千元以上二千元以下罚款。",
)
PARENTS = {ARTICLE.parent_id: ARTICLE}

MATERIAL_MD = """# 某某公司车辆管理规定

第一条 本公司车辆保险由行政部统一办理，驾驶员不得自行指定保险公司。

第十条 培训费用按每人每年一千二百元包干，超出部分由所在部门承担。
"""

NOT_A_LAW_MD = """# 会议纪要

今天开了个会，讨论了明年的预算安排，大家都觉得要节约。

散会。
"""


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "documents.db")
    monkeypatch.setattr(config, "SOURCE_DIR", tmp_path / "docx")
    init_db.init_database()
    return tmp_path


@pytest.fixture
def client(store):
    server.app.state.rt = server.Runtime(
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
        rag=None,
        boot_ms=0.0,
    )
    server.app.state.boot_error = None
    return TestClient(server.app)


def _upload(client, text: str, name: str, mode: str):
    return client.post(
        "/documents",
        files={"file": (name, text.encode("utf-8"), "text/markdown")},
        data={"mode": mode},
    )


def test_session_upload_is_listed_and_deletable(client) -> None:
    resp = _upload(client, MATERIAL_MD, "车辆管理规定.md", "session")
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True and body["mode"] == "session"

    listed = client.get("/documents", params={"mode": "session"}).json()
    assert [row["doc_id"] for row in listed["documents"]] == [body["doc_id"]]
    assert listed["documents"][0]["status"] == STATUS_READY

    assert client.delete(f"/documents/{body['doc_id']}").status_code == 200
    assert client.get("/documents").json()["count"] == 0
    assert not config.upload_dir(body["doc_id"]).exists()


def test_permanent_rejection_never_lands_in_the_source_dir(client) -> None:
    resp = _upload(client, NOT_A_LAW_MD, "会议纪要.md", "permanent")

    assert resp.status_code == 400
    body = resp.json()
    assert body["accepted"] is False and "没解析出任何条文" in body["note"]
    rows = client.get("/documents", params={"mode": "permanent"}).json()["documents"]
    assert [row["status"] for row in rows] == [STATUS_REJECTED]
    assert config.source_files() == []


def test_permanent_upload_is_renamed_to_a_dated_canonical_name(client) -> None:
    resp = _upload(client, MATERIAL_MD, "车辆管理规定.md", "permanent")

    assert resp.status_code == 200
    body = resp.json()
    assert body["file"].startswith("车辆管理规定_") and body["file"].endswith(".md")
    assert body["articles"] == 2
    assert body["display_name"] == "车辆管理规定.md"
    assert body["law_name"] == "车辆管理规定"
    assert body["citation"] == f"《车辆管理规定》({body['version']})"
    assert [path.name for path in config.source_files()] == [body["file"]]
    row = client.get("/documents").json()["documents"][0]
    assert row["version"] == body["version"] != "unknown"
    assert row["paragraphs"] == 3


def test_same_bytes_are_refused_the_second_time(client) -> None:
    first = _upload(client, MATERIAL_MD, "车辆管理规定.md", "permanent")
    assert first.status_code == 200
    again = _upload(client, MATERIAL_MD, "另一个名字.md", "permanent")
    assert again.status_code == 400
    assert "已经以" in again.json()["note"]
    assert len(config.source_files()) == 1


def test_permanent_rows_refuse_deletion(client) -> None:
    body = _upload(client, MATERIAL_MD, "车辆管理规定.md", "permanent").json()
    resp = client.delete(f"/documents/{body['doc_id']}")
    assert resp.status_code == 400
    assert "pipeline build" in resp.json()["detail"]
    assert len(config.source_files()) == 1


def test_unknown_mode_is_refused(client) -> None:
    resp = _upload(client, MATERIAL_MD, "x.md", "temporary")
    assert resp.status_code == 400
    assert "mode" in resp.json()["detail"]


def test_dry_run_refuses_unreadable_suffix() -> None:
    plan, note = dry_run(b"%PDF-1.7", "条例.pdf")
    assert plan is None and ".pdf" not in note and "docx" in note


def test_documents_endpoint_needs_a_live_service(store) -> None:
    server.app.state.rt = None
    server.app.state.boot_error = "Milvus 连不上"
    resp = TestClient(server.app).get("/documents")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Milvus 连不上"


class _ScriptedLLM:
    def __init__(self, replies: list[dict], *, model: str) -> None:
        self.cfg = config.LLMConfig(base_url="http://stub", api_key="stub", model=model)
        self._replies = list(replies)
        self.offered: list[set[str]] = []

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages, *, tools=None, temperature=None, name="llm.chat"):
        self.offered.append({tool["function"]["name"] for tool in tools or ()})
        return self._replies.pop(0), {"total_tokens": 7}


def _call(call_id: str, name: str, **arguments) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class _FakeRAG:
    top_k = 6

    def __init__(self, generator: AnswerGenerator) -> None:
        self.parents = PARENTS
        self._generator = generator

    def search(self, query, top_k=None, *, channel_debug=False, law_filter=()) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            articles=(RetrievedArticle(article=ARTICLE, score=0.9),),
            used_vector=False,
            used_bm25=True,
            elapsed_ms=1.0,
        )

    def expand(self, text: str) -> str:
        return text

    def answer(self, question, retrieval, *, timeliness=(), materials=()):
        return self._generator.generate(
            question, retrieval, timeliness=timeliness, materials=materials
        )


def _generator_citing_materials(monkeypatch, generator: AnswerGenerator) -> None:
    def fake_call_llm(question, evidences, *, timeliness=(), materials=()):
        prompt = generator.build_prompt(question, evidences, timeliness, materials)
        cited = ""
        if materials and MATERIAL_HEADER in prompt and materials[0].label in prompt:
            cited = f"培训费按每人每年一千二百元包干。[{materials[0].label[1:-1]}]"
        return f"{cited}依《{LAW_NAME}》第九十一条。[依据1]", {"total_tokens": 11}

    monkeypatch.setattr(generator, "_call_llm", fake_call_llm)


def _write_material(text: str, doc_id: str = "d0", name: str = "车辆管理规定.md") -> str:
    directory = config.upload_dir(doc_id)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [{"index": i, "text": block} for i, block in enumerate(text.split("\n\n")) if block.strip()]
    with (directory / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (directory / "meta.json").write_text(
        json.dumps({"doc_id": doc_id, "display_name": name, "paragraphs": len(rows)}, ensure_ascii=False),
        encoding="utf-8",
    )
    return doc_id


def _runner(replies: list[dict], generator: AnswerGenerator):
    llm = _ScriptedLLM(replies, model="stub-agent")
    runner = graph_mod.AgentRunner(
        _FakeRAG(generator),
        llm=llm,
        region_llm=_ScriptedLLM([{"role": "assistant", "content": '{"region": "?"}'}], model="stub-region"),
        review_llm=_ScriptedLLM(
            [{"role": "assistant", "content": json.dumps({"judgments": [{"n": 1, "supported": True}]})}],
            model="stub-review",
        ),
        tracer=Tracer(),
    )
    return runner, llm


def test_material_reaches_the_answer_prompt_and_trace(store, monkeypatch) -> None:
    doc_id = _write_material(MATERIAL_MD)
    generator = AnswerGenerator(
        config.LLMConfig(base_url="http://stub", api_key="stub", model="stub-gen")
    )
    _generator_citing_materials(monkeypatch, generator)
    runner, llm = _runner(
        [
            _call("c1", "search_materials", query="培训费用怎么算"),
            _call("c2", "search_law", query="饮酒驾驶怎么处罚"),
            {"role": "assistant", "content": "够了"},
        ],
        generator,
    )

    state = runner.invoke("我们公司培训费怎么算", material_ids=[doc_id])
    answer = state["answer"]

    assert llm.offered == [{"search_law", "get_article", "search_materials"}] * 3
    assert [m.display_name for m in answer.materials] == ["车辆管理规定.md"]
    assert answer.materials[0].label == "[材料3]"
    assert answer.materials[0].index == 2
    assert "一千二百元" in answer.materials[0].text
    assert "[材料3]" in answer.text and "[依据1]" in answer.text
    assert cited_labels(answer.text) == [1]
    assert answer.review is not None and answer.review.passed
    assert "1 次材料检索（1 段）" in "".join(answer.notes)

    out = render_trace(state, color=True)
    assert "材料#3" in out
    assert "未取到" not in out
    assert "\033[33m" not in out


def test_materials_are_not_loaded_without_a_doc_id(store, monkeypatch) -> None:
    _write_material(MATERIAL_MD)
    generator = AnswerGenerator(
        config.LLMConfig(base_url="http://stub", api_key="stub", model="stub-gen")
    )
    _generator_citing_materials(monkeypatch, generator)
    runner, llm = _runner(
        [
            _call("c1", "search_materials", query="培训费用怎么算"),
            _call("c2", "search_law", query="饮酒驾驶怎么处罚"),
            {"role": "assistant", "content": "够了"},
        ],
        generator,
    )

    state = runner.invoke("我们公司培训费怎么算")

    assert state["answer"].materials == ()
    assert state["messages"][1]["content"].startswith("本次会话没有上传材料")


def test_repeat_material_search_does_not_duplicate_the_label(store, monkeypatch) -> None:
    doc_id = _write_material(MATERIAL_MD)
    generator = AnswerGenerator(
        config.LLMConfig(base_url="http://stub", api_key="stub", model="stub-gen")
    )
    _generator_citing_materials(monkeypatch, generator)
    runner, llm = _runner(
        [
            _call("c1", "search_materials", query="培训费用怎么算"),
            _call("c2", "search_materials", query="培训费用怎么算"),
            _call("c3", "search_law", query="饮酒驾驶怎么处罚"),
            {"role": "assistant", "content": "够了"},
        ],
        generator,
    )

    state = runner.invoke("我们公司培训费怎么算", material_ids=[doc_id])

    assert [m.label for m in state["answer"].materials] == ["[材料3]"]
    assert "材料#3" in state["messages"][1]["content"]
    assert state["messages"][3]["content"].startswith("⚠ 这些段上一轮")
    assert "材料#3" in state["messages"][3]["content"]


def test_ledger_keeps_the_sha1_and_the_original_name(client) -> None:
    _upload(client, MATERIAL_MD, "车辆管理规定.md", "session")

    got = init_db.list_documents()[0]
    assert got.sha1 == hashlib.sha1(MATERIAL_MD.encode("utf-8")).hexdigest()
    assert got.display_name == "车辆管理规定.md"
    assert got.bytes == len(MATERIAL_MD.encode("utf-8"))
