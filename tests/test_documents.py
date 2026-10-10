from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：上传端点验收跳过")

from fastapi.testclient import TestClient  # noqa: E402

from rag_contracts import config  # noqa: E402
from rag_contracts.domain.answer import Question  # noqa: E402
from rag_contracts.domain.disk import ParentChunk  # noqa: E402
from rag_contracts.domain.reports import channel_label  # noqa: E402
from rag_contracts.domain.retrieval import RetrievalResult, RetrievedArticle  # noqa: E402
from rag_contracts.ports import TRUNCATED_FINISH_REASON  # noqa: E402
from rag_service.adapters.sqlite import (  # noqa: E402
    STATUS_READY,
    STATUS_REJECTED,
    init_database,
    list_documents,
)
from rag_service.api import app as server  # noqa: E402
from rag_service.api.runtime import Runtime  # noqa: E402
from rag_service.indexing.ingest import (  # noqa: E402
    MAX_BYTES,
    IngestPlan,
    dry_run,
    load_material_meta,
    promote,
    read_material,
)
from rag_service.indexing.parser import LawLibrary, ParseStage  # noqa: E402
from rag_service.indexing.readiness import ReadyState  # noqa: E402
from rag_service.indexing.sources import TextReader  # noqa: E402
from rag_service.prompts import MATERIAL_HEADER  # noqa: E402
from rag_service.query.generator import AnswerGenerator  # noqa: E402
from rag_service.query.materials import load_materials, search_materials  # noqa: E402

LAW_NAME = "中华人民共和国道路交通安全法"
QUESTION_TEXT = "饮酒后驾驶营运机动车怎么处罚"
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

MATERIAL_MD = """# 某某公司车辆管理规定

第一条 本公司车辆保险由行政部统一办理，驾驶员不得自行指定保险公司。

第十条 培训费用按每人每年一千二百元包干，超出部分由所在部门承担。
"""

NOT_A_LAW_MD = """# 会议纪要

今天开了个会，讨论了明年的预算安排，大家都觉得要节约。

散会。
"""

POISON_LAW_NAME = "../docx/中华人民共和国道路交通安全法"
POISON_MD = f"""\
{POISON_LAW_NAME}

第一条 为了维护道路交通秩序，预防和减少交通事故，保护人身安全，制定本法。

第二条 中华人民共和国境内的车辆驾驶人、行人、乘车人以及与道路交通活动有关的单位和个人，都应当遵守本法。
"""
POISON_MD_DOT = POISON_MD.replace("../docx/", "./", 1)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "documents.db")
    monkeypatch.setattr(config, "SOURCE_DIR", tmp_path / "docx")
    monkeypatch.setattr(config, "PDF_DIR", tmp_path / "pdf")
    (tmp_path / "pdf").mkdir()
    init_database()
    return tmp_path


@pytest.fixture
def client(store):
    server.app.state.rt = Runtime(
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


def _pdf_upload(client, data: bytes, name: str):
    return client.post(
        "/documents",
        files={"file": (name, data, "application/pdf")},
        data={"mode": "permanent"},
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


def test_a_file_with_no_paragraphs_is_refused_with_the_ingest_receipt(client) -> None:
    resp = _upload(client, "", "空文件.md", "session")

    assert resp.status_code == 400
    body = resp.json()
    assert body["accepted"] is False
    paragraphs, note = read_material(b"", "空文件.md")
    assert paragraphs is None and "没有可解析的段落" in note
    assert body["note"] == note, "路由另写了一份回执，不是上传判据那一份"
    rows = client.get("/documents", params={"mode": "session"}).json()["documents"]
    assert [row["status"] for row in rows] == [STATUS_REJECTED]
    assert not config.upload_dir(body["doc_id"]).exists()


def test_an_oversized_upload_is_refused_before_it_is_read(client) -> None:
    too_big = "x" * (MAX_BYTES + 1)

    for mode in ("session", "permanent"):
        resp = _upload(client, too_big, "超限.md", mode)
        assert resp.status_code == 400
        assert "超过" in resp.json()["detail"], mode

    assert client.get("/documents").json()["count"] == 0


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


def test_a_law_name_with_a_path_prefix_is_refused_before_it_touches_the_source_dir(client) -> None:
    for text in (POISON_MD, POISON_MD_DOT):
        resp = _upload(client, text, "_20210429.md", "permanent")
        assert resp.status_code == 400, resp.text
        assert "落盘文件名不安全" in resp.json()["note"]
        assert config.source_files() == []


def test_promote_refuses_a_plan_whose_filename_escapes_the_source_dir(store) -> None:
    plan = IngestPlan(
        filename="../escaped.md",
        display_name="escaped.md",
        suffix=".md",
        sha1="0" * 40,
        size=1,
        law_id="law_escaped",
        law_name="越界",
        version="2021-04-29",
        citation="《越界》(2021-04-29)",
        articles=1,
        chapters=0,
        sections=0,
        leftovers=0,
        paragraphs=1,
    )

    with pytest.raises(RuntimeError) as excinfo:
        promote(b"x", plan)

    assert "落盘路径越界" in str(excinfo.value)
    assert not (store / "escaped.md").exists()
    assert config.source_files() == []

    written = promote(b"x", replace(plan, filename="越界_20210429.md"))

    assert written.parent.resolve() == config.SOURCE_DIR.resolve()
    assert written.read_bytes() == b"x"


def test_permanent_rows_refuse_deletion(client) -> None:
    body = _upload(client, MATERIAL_MD, "车辆管理规定.md", "permanent").json()
    resp = client.delete(f"/documents/{body['doc_id']}")
    assert resp.status_code == 400
    assert "cli.build build" in resp.json()["detail"]
    assert len(config.source_files()) == 1


def test_unknown_mode_is_refused(client) -> None:
    resp = _upload(client, MATERIAL_MD, "x.md", "temporary")
    assert resp.status_code == 400
    assert "mode" in resp.json()["detail"]


def test_dry_run_refuses_unreadable_suffix() -> None:
    plan, note = dry_run(b"any bytes", "条例.doc")
    assert plan is None and ".doc" in note and "docx" in note


def test_dry_run_reports_a_pdf_that_has_no_readable_text_layer() -> None:
    plan, note = dry_run(b"%PDF-1.7", "条例.pdf")
    assert plan is None and "读不出内容" in note


PDF_LAW_PAGES = [
    ["2026/9/19 08:49 测试条例 _ 公安部 _ 中国政府网", "第一条 为了测试，制定本条例。"],
    ["2026/9/19 08:49 测试条例 _ 公安部 _ 中国政府网", "第二条 本条例所称测试，是指自动化测试。"],
    ["2026/9/19 08:49 测试条例 _ 公安部 _ 中国政府网", "第三条 本条例自2025年1月1日起施行。"],
]
PDF_LAW_PAGES_REVISED = [
    ["2026/9/19 08:49 测试条例 _ 公安部 _ 中国政府网", "第一条 为了测试，制定本条例（修订版）。"],
    ["2026/9/19 08:49 测试条例 _ 公安部 _ 中国政府网", "第二条 本条例所称测试，是指端到端的自动化测试。"],
    ["2026/9/19 08:49 测试条例 _ 公安部 _ 中国政府网", "第三条 本条例自2025年6月1日起施行。"],
]


def test_a_synthetic_pdf_passes_the_full_dry_run(store, pdf_maker) -> None:
    plan, note = dry_run(pdf_maker(PDF_LAW_PAGES), "测试条例_20250101.pdf")

    assert plan is not None
    assert plan.law_name == "测试条例" and plan.version == "2025-01-01"
    assert plan.articles == 3 and plan.filename == "测试条例_20250101.pdf"
    assert "可以入库" in note


def test_a_revised_copy_of_the_same_law_is_refused_by_law_id(client, pdf_maker) -> None:
    first = _pdf_upload(client, pdf_maker(PDF_LAW_PAGES), "测试条例_20250101.pdf")
    assert first.status_code == 200 and first.json()["file"] == "测试条例_20250101.pdf"

    again = _pdf_upload(client, pdf_maker(PDF_LAW_PAGES_REVISED), "测试条例_20250101.pdf")

    assert again.status_code == 400
    note = again.json()["note"]
    assert "同一部法规" in note and "测试条例_20250101.pdf" in note
    assert len(config.source_files()) == 1


def test_the_parse_stage_refuses_two_source_files_with_one_law_id(tmp_path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    for name in ("测试法规_20210101.md", "测试法规_20220202.md"):
        (source / name).write_text("第一条 为了测试，制定本规定。\n", encoding="utf-8")
    stage = ParseStage(
        source_dir=source,
        library=LawLibrary(parsed_dir=tmp_path / "parsed", text_dir=tmp_path / "text"),
        verbose=False,
    )

    with pytest.raises(ValueError) as excinfo:
        stage.run()

    message = str(excinfo.value)
    assert "测试法规_20210101.md" in message and "测试法规_20220202.md" in message
    assert "先删旧版本" in message


def test_documents_endpoint_needs_a_live_service(store) -> None:
    server.app.state.rt = None
    server.app.state.boot_error = "Milvus 连不上"
    resp = TestClient(server.app).get("/documents")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Milvus 连不上"


def _rebuilt(reason: str = "docx 与索引不一致") -> Runtime:
    return Runtime(
        ready=ReadyState(
            action="rebuild",
            reason=reason,
            rows=812,
            laws=6,
            articles=508,
            parents=508,
            dense=False,
            milvus="v2.6.24",
        ),
        rag=None,
        boot_ms=12.0,
    )


def test_reindex_rebuilds_and_replaces_the_runtime(client, monkeypatch) -> None:
    from rag_service.api import routes

    rebuilt = _rebuilt()
    calls = []
    monkeypatch.setattr(routes, "boot_runtime", lambda **kwargs: calls.append(kwargs) or rebuilt)

    resp = client.post("/reindex")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["action"] == "rebuild" and body["reason"] == "docx 与索引不一致"
    assert body["articles"] == 508
    assert body["channels"] == channel_label(False)
    assert server.app.state.rt is rebuilt
    assert calls == [{"rebuild": True}], "手动重建必须显式覆盖权重门，否则指纹对不上时它也会被拦"


def test_reindex_is_still_the_way_out_when_the_boot_failed(client, monkeypatch) -> None:
    from rag_service.api import routes

    server.app.state.rt = None
    server.app.state.boot_error = "权重与索引快照对不上"
    monkeypatch.setattr(routes, "boot_runtime", lambda **kwargs: _rebuilt("重建后指纹已刷新"))

    resp = client.post("/reindex")

    assert resp.status_code == 200, "全站 503 时 /reindex 是唯一出口，不能先要求有一个健康的 runtime"
    assert server.app.state.boot_error is None and server.app.state.rt is not None


class _ScriptedLLM:
    def __init__(self, replies: list[dict], *, model: str, finish: str = "stop") -> None:
        self.cfg = config.LLMConfig(base_url="http://stub", api_key="stub", model=model)
        self._replies = list(replies)
        self.finish = finish
        self.offered: list[set[str]] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self.cfg.model

    def chat(self, messages, *, tools=None, temperature=None, name="llm.chat"):
        self.offered.append({tool["function"]["name"] for tool in tools or ()})
        return self._replies.pop(0), {"total_tokens": 7}, self.finish


def _retrieval(query: str) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        articles=(RetrievedArticle(article=ARTICLE, score=0.9),),
        used_vector=False,
        used_bm25=True,
        elapsed_ms=1.0,
        matched_text=query,
    )


def _generator_citing_materials(monkeypatch, generator: AnswerGenerator) -> None:
    def fake_call_llm(question, evidences, *, timeliness=(), materials=()):
        prompt = generator.build_prompt(question, evidences, timeliness, materials)
        cited = ""
        if materials and MATERIAL_HEADER in prompt and materials[0].label in prompt:
            cited = f"培训费按每人每年一千二百元包干。[{materials[0].label[1:-1]}]"
        return f"{cited}依《{LAW_NAME}》第九十一条。[依据1]", {"total_tokens": 11}, "stop"

    monkeypatch.setattr(generator, "_call_llm", fake_call_llm)


def _write_material(text: str, doc_id: str = "d0d0d0d0d0d0", name: str = "车辆管理规定.md") -> str:
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


def _generate_once(finish: str):
    generator = AnswerGenerator(
        config.LLMConfig(base_url="http://stub", api_key="stub", model="stub-gen"),
        llm=_ScriptedLLM(
            [{"role": "assistant", "content": f"依《{LAW_NAME}》第九十一条。[依据1]"}],
            model="stub-gen",
            finish=finish,
        ),
    )
    return generator.generate(Question(text=QUESTION_TEXT), _retrieval(QUESTION_TEXT))


def test_a_truncated_answer_carries_the_cut_note() -> None:
    answer = _generate_once(TRUNCATED_FINISH_REASON)

    assert answer.text.endswith("[依据1]")
    assert [note for note in answer.notes if "截断" in note] == [
        "模型输出被 max_tokens 截断，答案可能不完整"
    ]


def test_a_complete_answer_carries_no_cut_note() -> None:
    assert [note for note in _generate_once("stop").notes if "截断" in note] == []


def test_the_material_section_reaches_the_generation_prompt(store, monkeypatch) -> None:
    doc_id = _write_material(MATERIAL_MD)
    generator = AnswerGenerator(
        config.LLMConfig(base_url="http://stub", api_key="stub", model="stub-gen")
    )
    _generator_citing_materials(monkeypatch, generator)
    passages, _note = search_materials("我们公司培训费怎么算", load_materials([doc_id]))

    answer = generator.generate(
        Question(text="我们公司培训费怎么算"), _retrieval("我们公司培训费怎么算"), materials=passages
    )

    assert len(answer.materials) == 3
    assert {m.display_name for m in answer.materials} == {"车辆管理规定.md"}
    assert answer.materials[0].label == "[材料3]"
    assert "一千二百元" in answer.materials[0].text
    assert "[材料3]" in answer.text, (
        "假生成器只在这两样都在 prompt 里时才吐 [材料3] —— 这句在，材料段就进了 prompt"
    )


def test_load_materials_skips_doc_ids_that_are_not_twelve_hex(store) -> None:
    _write_material(MATERIAL_MD, doc_id="../escaped")
    assert (store / "data" / "escaped" / "chunks.jsonl").exists()

    assert load_materials(["../escaped"]) == ()
    assert load_material_meta("../escaped") == {}
    valid = _write_material(MATERIAL_MD, doc_id="d0d0d0d0d0d0")
    assert len(load_materials([valid])) == 3


def test_materials_search_refuses_doc_ids_that_are_not_twelve_hex(client) -> None:
    for bad in ("../escaped", "..\\escaped", "d0", "", "D0D0D0D0D0D0"):
        resp = client.post("/materials/search", json={"query": "培训费", "doc_ids": [bad]})
        assert resp.status_code == 400, bad
        assert "doc_id 形状不对" in resp.json()["detail"]


def test_the_generation_prompt_has_no_material_section_without_passages() -> None:
    generator = AnswerGenerator(
        config.LLMConfig(base_url="http://stub", api_key="stub", model="stub-gen")
    )
    retrieval = _retrieval(QUESTION_TEXT)

    prompt = generator.build_prompt(
        Question(text=QUESTION_TEXT), generator.build_evidence(retrieval)
    )

    assert MATERIAL_HEADER not in prompt



def test_ledger_keeps_the_sha1_and_the_original_name(client) -> None:
    _upload(client, MATERIAL_MD, "车辆管理规定.md", "session")

    got = list_documents()[0]
    assert got.sha1 == hashlib.sha1(MATERIAL_MD.encode("utf-8")).hexdigest()
    assert got.display_name == "车辆管理规定.md"
    assert got.bytes == len(MATERIAL_MD.encode("utf-8"))


def test_text_reader_does_not_let_a_repeated_heading_double_the_articles(tmp_path) -> None:
    source = tmp_path / "law.md"
    source.write_text(
        "# 第一章 总则\n"
        "\n"
        "第一条 为了维护道路交通秩序，预防和减少交通事故，制定本法。\n"
        "\n"
        "#### 第九十条\n"
        "\n"
        "第九十条　机动车驾驶人违反道路通行规定的，处警告或者二十元以上二百元以下罚款。\n",
        encoding="utf-8",
    )

    assert [paragraph.text for paragraph in TextReader().read(source)] == [
        "第一章 总则",
        "第一条 为了维护道路交通秩序，预防和减少交通事故，制定本法。",
        "第九十条　机动车驾驶人违反道路通行规定的，处警告或者二十元以上二百元以下罚款。",
    ], (
        "`LawLibrary.save` 写出的 text/*.md 是这个形状：`#### 第九十条` 之后另起一段 `第九十条　正文…`，"
        "条号写了两遍；而 LawParser 的正文锚点是 `^第…条`，两行都命中 —— "
        "留着标题行会把条号与正文错开并让条数翻倍（实测 82 → 164），"
        "一个都不剥则标题行被并进上一条的正文尾部（实测每条正文尾部挂着下一行的 `#### 第X条`）。"
        "所以：剥掉 `#`，若下一段以它开头就连标题一起丢掉；用户自己写的 md 正文不重复条号时标题照常保留。"
    )
