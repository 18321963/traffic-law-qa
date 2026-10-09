from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：服务端工具面验收跳过")
pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")

from conftest import ARTICLE, NOTES  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from agent_service.tools import handlers  # noqa: E402
from agent_service.tools.handlers import ToolCall, ToolEnv  # noqa: E402
from agent_service.tools.render import render_tool_result  # noqa: E402
from api_contracts import to_passages, to_retrieval  # noqa: E402
from rag_contracts import config  # noqa: E402
from rag_contracts.domain.answer import Question  # noqa: E402
from rag_contracts.observability.tracer import Tracer  # noqa: E402

ARTICLE_TEXT = ARTICLE.text

FROZEN_SEARCH_LAW = (
    "检索#1「醉驾怎么处罚」｜命中 2 条（新增 2 / 已知 0）\n"
    "【1】《中华人民共和国道路交通安全法》第九十一条\n"
    "    醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不"
    "得重新取得机动车驾驶证。饮酒后驾驶营运机动车的，处十五日拘留，并处五千元罚款，吊销机动车驾驶证，五年内不"
    "得重新取得机动车驾驶证。因饮酒后驾驶机动车…\n"
    "    相关条：第九十条\n"
    "【2】《深圳经济特区道路交通安全违法行为处罚条例》第九十条\n"
    "    驾驶电动自行车未佩戴安全头盔的，处警告或者五十元罚款。\n"
    "提示：口语对齐：醉驾 → 醉酒驾驶"
)
FROZEN_GET_ARTICLE = (
    "检索#1「第九十一条」｜命中 1 条（新增 1 / 已知 0）\n"
    "【1】《中华人民共和国道路交通安全法》第九十一条\n"
    "    醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不"
    "得重新取得机动车驾驶证。饮酒后驾驶营运机动车的，处十五日拘留，并处五千元罚款，吊销机动车驾驶证，五年内不"
    "得重新取得机动车驾驶证。因饮酒后驾驶机动车被处罚，再次饮酒后驾驶机动车的，处十日以下拘留，并处一千元以上"
    "二千元以下罚款。\n"
    "    相关条：第九十条\n"
    "提示：精确取条，未走向量/BM25 检索"
)
FROZEN_MATERIALS = (
    "材料#1「培训费」｜命中 1 段（共 2 段，1 份材料）\n"
    "[材料1] 车辆管理规定.md 第 0 段\n"
    "培训费用按每人每年一千二百元包干，超出部分由所在部门承担。\n"
    "（会话材料，未入知识库 —— 只能标 [材料N]，标成 [依据N] 会让整篇答案作废）"
)


@pytest.fixture
def parity(agent_wire):
    remote, rt, app = agent_wire
    return TestClient(app), remote, rt


def _call(**arguments) -> ToolCall:
    doc_ids = arguments.pop("doc_ids", ())
    return ToolCall(
        arguments=json.dumps(arguments, ensure_ascii=False),
        retrieval_no=1,
        web_no=1,
        seen=frozenset(),
        seen_labels=frozenset(),
        region_scope=(),
        default_top_k=6,
        doc_ids=tuple(doc_ids),
    )


def _env(remote) -> ToolEnv:
    return ToolEnv(rag=remote, cfg=config.agent_config(), observer=Tracer())


def _rerender(payload: dict, call: ToolCall, *, snippet_chars: int, match_text: str) -> str:
    result = to_retrieval(payload)
    return render_tool_result(
        result,
        index=call.retrieval_no,
        seen=call.seen,
        snippet_chars=snippet_chars,
        match_text=match_text,
    )


def test_search_text_matches_the_text_frozen_before_the_split(parity):
    _, remote, _rt = parity
    env = _env(remote)
    call = _call(query="醉驾怎么处罚")

    outcome = handlers.search_law(env, call)

    assert outcome.text == FROZEN_SEARCH_LAW, (
        "工具渲染文本与 2026-09-28 出包前冻结的那份不一致：要么迁移引入了差异，要么渲染口径变了"
    )
    assert "口语对齐：醉驾 → 醉酒驾驶" in outcome.text
    assert "法名线索" not in outcome.text and "未重排" not in outcome.text
    assert len(ARTICLE_TEXT) > env.cfg.snippet_chars
    assert "…" in outcome.text


def test_a_law_name_lands_as_the_law_filter_the_service_can_use(parity):
    _, remote, rt = parity
    call = _call(query="头盔", law_name="道路交通安全法")

    outcome = handlers.search_law(_env(remote), call)

    assert rt.rag.searches[-1]["question"] == "头盔"
    assert rt.rag.searches[-1]["law_filter"] == ("road_traffic_safety",)
    assert "《中华人民共和国道路交通安全法》第九十一条" in outcome.text


def test_an_unknown_law_name_never_reaches_the_search(parity):
    _, remote, rt = parity
    before = len(rt.rag.searches)

    outcome = handlers.search_law(_env(remote), _call(query="头盔", law_name="中华人民共和国航空法"))

    assert "库内只有这几部" in outcome.text
    assert "中华人民共和国道路交通安全法" in outcome.text
    assert len(rt.rag.searches) == before, "法名不确定时一个字都不该发出去"


def test_get_article_text_matches_the_text_frozen_before_the_split(parity):
    _, remote, _rt = parity
    env = _env(remote)
    call = _call(article_no="第九十一条", law_name="道路交通安全法")

    payload = remote.article(article_no="第九十一条", law_name="道路交通安全法")
    outcome = handlers.get_article(env, call)

    assert payload["found"] is True and payload["note"] == ""
    assert outcome.text == FROZEN_GET_ARTICLE, (
        "工具渲染文本与 2026-09-28 出包前冻结的那份不一致：要么迁移引入了差异，要么渲染口径变了"
    )
    assert ARTICLE_TEXT in outcome.text
    assert (
        _rerender(payload, call, snippet_chars=env.cfg.article_chars, match_text="") == outcome.text
    )


def test_material_text_matches_the_text_frozen_before_the_split(parity):
    _, remote, _rt = parity
    call = _call(query="培训费", doc_ids=("d0d0d0d0d0d0",))

    outcome = handlers.search_materials(_env(remote), call)
    payload = remote.materials("培训费", ["d0d0d0d0d0d0"])

    assert outcome.text == FROZEN_MATERIALS, (
        "工具渲染文本与 2026-09-28 出包前冻结的那份不一致：要么迁移引入了差异，要么渲染口径变了"
    )
    assert to_passages(payload) == outcome.passages
    assert "[材料1]" in outcome.text and "一千二百元" in outcome.text


def test_a_missing_article_comes_back_as_a_note_not_a_status_code(parity):
    http, remote, _rt = parity
    local = handlers.get_article(_env(remote), _call(article_no="第九十九条"))

    resp = http.post("/articles/lookup", json={"article_no": "第九十九条"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["found"] is False and payload["articles"] == []
    assert payload["note"] == local.text
    assert "第 99 条" in payload["note"]


def test_the_payload_rebuilds_a_different_object_graph(parity):
    _, remote, rt = parity

    payload = remote.qa("醉驾怎么处罚", mode="search")
    rebuilt = to_retrieval(payload)

    hit = rebuilt.articles[0]
    assert hit.article.parent_id == ARTICLE.parent_id
    assert hit.article is not ARTICLE
    assert hit.article.text == ARTICLE.text and hit.article.refs == ARTICLE.refs
    assert hit.hit_chunks == ("c1", "c2") and hit.law_hint == "道路交通安全法"
    assert rebuilt.matched_text == "醉驾怎么处罚 醉酒驾驶"
    assert rebuilt.notes == NOTES
    assert {a.article.parent_id for a in rebuilt.articles} == set(rt.rag.parents)


def test_the_tools_send_their_knobs_over_the_wire(parity):
    _, remote, rt = parity
    rt.rag.searches.clear()

    remote.qa(
        "醉驾", mode="search", top_k=3, pool=50, debug=True, law_filter=["road_traffic_safety"]
    )

    assert rt.rag.searches == [
        {
            "question": "醉驾",
            "top_k": 50,
            "channel_debug": True,
            "law_filter": ("road_traffic_safety",),
            "candidates": 50,
        }
    ]


def test_the_endpoint_refuses_to_guess_when_nothing_was_named(parity):
    http, _, _ = parity
    resp = http.post("/articles/lookup", json={})
    assert resp.status_code == 400 and "text" in resp.json()["detail"]


def test_the_answer_endpoint_generates_from_the_retrieval_it_was_handed(parity):
    _, remote, rt = parity
    question = Question(text="醉驾怎么处罚", history=(("user", "上一轮问的是疲劳驾驶"),))
    retrieval = remote.search(question.text, 3)
    passages, _ = remote.search_materials("培训费", ["d0d0d0d0d0d0"])

    searches_before = len(rt.rag.searches)
    answer = remote.answer(question, retrieval, materials=passages)

    assert len(rt.rag.searches) == searches_before, (
        "生成侧又自己检索了一遍：请求里带过来的 retrieval 被丢了"
    )
    sent = rt.rag.answers[-1]
    assert sent["question"].text == "醉驾怎么处罚"
    assert sent["question"].history == (("user", "上一轮问的是疲劳驾驶"),)
    assert [hit.citation for hit in sent["retrieval"].articles] == retrieval.citations()
    assert [m.doc_id for m in sent["materials"]] == ["d0"]

    assert answer.text == "答案正文[依据1]"
    assert [e.citation for e in answer.evidences] == retrieval.citations()
    assert answer.evidences[0].article is None, "法条原文对象不该跟着载荷过线"
    assert answer.notes == ("桩生成器：没花真调用",)
    assert answer.retrieval is not None
    assert [hit.article.parent_id for hit in answer.retrieval.articles] == [
        hit.article.parent_id for hit in retrieval.articles
    ]


def test_laws_lists_what_the_search_side_can_filter_by(parity):
    _, remote, rt = parity

    laws = remote.laws()

    assert laws == rt.rag.laws()
    assert {law.law_id for law in laws} == {"road_traffic_safety", "shenzhen_penalty"}


def test_the_tools_never_build_their_own_transport():
    source = Path(handlers.__file__).read_text(encoding="utf-8")

    assert "httpx" not in source
    assert "RagClient(" not in source
