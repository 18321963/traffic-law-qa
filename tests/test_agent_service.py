from __future__ import annotations

import json

import pytest

from rag_contracts import config
from rag_contracts.domain.answer import Question
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.retrieval import (
    MATERIAL_TOP_K_DEFAULT,
    RetrievalResult,
    RetrievedArticle,
)
from rag_contracts.observability.tracer import Tracer
from rag_contracts.ports import TRUNCATED_FINISH_REASON

pytest.importorskip("langgraph", reason='agent 图要 pip install -e ".[agent]"')

from conftest import (  # noqa: E402
    ARTICLE,
    MATERIALS,
    NOTES,
    OTHER,
    PENALTY,
    ROAD,
    AgentStubLLM,
    AgentStubService,
)
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent_service.agents import region as region_mod  # noqa: E402
from agent_service.agents.graph import AgentRunner  # noqa: E402
from agent_service.agents.nodes import make_agent_node, make_tools_node  # noqa: E402
from agent_service.merge import merge_retrievals  # noqa: E402
from agent_service.prompts import REVIEW_DOWNGRADE_ANSWER  # noqa: E402
from agent_service.trace import render_trace  # noqa: E402

ROAD_ID = "road_traffic_safety"
PENALTY_ID = "shenzhen_penalty"
QUESTION = "在深圳，醉酒驾驶机动车怎么处罚？"
NATIONAL_NOTE = "这题涉及「深圳」：按全国法作答（未叠加地方性法规）"


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _call(call_id: str, name: str, **arguments) -> dict:
    return _tool_call(call_id, name, arguments)


def _region(text: str, place: str | None = None) -> list[dict]:
    payload = {"region": text} if place is None else {"region": text, "place": place}
    return [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]


def _review(*judgments: bool) -> list[dict]:
    payload = {"judgments": [{"n": n, "supported": ok} for n, ok in enumerate(judgments, start=1)]}
    return [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]


PLAN_THEN_ANSWER = [
    {"role": "assistant", "content": "", "tool_calls": [_call("c1", "search_law", query="醉酒驾驶机动车怎么处罚")]},
    {"role": "assistant", "content": "够了"},
]

MATERIAL_THEN_ANSWER = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            _tool_call("c1", "search_materials", {"query": "培训费"}),
            _tool_call("c2", "search_law", {"query": QUESTION}),
        ],
    },
    {"role": "assistant", "content": "够了"},
]


def _retrieval(query: str, articles, *, notes=()) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        articles=tuple(RetrievedArticle(article=item, score=0.5) for item in articles),
        used_vector=True,
        used_bm25=False,
        elapsed_ms=1.25,
        notes=tuple(notes),
        matched_text=query,
    )


def _state(*calls, **extra) -> dict:
    state = {
        "question": QUESTION,
        "history": [],
        "top_k": 6,
        "messages": [{"role": "assistant", "content": "", "tool_calls": list(calls)}],
        "search_log": [],
        "external": [],
        "materials": [],
        "material_ids": [],
        "region_scope": (),
    }
    state.update(extra)
    return state


def _tools_node(client):
    return make_tools_node(client, config.AgentConfig(max_steps=2), observer=Tracer())


def _runner(
    *,
    client=None,
    plan=(),
    region="深圳",
    place=None,
    llm=None,
    review_llm=None,
    finishes=(),
    tokens=7,
    cfg=None,
    sessions=None,
    clarify=False,
):
    client = client if client is not None else AgentStubService()
    runner = AgentRunner(
        client,
        llm=llm
        if llm is not None
        else AgentStubLLM(list(plan), model="stub-agent", finishes=finishes, tokens=tokens),
        region_llm=AgentStubLLM(_region(region, place), model="stub-region"),
        review_llm=review_llm
        if review_llm is not None
        else AgentStubLLM(_review(True), model="stub-review"),
        cfg=cfg or config.AgentConfig(max_steps=2, clarify=clarify),
        tracer=Tracer(),
        sessions=sessions,
    )
    return runner, client


def _review_runner(content: str, *, finish: str = "stop"):
    return _runner(
        plan=PLAN_THEN_ANSWER,
        review_llm=AgentStubLLM(
            [{"role": "assistant", "content": content}],
            model="stub-review",
            finishes=(finish,),
        ),
    )[0]


class _StubGraph:
    def __init__(self, chunks: list) -> None:
        self._chunks = chunks

    def stream(self, initial, config=None, stream_mode=None):
        yield from self._chunks


def test_the_region_node_takes_its_law_table_from_the_service() -> None:
    client = AgentStubService()
    llm = AgentStubLLM(_region("深圳"))

    update = region_mod.make_region_node(llm, client.laws())(_state())

    assert update["region"] == "深圳"
    assert update["region_scope"] == (ROAD_ID, PENALTY_ID)
    system = llm.prompts[0][0]["content"]
    assert ROAD in system and PENALTY in system


def test_the_agent_prompt_lists_the_laws_the_service_reported() -> None:
    client = AgentStubService()
    llm = AgentStubLLM([{"role": "assistant", "content": "够了"}])

    node = make_agent_node(llm, config.AgentConfig(max_steps=2), client.laws())
    node(_state(search_log=[{"articles": [{"parent_id": ARTICLE.parent_id}]}]))

    system = llm.prompts[0][0]["content"]
    assert ROAD in system and PENALTY in system
    assert llm.prompts[0][1] == {"role": "user", "content": QUESTION}
    assert len(llm.prompts) == 1, "手上已有证据时不该再触发「零证据」的补一枪"


def test_an_unknown_law_name_never_reaches_the_search() -> None:
    client = AgentStubService()
    call = _call("c1", "search_law", query=QUESTION, law_name="深圳经济特区停车管理条例")

    update = _tools_node(client)(_state(call))

    assert client.searches == []
    text = update["messages"][0]["content"]
    assert "无法确定法规「深圳经济特区停车管理条例」" in text
    assert PENALTY in text


def test_a_known_law_name_narrows_the_search_to_its_id() -> None:
    client = AgentStubService()
    call = _call("c1", "search_law", query=QUESTION, law_name="道路交通安全法")

    _tools_node(client)(_state(call, region_scope=(ROAD_ID, PENALTY_ID)))

    assert client.searches[0]["law_filter"] == (ROAD_ID,)
    assert client.searches[0]["question"] == QUESTION
    assert client.searches[0]["top_k"] == 6


def test_the_materials_tool_forwards_the_session_doc_ids() -> None:
    client = AgentStubService()
    call = _call("c1", "search_materials", query="培训费")

    update = _tools_node(client)(_state(call, material_ids=["d7", "d8"]))

    assert client.material_calls == [("培训费", ("d7", "d8"), MATERIAL_TOP_K_DEFAULT)]
    assert [passage.label for passage in update["materials"]] == ["[材料1]"]
    assert not update["messages"][0]["content"].startswith("注意")


def test_no_session_doc_ids_means_no_materials() -> None:
    client = AgentStubService()
    call = _call("c1", "search_materials", query="培训费")

    update = _tools_node(client)(_state(call))

    assert client.material_calls == [("培训费", (), MATERIAL_TOP_K_DEFAULT)]
    assert update["materials"] == []


def test_a_repeated_material_call_says_nothing_is_new() -> None:
    client = AgentStubService()
    call = _call("c1", "search_materials", query="培训费")

    update = _tools_node(client)(_state(call, material_ids=["d0"], materials=list(MATERIALS)))

    assert update["materials"] == []
    assert update["messages"][0]["content"].startswith("注意：这些段上一轮已经给过你了")


def test_a_missing_article_comes_back_as_the_service_note() -> None:
    client = AgentStubService()
    call = _call("c1", "get_article", article_no="第九十九条")

    update = _tools_node(client)(_state(call))

    assert client.lookups == [("第九十九条", None)]
    assert update["messages"][0]["content"] == "库里没有任何一部法规有第 99 条。"
    assert update["search_log"] == []


def test_merge_takes_the_articles_straight_from_the_payload() -> None:
    first = _retrieval("第一轮", (ARTICLE, OTHER), notes=("口语对齐：醉驾 → 醉酒驾驶",))
    second = _retrieval("第二轮", (OTHER,))

    merged = merge_retrievals(
        [first.to_dict(), second.to_dict()], question=QUESTION, max_evidence=2
    )

    assert [hit.article.parent_id for hit in merged.articles] == [
        ARTICLE.parent_id,
        OTHER.parent_id,
    ]
    assert merged.articles[0].article.text == ARTICLE.text
    assert merged.articles[0].score == 0.5
    assert merged.query == QUESTION
    assert (merged.used_vector, merged.used_bm25) == (True, False)
    assert merged.elapsed_ms == 2.5
    assert merged.notes == (
        "检索#1：口语对齐：醉驾 → 醉酒驾驶",
        "共 2 轮工具调用，合并去重后 2 条",
    )


def test_describe_reports_agent_facts_without_corpus_stats() -> None:
    runner, _client = _runner(client=AgentStubService(top_k=4))

    text = runner.describe()

    assert runner.top_k == 4
    assert not hasattr(_client, "stats"), "这个假件就是 RagClient 的面，多出来的属性说明 agent 伸手了"
    assert [word for word in ("条法条", "个子块", "稠密通道") if word in text] == []
    assert "最多 2 轮" in text
    assert "search_law" in text and "search_materials" in text
    assert "stub-agent" in text and "stub-region" in text and "stub-review" in text


def test_the_fallback_asks_the_service_to_search_and_generate() -> None:
    runner, client = _runner(llm=AgentStubLLM(model="none", available=False))

    answer = runner.ask(Question(text=QUESTION, top_k=3))

    assert [call["question"] for call in client.searches] == [QUESTION]
    assert client.searches[0]["top_k"] == 3
    assert client.answers[0]["question"].text == QUESTION
    assert answer.text == "答案正文[依据1]"
    assert answer.model == "stub-model"


def test_a_truncated_planning_turn_leaves_a_note() -> None:
    runner, _client = _runner(plan=PLAN_THEN_ANSWER, finishes=("length", "stop"))

    state = runner.invoke(QUESTION)

    assert state["truncated"] == 1
    assert [note for note in state["answer"].notes if "截断" in note] == [
        "规划轮 1 轮输出被 max_tokens 截断，判定可能不完整"
    ]


def test_a_clean_run_leaves_no_cut_note() -> None:
    runner, _client = _runner(plan=PLAN_THEN_ANSWER, finishes=("stop", "stop"))

    state = runner.invoke(QUESTION)

    assert state["truncated"] == 0
    assert [note for note in state["answer"].notes if "截断" in note] == []


def test_only_the_second_call_of_a_nudged_turn_counts() -> None:
    runner, _client = _runner(
        plan=[
            {"role": "assistant", "content": "够了"},
            {"role": "assistant", "content": "够了"},
        ],
        finishes=("length", "stop"),
    )

    state = runner.invoke(QUESTION)

    assert len(runner.llm.offered) == 2
    assert state["truncated"] == 0


def test_stream_emits_each_node_then_the_answer() -> None:
    runner, _client = _runner(plan=PLAN_THEN_ANSWER)

    events = list(runner.stream(QUESTION))

    assert [kind for kind, _payload in events].count("answer") == 1
    assert events[-1][0] == "answer"
    steps = [payload for kind, payload in events if kind == "step"]
    assert [step["node"] for step in steps] == [
        "region",
        "agent",
        "tools",
        "agent",
        "finalize",
        "review",
    ]
    assert [step["turn"] for step in steps if step["node"] == "agent"] == [1, 2]
    assert steps[2]["retrievals"] == 1 and steps[2]["hits"] == 2
    assert steps[5]["passed"] is True


def test_the_streamed_answer_carries_the_review_rewrite() -> None:
    runner, _client = _runner(
        plan=PLAN_THEN_ANSWER, cfg=config.AgentConfig(max_steps=2, review_min_score=1.5)
    )

    _kind, payload = list(runner.stream(QUESTION))[-1]

    assert payload["answer"].startswith(REVIEW_DOWNGRADE_ANSWER)
    assert payload["review"]["passed"] is False


def test_stream_without_an_answer_raises() -> None:
    runner, _client = _runner(plan=PLAN_THEN_ANSWER)
    runner.graph = lambda: _StubGraph([("values", {"answer": None})])

    with pytest.raises(RuntimeError, match="未产出答案"):
        list(runner.stream(QUESTION))


def test_stream_without_a_key_falls_back_to_the_linear_path() -> None:
    runner, client = _runner(llm=AgentStubLLM(model="none", available=False))

    events = list(runner.stream(QUESTION))

    assert [kind for kind, _payload in events] == ["answer"]
    assert events[0][1]["answer"] == "答案正文[依据1]"
    assert [call["question"] for call in client.searches] == [QUESTION]


def test_a_truncated_review_says_it_was_cut() -> None:
    runner = _review_runner("被截断的半句判据", finish=TRUNCATED_FINISH_REASON)

    answer = runner.invoke(QUESTION)["answer"]

    assert "复核未完成：输出被 max_tokens 截断，本次未拦截" in answer.notes


def test_a_parseable_review_stays_quiet_about_the_cut() -> None:
    runner = _review_runner(
        json.dumps({"judgments": [{"n": 1, "supported": True}]}),
        finish=TRUNCATED_FINISH_REASON,
    )

    answer = runner.invoke(QUESTION)["answer"]

    assert answer.review is not None and answer.review.passed
    assert [note for note in answer.notes if "复核未完成" in note] == []


def test_an_unparsed_review_still_blames_the_parsing() -> None:
    runner = _review_runner("不是 JSON")

    answer = runner.invoke(QUESTION)["answer"]

    assert "复核未完成：判据解析失败，本次未拦截" in answer.notes


def test_material_passages_reach_the_answer_call_and_the_trace() -> None:
    runner, client = _runner(plan=MATERIAL_THEN_ANSWER)

    state = runner.invoke(QUESTION, material_ids=["d0"])
    answer = state["answer"]

    assert client.material_calls == [("培训费", ("d0",), MATERIAL_TOP_K_DEFAULT)]
    assert [m.doc_id for m in client.answers[0]["materials"]] == ["d0"]
    assert [m.display_name for m in answer.materials] == ["车辆管理规定.md"]
    assert answer.materials[0].label == "[材料1]"
    assert "一千二百元" in answer.materials[0].text
    assert "1 次材料检索（1 段）" in "".join(answer.notes)
    assert answer.review is not None and answer.review.passed

    out = render_trace(state, color=True)
    assert "材料#1" in out
    assert "未取到" not in out
    assert "\033[33m" not in out


def test_a_budget_that_runs_out_still_reaches_an_answer() -> None:
    runner, _client = _runner(
        plan=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call("c1", "search_law", {"query": "饮酒驾驶怎么处罚"}),
                    _tool_call("c2", "get_article", {"article_no": "第九十九条"}),
                ],
            }
        ],
        cfg=config.AgentConfig(max_steps=1),
    )

    state = runner.invoke("饮酒驾驶怎么处罚")
    notes = "".join(state["answer"].notes)

    assert state["steps"] == 1
    assert "已达最大轮数 1，强制进入作答" in notes
    assert "规划轮 token 合计 7" in notes

    out = render_trace(state, color=True)
    assert "→ search_law" in out and "→ 命中 2 条" in out
    assert "未取到：库里没有任何一部法规有第 99 条。\033[0m" in out
    assert "\033[33m[agent] 已达最大轮数 1，强制进入作答\033[0m" in out
    assert "[agent] 规划轮 token 合计 7" in out
    assert "[agent] Agent：" in out


def test_merge_truncates_to_the_evidence_budget() -> None:
    first = _retrieval("第一轮", (ARTICLE, OTHER))

    merged = merge_retrievals([first.to_dict()], question=QUESTION, max_evidence=1)

    assert [hit.article.parent_id for hit in merged.articles] == [ARTICLE.parent_id]
    assert merged.notes == ("证据按排名截断至 1 条（合并后共 2 条）",)


def test_merge_notes_keep_the_retrieval_notes_verbatim() -> None:
    first = _retrieval("第一轮", (ARTICLE, OTHER), notes=NOTES)

    merged = merge_retrievals([first.to_dict()], question=QUESTION, max_evidence=2)

    assert merged.notes == tuple(f"检索#1：{note}" for note in NOTES)


class _NoStreamService:

    def __init__(self, inner: AgentStubService) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        if name == "answer_stream":
            raise AttributeError(name)
        return getattr(self._inner, name)


class _TruncatedStreamService(AgentStubService):

    def answer_stream(self, question, retrieval, *, timeliness=(), materials=()):
        yield "delta", "答案"


def test_the_draft_answer_streams_out_before_the_finalize_step() -> None:
    runner, client = _runner(plan=PLAN_THEN_ANSWER)

    events = list(runner.stream(QUESTION))

    assert [kind for kind, _payload in events] == [
        "step",
        "step",
        "step",
        "step",
        "delta",
        "delta",
        "step",
        "step",
        "answer",
    ]
    deltas = [payload["text"] for kind, payload in events if kind == "delta"]
    assert deltas == ["答案", "正文[依据1]"]
    assert "".join(deltas) == "答案正文[依据1]"
    assert events[6][1]["node"] == "finalize", "delta 应紧贴最后一次规划之后、finalize 定稿之前"
    done = events[-1]
    assert done[0] == "answer"
    assert done[1]["answer"] == "答案正文[依据1]"
    assert len(client.streams) == 1 and client.answers == []


def test_the_done_frame_stays_authoritative_when_the_review_downgrades() -> None:
    runner, _client = _runner(
        plan=PLAN_THEN_ANSWER, cfg=config.AgentConfig(max_steps=2, review_min_score=1.5)
    )

    events = list(runner.stream(QUESTION))

    deltas = [payload["text"] for kind, payload in events if kind == "delta"]
    assert "".join(deltas) == "答案正文[依据1]"
    done = events[-1]
    assert done[0] == "answer"
    assert done[1]["answer"].startswith(REVIEW_DOWNGRADE_ANSWER)
    assert done[1]["review"]["passed"] is False


def test_a_service_without_answer_stream_keeps_the_plain_answer_call() -> None:
    inner = AgentStubService()
    runner, _client = _runner(client=_NoStreamService(inner), plan=PLAN_THEN_ANSWER)

    events = list(runner.stream(QUESTION))

    assert [kind for kind, _payload in events] == [
        "step",
        "step",
        "step",
        "step",
        "step",
        "step",
        "answer",
    ]
    assert events[-1][1]["answer"] == "答案正文[依据1]"
    assert len(inner.answers) == 1 and inner.streams == []


def test_a_stream_that_ends_before_the_final_frame_raises() -> None:
    runner, _client = _runner(client=_TruncatedStreamService(), plan=PLAN_THEN_ANSWER)

    with pytest.raises(QaError, match="流里没有 answer 尾帧"):
        list(runner.stream(QUESTION))


def test_a_national_resume_notes_the_place_it_skipped() -> None:
    runner, _client = _runner(
        plan=PLAN_THEN_ANSWER, region="深圳", place="深圳", clarify=True, sessions=InMemorySaver()
    )

    first = list(runner.stream(QUESTION, session_id="s1"))
    assert [kind for kind, _payload in first] == ["step", "interrupt"]

    resumed = list(runner.resume_stream("s1", {"region": "national"}))

    assert resumed[-1][0] == "answer"
    assert NATIONAL_NOTE in resumed[-1][1]["notes"]


def test_a_national_verdict_without_a_place_gets_no_note() -> None:
    runner, _client = _runner(plan=PLAN_THEN_ANSWER, region="national", clarify=True)

    events = list(runner.stream(QUESTION))

    assert events[-1][0] == "answer"
    assert [note for note in events[-1][1]["notes"] if "按全国法作答" in note] == []


def test_the_national_note_also_lands_on_the_non_streaming_resume() -> None:
    runner, _client = _runner(
        plan=PLAN_THEN_ANSWER, region="深圳", place="深圳", clarify=True, sessions=InMemorySaver()
    )

    list(runner.stream(QUESTION, session_id="s1"))

    status, payload = runner.resume("s1", {"region": "national"})

    assert status == "ok"
    assert NATIONAL_NOTE in payload["notes"]
