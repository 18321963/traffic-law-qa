from __future__ import annotations

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rag_contracts import config
from rag_contracts.domain.answer import Question
from rag_contracts.observability.tracer import Tracer

pytest.importorskip("langgraph", reason='agent 图要 pip install -e ".[agent]"')

from conftest import (  # noqa: E402
    PENALTY,
    ROAD,
    AgentStubLLM,
    AgentStubService,
)
from langgraph.checkpoint.base import empty_checkpoint  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from agent_service.agents import region as region_mod  # noqa: E402
from agent_service.agents import session as session_mod  # noqa: E402
from agent_service.agents.errors import ResumeConflict, SessionUnsupported  # noqa: E402
from agent_service.agents.graph import AgentRunner, make_run_config  # noqa: E402
from agent_service.agents.region import HISTORY_ANSWER_CHARS  # noqa: E402
from agent_service.agents.session import open_sessions  # noqa: E402
from agent_service.prompts import REVIEW_DOWNGRADE_ANSWER  # noqa: E402
from agent_service.storage import open_sessions_db  # noqa: E402

ROAD_ID = "road_traffic_safety"
PENALTY_ID = "shenzhen_penalty"
QUESTION = "在深圳，醉酒驾驶机动车怎么处罚？"
FOLLOW_UP = "那记分呢？"
REGION_GUARD = "\n\n只输出 JSON 对象（region / place 两个字段），不要回答问题本身。"

_MISSING = object()


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


PLAN = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [_tool_call("c1", "search_law", {"query": "醉酒驾驶机动车怎么处罚"})],
    },
    {"role": "assistant", "content": "够了"},
]


def _plan(turns: int) -> list[dict]:
    return [dict(message) for _ in range(turns) for message in PLAN]


def _region(region: str, place: str = "") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": json.dumps({"region": region, "place": place}, ensure_ascii=False),
        }
    ]


def _intake(switch: str = "continue", *, intent: str = "other", query: str = "", topic: str = "") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": json.dumps(
                {"switch": switch, "intent": intent, "query": query, "topic_id": topic},
                ensure_ascii=False,
            ),
        }
    ]


def _review(*judgments: bool) -> list[dict]:
    payload = {"judgments": [{"n": n, "supported": ok} for n, ok in enumerate(judgments, start=1)]}
    return [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]


def _runner(
    *,
    client=None,
    plan=(),
    region=("深圳", ""),
    llm=None,
    review_llm=None,
    intake_llm=None,
    sessions=_MISSING,
    laws=_MISSING,
    cfg=None,
    clarify=False,
    history_turns=3,
):
    client = client if client is not None else AgentStubService()
    if sessions is _MISSING:
        sessions = InMemorySaver()
    extra = {} if laws is _MISSING else {"laws": laws}
    runner = AgentRunner(
        client,
        llm=llm if llm is not None else AgentStubLLM(list(plan), model="stub-agent"),
        region_llm=AgentStubLLM(_region(*region), model="stub-region"),
        review_llm=review_llm
        if review_llm is not None
        else AgentStubLLM(_review(True) * 4, model="stub-review"),
        intake_llm=intake_llm,
        cfg=cfg or config.AgentConfig(max_steps=2, clarify=clarify, history_turns=history_turns),
        tracer=Tracer(),
        sessions=sessions,
        **extra,
    )
    return runner, client


def _values(runner: AgentRunner, sid: str) -> dict:
    return runner.graph().get_state({"configurable": {"thread_id": sid}}).values


class _LongAnswerService(AgentStubService):

    def _answer(self, question, retrieval, timeliness, materials):
        return replace(
            super()._answer(question, retrieval, timeliness, materials),
            text="答" * 400 + "[依据1]",
        )


class _FakeTask:

    def __init__(self, interrupts) -> None:
        self.interrupts = interrupts


class _FakeSnapshot:

    def __init__(self, tasks) -> None:
        self.tasks = tasks


def test_a_second_turn_on_one_session_counts_only_its_own_run() -> None:
    runner, _client = _runner(plan=_plan(2))

    first = runner.invoke(QUESTION, session_id="s1")
    second = runner.invoke(FOLLOW_UP, session_id="s1")

    assert first["steps"] == second["steps"] == 2
    assert [row["query"] for row in second["search_log"]] == ["醉酒驾驶机动车怎么处罚"]
    assert len(second["usage"]) == 2
    assert [entry["question"] for entry in second["conversation"]] == [QUESTION, FOLLOW_UP]
    assert second["answer"].text == "答案正文[依据1]"


def test_the_second_turn_prompts_carry_the_first_qa() -> None:
    runner, _client = _runner(plan=_plan(2))

    runner.invoke(QUESTION, session_id="s1")
    runner.invoke(FOLLOW_UP, session_id="s1")

    tagged = "（本轮按「深圳」的规定作答）答案正文[依据1]"
    turn2_region = runner.region_llm.prompts[1]
    assert {"role": "user", "content": QUESTION} in turn2_region
    assert {"role": "assistant", "content": tagged} in turn2_region
    assert turn2_region[-1] == {"role": "user", "content": FOLLOW_UP + REGION_GUARD}

    turn2_agent = runner.llm.prompts[2]
    assert {"role": "user", "content": QUESTION} in turn2_agent
    assert {"role": "assistant", "content": tagged} in turn2_agent


def test_a_first_turn_prompt_is_still_system_plus_question() -> None:
    runner, _client = _runner(plan=PLAN)

    runner.invoke(QUESTION, session_id="s1")

    prompt = runner.region_llm.prompts[0]
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[-1] == {"role": "user", "content": QUESTION + REGION_GUARD}


def test_input_history_is_normalized_into_the_first_turn_prompt() -> None:
    runner, _client = _runner(plan=PLAN)
    query = Question(text=FOLLOW_UP, history=[("user", "醉驾怎么罚"), ("assistant", "答案A")])

    runner.invoke(query, session_id="s1")

    prompt = runner.region_llm.prompts[0]
    assert {"role": "user", "content": "醉驾怎么罚"} in prompt
    assert {"role": "assistant", "content": "答案A"} in prompt
    assert prompt[-1] == {"role": "user", "content": FOLLOW_UP + REGION_GUARD}


def test_the_history_snippet_clips_long_answers() -> None:
    runner, _client = _runner(plan=_plan(2), client=_LongAnswerService())

    runner.invoke(QUESTION, session_id="s1")
    runner.invoke(FOLLOW_UP, session_id="s1")

    tag = "（本轮按「深圳」的规定作答）"
    assistant = [m["content"] for m in runner.region_llm.prompts[1] if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0].startswith(tag)
    assert assistant[0].endswith("…")
    assert len(assistant[0]) == len(tag) + HISTORY_ANSWER_CHARS + 1


def test_two_sessions_do_not_bleed() -> None:
    runner, _client = _runner(plan=_plan(2))

    runner.invoke(QUESTION, session_id="a")
    second = runner.invoke(FOLLOW_UP, session_id="b")

    assert [entry["question"] for entry in second["conversation"]] == [FOLLOW_UP]
    assert [entry["question"] for entry in _values(runner, "a")["conversation"]] == [QUESTION]


def test_make_run_config_is_the_only_config_builder() -> None:
    assert make_run_config(None, sessions=None) == {"config": {}}
    with pytest.raises(SessionUnsupported):
        make_run_config("s1", sessions=None)

    saver = InMemorySaver()
    built = make_run_config(None, sessions=saver)
    assert built["durability"] == "sync"
    assert len(built["config"]["configurable"]["thread_id"]) == 32
    named = make_run_config("s7", sessions=saver)
    assert named["config"]["configurable"]["thread_id"] == "s7"


def test_a_sessionless_runner_rejects_a_session_id() -> None:
    runner, _client = _runner(sessions=None, plan=PLAN)

    with pytest.raises(SessionUnsupported):
        runner.invoke(QUESTION, session_id="s1")
    with pytest.raises(SessionUnsupported):
        runner.ask_payload(QUESTION, session_id="s1")
    with pytest.raises(SessionUnsupported):
        runner.resume("s1", {"region": "national"})
    with pytest.raises(SessionUnsupported):
        runner.delete_session("s1")
    assert runner.invoke(QUESTION)["answer"] is not None


def test_a_downgraded_answer_is_what_the_session_remembers() -> None:
    runner, _client = _runner(
        plan=PLAN, cfg=config.AgentConfig(max_steps=2, review_min_score=1.5)
    )

    state = runner.invoke(QUESTION, session_id="s1")

    assert state["answer"].text.startswith(REVIEW_DOWNGRADE_ANSWER)
    assert state["conversation"][-1]["answer"].startswith(REVIEW_DOWNGRADE_ANSWER)
    assert state["conversation"][-1]["question"] == QUESTION


def test_memory_survives_a_disabled_review() -> None:
    runner, _client = _runner(
        plan=PLAN, cfg=config.AgentConfig(max_steps=2, review_min_score=-1)
    )

    state = runner.invoke(QUESTION, session_id="s1")

    assert state["answer"].review is None
    assert state["conversation"] == [
        {
            "question": QUESTION,
            "answer": "答案正文[依据1]",
            "region": "深圳",
            "topic_id": "t1",
            "intent": "",
            "turn": 1,
        }
    ]


def test_the_runner_fallback_never_touches_the_session() -> None:
    saver = InMemorySaver()
    runner, _client = _runner(
        sessions=saver, llm=AgentStubLLM(model="none", available=False)
    )

    events = list(runner.stream(QUESTION, session_id="s1"))

    assert [kind for kind, _payload in events] == ["answer"]
    assert events[0][1]["session_id"] == "s1"
    assert saver.get_tuple({"configurable": {"thread_id": "s1"}}) is None


def test_an_unresolved_region_pauses_for_clarification_then_resumes_national() -> None:
    runner, _client = _runner(plan=PLAN, clarify=True, region=("?", "深圳"))

    events = list(runner.stream(QUESTION, session_id="s1"))

    kinds = [kind for kind, _payload in events]
    assert kinds == ["step", "interrupt"]
    assert events[0][1]["node"] == "region"
    payload = events[1][1]
    assert payload["session_id"] == "s1"
    assert payload["interrupt"]["id"]
    assert payload["interrupt"]["value"]["type"] == "region_clarify"
    assert payload["interrupt"]["value"]["place"] == "深圳"
    assert payload["interrupt"]["value"]["laws"] == [PENALTY]
    assert _values(runner, "s1")["stream_tokens"] is True

    resumed = list(runner.resume_stream("s1", {"region": "national"}))

    steps = [payload["node"] for kind, payload in resumed if kind == "step"]
    assert steps[0] == "clarify"
    assert steps[-1] == "review"
    assert resumed[-1][0] == "answer"
    assert resumed[-1][1]["answer"] == "答案正文[依据1]"
    values = _values(runner, "s1")
    assert values["region"] == "national"
    assert set(values["region_scope"]) == {ROAD_ID}


def test_a_resolved_region_answers_without_clarifying() -> None:
    runner, _client = _runner(plan=PLAN, clarify=True, region=("深圳", "深圳"))

    events = list(runner.stream(QUESTION, session_id="s1"))

    kinds = [kind for kind, _payload in events]
    assert "interrupt" not in kinds
    nodes = [payload["node"] for kind, payload in events if kind == "step"]
    assert nodes[0] == "region"
    assert events[-1][0] == "answer"
    assert events[-1][1]["answer"] == "答案正文[依据1]"
    values = _values(runner, "s1")
    assert values["region"] == "深圳"
    assert set(values["region_scope"]) == {ROAD_ID, PENALTY_ID}


def test_a_no_place_question_still_clarifies() -> None:
    runner, _client = _runner(plan=PLAN, clarify=True, region=("?", ""))

    events = list(runner.stream(QUESTION, session_id="s1"))

    kinds = [kind for kind, _payload in events]
    assert kinds == ["step", "interrupt"]
    payload = events[1][1]
    assert payload["interrupt"]["value"]["place"] == ""
    message = payload["interrupt"]["value"]["message"]
    assert "按哪里的规定回答" in message
    assert "涉及具体的地区" not in message


def test_a_law_name_reply_still_pauses_for_clarification() -> None:
    runner, _client = _runner(plan=PLAN, clarify=True, region=(ROAD, ""))

    events = list(runner.stream(QUESTION, session_id="s1"))

    kinds = [kind for kind, _payload in events]
    assert kinds == ["step", "interrupt"]
    assert events[0][1]["region"] == "?"
    assert events[1][1]["interrupt"]["value"]["type"] == "region_clarify"


def test_resuming_with_a_region_keeps_national_plus_that_local_law() -> None:
    runner, _client = _runner(plan=_plan(2), clarify=True, region=("?", "深圳"))

    list(runner.stream(QUESTION, session_id="s1"))
    list(runner.resume_stream("s1", {"region": "深圳经济特区"}))

    values = _values(runner, "s1")
    assert values["region"] == "深圳经济特区"
    assert set(values["region_scope"]) == {ROAD_ID, PENALTY_ID}
    assert values["conversation"][-1]["region"] == "深圳经济特区"


def test_the_resolved_region_rides_the_history_into_the_next_region_prompt() -> None:
    runner, _client = _runner(plan=_plan(2), clarify=True, region=("?", "深圳"))

    list(runner.stream(QUESTION, session_id="s1"))
    list(runner.resume_stream("s1", {"region": "深圳经济特区"}))
    list(runner.stream(FOLLOW_UP, session_id="s1"))

    prompts = runner.region_llm.prompts
    assert len(prompts) >= 2
    first, second = prompts[0], prompts[-1]
    assert second[-1] == {"role": "user", "content": FOLLOW_UP + REGION_GUARD}
    assert any(
        "（本轮按「深圳经济特区」的规定作答）" in message["content"] for message in second
    )
    assert "已定地区" in second[0]["content"]
    assert all("（本轮按" not in message["content"] for message in first)


def test_a_resume_reply_off_the_library_lands_a_note() -> None:
    runner, _client = _runner(plan=_plan(2), clarify=True, region=("?", "深圳"))

    list(runner.stream(QUESTION, session_id="s1"))
    resumed = list(runner.resume_stream("s1", {"region": "广州"}))

    assert resumed[-1][0] == "answer"
    notes = resumed[-1][1]["notes"]
    assert "回复的「广州」未匹配到库内地区，本次按全国法作答" in notes


def test_the_off_library_note_does_not_stick_to_the_next_turn() -> None:
    runner, _client = _runner(plan=_plan(4), clarify=True, region=("?", "深圳"))

    list(runner.stream(QUESTION, session_id="s1"))
    list(runner.resume_stream("s1", {"region": "广州"}))
    runner.region_llm._replies.append(_region("深圳", "深圳")[0])
    state = runner.invoke(FOLLOW_UP, session_id="s1")

    assert state["region"] == "深圳"
    assert not any("未匹配到库内地区" in note for note in state["answer"].notes)
    assert state["conversation"][-1]["region"] == "深圳"


def test_an_unknown_region_resume_falls_back_to_national() -> None:
    runner, _client = _runner(plan=_plan(2), clarify=True, region=("?", "深圳"))

    list(runner.stream(QUESTION, session_id="s1"))
    list(runner.resume_stream("s1", {"region": "火星"}))

    values = _values(runner, "s1")
    assert values["region"] == "national"
    assert set(values["region_scope"]) == {ROAD_ID}


def test_a_non_streaming_resume_returns_the_payload_and_keeps_flags_off() -> None:
    runner, client = _runner(plan=PLAN, clarify=True, region=("?", "深圳"))

    list(runner.stream(QUESTION, session_id="s1"))
    status, payload = runner.resume("s1", {"region": "national"})

    assert status == "ok"
    assert payload["session_id"] == "s1"
    assert payload["answer"] == "答案正文[依据1]"
    assert len(client.answers) == 1 and client.streams == []
    assert _values(runner, "s1")["stream_tokens"] is False


def test_resuming_a_finished_or_unknown_session_conflicts() -> None:
    runner, _client = _runner(plan=PLAN)

    runner.invoke(QUESTION, session_id="s1")
    with pytest.raises(ResumeConflict):
        runner.resume("s1", {"region": "national"})
    with pytest.raises(ResumeConflict):
        runner.resume("s1", {"region": "national"})
    with pytest.raises(ResumeConflict):
        runner.resume("never-started", {"region": "national"})


def test_resume_rejects_an_interrupt_that_is_not_a_region_clarify() -> None:
    runner, _client = _runner(plan=PLAN)
    foreign = SimpleNamespace(id="i1", value={"type": "something_else"})
    runner.graph = lambda: SimpleNamespace(
        get_state=lambda config: _FakeSnapshot([_FakeTask([foreign])])
    )

    with pytest.raises(ResumeConflict, match="澄清"):
        runner.resume("s1", {"region": "national"})


def test_stream_tokens_is_written_on_every_route_and_never_sticks() -> None:
    runner, _client = _runner(plan=_plan(2))

    list(runner.stream(QUESTION, session_id="s1"))
    assert _values(runner, "s1")["stream_tokens"] is True

    runner.invoke(FOLLOW_UP, session_id="s1")
    assert _values(runner, "s1")["stream_tokens"] is False


def test_the_non_stream_faces_never_ask_for_a_streamed_generation() -> None:
    runner, client = _runner(plan=PLAN)
    runner.invoke(QUESTION, session_id="s1")
    assert len(client.answers) == 1 and client.streams == []

    other, other_client = _runner(plan=PLAN)
    other.ask_payload(QUESTION, session_id="s2")
    assert len(other_client.answers) == 1 and other_client.streams == []


def test_ask_payload_returns_an_answer_under_a_generated_thread() -> None:
    runner, _client = _runner(plan=PLAN)

    status, payload = runner.ask_payload(QUESTION)

    assert status == "ok"
    assert payload["answer"] == "答案正文[依据1]"
    assert isinstance(payload["session_id"], str) and len(payload["session_id"]) == 32


def test_ask_payload_reports_an_interrupt_instead_of_an_answer() -> None:
    runner, _client = _runner(plan=PLAN, clarify=True, region=("?", "深圳"))

    status, payload = runner.ask_payload(QUESTION, session_id="s1")

    assert status == "interrupted"
    assert payload["session_id"] == "s1"
    assert payload["interrupt"]["value"]["type"] == "region_clarify"
    assert "answer" not in payload


def test_build_history_annotates_resolved_regions_only() -> None:
    history = region_mod._build_history(
        [
            {"question": "q1", "answer": "a1", "region": "深圳"},
            {"question": "q2", "answer": "a2", "region": "national"},
            {"question": "q3", "answer": "a3", "region": "?"},
            {"question": "q4", "answer": "a4"},
        ],
        (),
        4,
    )

    assert history == [
        ("user", "q1"),
        ("assistant", "（本轮按「深圳」的规定作答）a1"),
        ("user", "q2"),
        ("assistant", "（本轮按全国法作答）a2"),
        ("user", "q3"),
        ("assistant", "a3"),
        ("user", "q4"),
        ("assistant", "a4"),
    ]


def test_parse_region_reads_both_fields_and_survives_garbage() -> None:
    parse = region_mod._parse_region

    assert parse('{"region": "深圳", "place": "深圳龙岗"}') == ("深圳", "深圳龙岗")
    assert parse('开场白 {"region": "深圳"} 收尾') == ("深圳", "")
    assert parse('{"region": "  ", "place": 7}') == (region_mod.REGION_UNKNOWN, "7")
    assert parse('{"region": "?", "place": "?"}') == (region_mod.REGION_UNKNOWN, "")
    assert parse('{"region": "？", "place": " ？ "}') == (region_mod.REGION_UNKNOWN, "")
    assert parse('{"region": "全国", "place": "全国"}') == (region_mod.REGION_UNKNOWN, "")
    assert parse('{"region": "空串", "place": "空串"}') == (region_mod.REGION_UNKNOWN, "")
    assert parse('{"region": "national", "place": "?"}') == ("national", "")
    assert parse("不是 JSON") == (region_mod.REGION_UNKNOWN, "")
    assert parse("") == (region_mod.REGION_UNKNOWN, "")


def test_only_the_review_node_writes_the_conversation_channel() -> None:
    runner, _client = _runner(plan=PLAN)
    kwargs, _sid = runner._config_kwargs("plain")
    initial = runner._initial(Question(text=QUESTION), ())

    writers = set()
    for _mode, chunk in runner.graph().stream(initial, **kwargs, stream_mode=["updates"]):
        for node, update in chunk.items():
            if isinstance(update, dict) and "conversation" in update:
                writers.add(node)

    assert writers == {"review"}

    paused, _client = _runner(plan=PLAN, clarify=True, region=("?", "深圳"))
    kwargs, _sid = paused._config_kwargs("paused")
    initial = paused._initial(Question(text=QUESTION), ())

    writers = set()
    for _mode, chunk in paused.graph().stream(initial, **kwargs, stream_mode=["updates"]):
        if "__interrupt__" in chunk:
            break
        for node, update in chunk.items():
            if isinstance(update, dict) and "conversation" in update:
                writers.add(node)

    assert writers == set()


def test_every_turn_stays_in_the_conversation_intact() -> None:
    runner, _client = _runner(
        plan=_plan(6),
        cfg=config.AgentConfig(max_steps=2, history_turns=2),
        review_llm=AgentStubLLM(_review(True) * 6, model="stub-review"),
    )

    for index in range(6):
        state = runner.invoke(f"第 {index} 问", session_id="s1")

    assert [entry["question"] for entry in state["conversation"]] == [
        f"第 {index} 问" for index in range(6)
    ]
    assert all(entry["answer"] == "答案正文[依据1]" for entry in state["conversation"])


def test_an_injected_law_table_wins_over_the_service_laws() -> None:
    law = replace(AgentStubService().laws()[0], law_name="中华人民共和国海商法", local=False)
    runner, _client = _runner(plan=PLAN, region=("?", ""), laws=(law,))

    runner.invoke(QUESTION, session_id="s1")

    system = runner.region_llm.prompts[0][0]["content"]
    assert "海商法" in system
    assert ROAD not in system


def test_the_locked_saver_survives_interleaved_reads_and_writes(tmp_path) -> None:
    saver = open_sessions(str(tmp_path / "sessions.db"))
    errors: list[Exception] = []

    def worker(sid: str) -> None:
        try:
            runner, _client = _runner(plan=_plan(3), sessions=saver)
            for _ in range(3):
                runner.invoke(QUESTION, session_id=sid)
                saver.get_tuple({"configurable": {"thread_id": sid}})
                saver.list({"configurable": {"thread_id": sid}})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        threads = [threading.Thread(target=worker, args=(sid,)) for sid in ("t0", "t1")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads), "并发读写卡住了（可能死锁）"
        assert errors == []
        assert saver.get_tuple({"configurable": {"thread_id": "t0"}}) is not None
        assert saver.get_tuple({"configurable": {"thread_id": "t1"}}) is not None
    finally:
        saver.close()


def test_deleting_a_session_is_physical_and_vacuums(tmp_path, monkeypatch) -> None:
    saver = open_sessions(str(tmp_path / "sessions.db"))
    runner, _client = _runner(plan=_plan(2), sessions=saver)
    runner.invoke(QUESTION, session_id="s1")
    assert saver.get_tuple({"configurable": {"thread_id": "s1"}}) is not None

    calls: list[str] = []
    original = saver.vacuum
    monkeypatch.setattr(saver, "vacuum", lambda: (calls.append("vacuum"), original())[1])

    assert runner.delete_session("s1") is True
    assert calls == ["vacuum"]
    assert saver.get_tuple({"configurable": {"thread_id": "s1"}}) is None

    assert runner.delete_session("s1") is False
    assert calls == ["vacuum"]

    fresh = runner.invoke(FOLLOW_UP, session_id="s1")
    assert [entry["question"] for entry in fresh["conversation"]] == [FOLLOW_UP]
    saver.close()


def test_a_saver_without_vacuum_still_deletes() -> None:
    saver = InMemorySaver()
    runner, _client = _runner(plan=PLAN, sessions=saver)
    runner.invoke(QUESTION, session_id="s1")

    assert runner.delete_session("s1") is True
    assert saver.get_tuple({"configurable": {"thread_id": "s1"}}) is None
    assert runner.delete_session("s1") is False


def test_a_follow_up_rides_the_rewrite_into_the_plan_and_the_generator() -> None:
    intake = AgentStubLLM(
        _intake("continue", intent="followup", query="深圳醉酒驾驶记多少分"), model="stub-intake"
    )
    runner, client = _runner(plan=_plan(2), intake_llm=intake)

    runner.invoke(QUESTION, session_id="s1")
    state = runner.invoke(FOLLOW_UP, session_id="s1")

    assert state["query_rewrite"] == "深圳醉酒驾驶记多少分"
    assert state["topic_switch"] == "continue"
    assert state["intent"] == "followup"
    assert state["topic_id"] == "t1"
    assert state["turn_no"] == 2
    planner = runner.llm.prompts[2]
    assert planner[-1]["content"] == (
        "深圳醉酒驾驶记多少分\n\n（入口已把用户问句整理为上面这句；用户原话：那记分呢？）"
    )
    asked = client.answers[-1]["question"]
    assert asked.text == "深圳醉酒驾驶记多少分"
    assert asked.history == ()


def test_a_challenging_follow_up_hands_the_topic_view_to_the_generator() -> None:
    intake = AgentStubLLM(
        _intake("continue", intent="challenge", query="深圳醉驾真的吊销驾照吗"), model="stub-intake"
    )
    runner, client = _runner(plan=_plan(2), intake_llm=intake)

    runner.invoke(QUESTION, session_id="s1")
    runner.invoke(FOLLOW_UP, session_id="s1")

    asked = client.answers[-1]["question"]
    assert asked.text == "深圳醉驾真的吊销驾照吗"
    assert asked.history == (
        ("user", QUESTION),
        ("assistant", "（本轮按「深圳」的规定作答）答案正文[依据1]"),
    )


def test_topics_stay_isolated_and_an_old_one_can_be_resumed() -> None:
    intake = AgentStubLLM(
        _intake("continue", intent="followup", query="深圳醉酒驾驶记多少分")
        + _intake("new", intent="other", query="广州电动自行车上牌要什么材料")
        + _intake("continue", intent="followup", query="广州电动自行车上牌要多少钱")
        + _intake("resume", intent="other", query="深圳醉酒驾驶的处罚是什么", topic="t1"),
        model="stub-intake",
    )
    runner, client = _runner(
        plan=_plan(5),
        intake_llm=intake,
        review_llm=AgentStubLLM(_review(True) * 5, model="stub-review"),
    )

    runner.invoke(QUESTION, session_id="s1")
    runner.invoke(FOLLOW_UP, session_id="s1")
    runner.invoke("广州电动自行车上牌要什么材料？", session_id="s1")
    runner.invoke("那要多少钱？", session_id="s1")
    state = runner.invoke("回到醉驾那个，罚多少钱？", session_id="s1")

    assert state["topic_id"] == "t1"
    assert [entry["topic_id"] for entry in state["conversation"]] == ["t1", "t1", "t3", "t3", "t1"]

    planner4 = runner.llm.prompts[6]
    assert planner4[-1]["content"].startswith("广州电动自行车上牌要多少钱")
    assert {"role": "user", "content": QUESTION} not in planner4
    assert {"role": "user", "content": FOLLOW_UP} not in planner4
    assert client.answers[3]["question"].history == ()

    ask4 = intake.prompts[2][-1]["content"]
    assert "[t1] 在深圳，醉酒驾驶机动车怎么处罚？（共 2 轮，最近第 2 轮）" in ask4

    planner5 = runner.llm.prompts[8]
    assert {"role": "user", "content": QUESTION} in planner5
    assert {"role": "user", "content": "深圳醉酒驾驶记多少分"} in planner5
    assert {"role": "user", "content": "广州电动自行车上牌要什么材料"} not in planner5
    asked5 = client.answers[4]["question"]
    assert asked5.text == "深圳醉酒驾驶的处罚是什么"
    assert ("user", QUESTION) in asked5.history
    assert ("user", "深圳醉酒驾驶记多少分") in asked5.history
    assert ("user", "广州电动自行车上牌要什么材料") not in asked5.history

    ask5 = intake.prompts[3][-1]["content"]
    assert "当前活跃话题：t3。" in ask5
    assert "[t1] 在深圳，醉酒驾驶机动车怎么处罚？（共 2 轮，最近第 2 轮）" in ask5


def test_without_an_intake_the_generator_still_sees_the_plain_window() -> None:
    runner, client = _runner(plan=_plan(2))

    runner.invoke(QUESTION, session_id="s1")
    runner.invoke(FOLLOW_UP, session_id="s1")

    asked = client.answers[-1]["question"]
    assert asked.text == FOLLOW_UP
    assert asked.history == (
        ("user", QUESTION),
        ("assistant", "（本轮按「深圳」的规定作答）答案正文[依据1]"),
    )


def test_the_topic_window_keeps_only_the_last_three_turns() -> None:
    intake = AgentStubLLM(
        _intake("continue", query="问题的改写1")
        + _intake("continue", query="问题的改写2")
        + _intake("continue", query="问题的改写3")
        + _intake("continue", query="问题的改写4"),
        model="stub-intake",
    )
    runner, _client = _runner(
        plan=_plan(5),
        intake_llm=intake,
        history_turns=3,
        review_llm=AgentStubLLM(_review(True) * 5, model="stub-review"),
    )

    for index in range(5):
        runner.invoke(f"第 {index} 问", session_id="s1")

    planner5 = runner.llm.prompts[8]
    users = [message["content"] for message in planner5 if message["role"] == "user"]
    assert users[:-1] == ["问题的改写1", "问题的改写2", "问题的改写3"]
    assert "第 0 问" not in "".join(users)


def test_every_run_prunes_the_thread_to_a_single_checkpoint(tmp_path) -> None:
    saver = open_sessions(str(tmp_path / "sessions.db"))
    runner, _client = _runner(plan=_plan(3), sessions=saver)

    for _ in range(3):
        runner.invoke(QUESTION, session_id="s1")

    kept = saver._conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 's1'"
    ).fetchone()[0]
    stale = saver._conn.execute(
        "SELECT COUNT(*) FROM writes WHERE thread_id = 's1' AND checkpoint_id !="
        " (SELECT MAX(checkpoint_id) FROM checkpoints WHERE thread_id = 's1')"
    ).fetchone()[0]
    saver.close()

    assert kept == 1
    assert stale == 0


def _write_checkpoint(saver, thread_id: str, *, ns: str = "", task_id: str = "") -> str:
    placed = saver.put(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns}},
        empty_checkpoint(),
        {},
        {},
    )
    if task_id:
        saver.put_writes(placed, [("channel", task_id)], task_id=task_id)
    return placed["configurable"]["checkpoint_id"]


def test_prune_keeps_only_the_latest_checkpoint_per_namespace(tmp_path) -> None:
    saver = open_sessions(str(tmp_path / "sessions.db"))
    _write_checkpoint(saver, "t1", task_id="task-old")
    latest = _write_checkpoint(saver, "t1", task_id="task-new")
    _write_checkpoint(saver, "t1", ns="sub", task_id="task-sub")
    _write_checkpoint(saver, "t2")

    saver.prune(["t1"])

    kept = saver._conn.execute(
        "SELECT checkpoint_id FROM checkpoints WHERE thread_id = 't1' AND checkpoint_ns = ''"
    ).fetchall()
    assert kept == [(latest,)]
    writes = saver._conn.execute(
        "SELECT task_id FROM writes WHERE thread_id = 't1' AND checkpoint_ns = ''"
    ).fetchall()
    assert writes == [("task-new",)]
    sub = saver._conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 't1' AND checkpoint_ns = 'sub'"
    ).fetchone()[0]
    assert sub == 1
    other = saver._conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 't2'"
    ).fetchone()[0]
    assert other == 1
    saver.close()


def test_prune_is_idempotent(tmp_path) -> None:
    saver = open_sessions(str(tmp_path / "sessions.db"))
    _write_checkpoint(saver, "t1", task_id="task-old")
    _write_checkpoint(saver, "t1", task_id="task-new")

    def snapshot() -> tuple:
        return (
            saver._conn.execute("SELECT * FROM checkpoints ORDER BY 1, 2, 3").fetchall(),
            saver._conn.execute("SELECT * FROM writes ORDER BY 1, 2, 3, 4").fetchall(),
        )

    saver.prune(["t1"])
    after_first = snapshot()
    saver.prune(["t1"])
    assert snapshot() == after_first
    saver.close()


def test_prune_delete_strategy_wipes_the_threads(tmp_path) -> None:
    saver = open_sessions(str(tmp_path / "sessions.db"))
    _write_checkpoint(saver, "t1", task_id="task-a")
    _write_checkpoint(saver, "t2")
    _write_checkpoint(saver, "t3")

    saver.prune(["t1", "t2"], strategy="delete")

    ghosts = saver._conn.execute(
        "SELECT thread_id FROM checkpoints ORDER BY thread_id"
    ).fetchall()
    assert ghosts == [("t3",)]
    assert saver._conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 0
    saver.close()


def test_a_foreign_schema_disables_prune_and_says_so(tmp_path, capsys) -> None:
    path = str(tmp_path / "sessions.db")
    conn = open_sessions_db(path)
    SqliteSaver(conn).setup()
    conn.executescript(
        "DROP TABLE writes;"
        " DROP TABLE checkpoints;"
        " CREATE TABLE checkpoints (thread_id TEXT);"
        " CREATE TABLE writes (thread_id TEXT);"
    )
    saver = session_mod._LockedSaver(SqliteSaver(conn), conn)

    assert "prune disabled" in capsys.readouterr().err
    saver.prune(["t1"])
    assert saver._conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
    saver.close()


def test_prune_failures_never_take_a_run_down(capsys) -> None:
    runner, _client = _runner()

    class _Unavailable:
        def prune(self, thread_ids, *, strategy="keep_latest"):
            raise NotImplementedError

    class _Broken:
        def prune(self, thread_ids, *, strategy="keep_latest"):
            raise RuntimeError("库坏了")

    runner.sessions = _Unavailable()
    runner._prune("s1")
    assert capsys.readouterr().err == ""

    runner.sessions = _Broken()
    runner._prune("s1")
    assert "[agent] prune skipped" in capsys.readouterr().err

    runner.sessions = None
    runner._prune("s1")
    runner._prune(None)
    assert capsys.readouterr().err == ""
