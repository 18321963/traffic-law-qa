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
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent_service.agents import region as region_mod  # noqa: E402
from agent_service.agents.errors import ResumeConflict, SessionUnsupported  # noqa: E402
from agent_service.agents.graph import AgentRunner, make_run_config  # noqa: E402
from agent_service.agents.region import HISTORY_ANSWER_CHARS  # noqa: E402
from agent_service.agents.session import open_sessions  # noqa: E402
from agent_service.prompts import REVIEW_DOWNGRADE_ANSWER  # noqa: E402

ROAD_ID = "road_traffic_safety"
PENALTY_ID = "shenzhen_penalty"
QUESTION = "在深圳，醉酒驾驶机动车怎么处罚？"
FOLLOW_UP = "那记分呢？"

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
    sessions=_MISSING,
    laws=_MISSING,
    cfg=None,
    clarify=False,
    history_turns=5,
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

    turn2_region = runner.region_llm.prompts[1]
    assert {"role": "user", "content": QUESTION} in turn2_region
    assert {"role": "assistant", "content": "答案正文[依据1]"} in turn2_region
    assert turn2_region[-1] == {"role": "user", "content": FOLLOW_UP}

    turn2_agent = runner.llm.prompts[2]
    assert {"role": "user", "content": QUESTION} in turn2_agent
    assert {"role": "assistant", "content": "答案正文[依据1]"} in turn2_agent


def test_a_first_turn_prompt_is_still_system_plus_question() -> None:
    runner, _client = _runner(plan=PLAN)

    runner.invoke(QUESTION, session_id="s1")

    prompt = runner.region_llm.prompts[0]
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[-1] == {"role": "user", "content": QUESTION}


def test_input_history_is_normalized_into_the_first_turn_prompt() -> None:
    runner, _client = _runner(plan=PLAN)
    query = Question(text=FOLLOW_UP, history=[("user", "醉驾怎么罚"), ("assistant", "答案A")])

    runner.invoke(query, session_id="s1")

    prompt = runner.region_llm.prompts[0]
    assert {"role": "user", "content": "醉驾怎么罚"} in prompt
    assert {"role": "assistant", "content": "答案A"} in prompt
    assert prompt[-1] == {"role": "user", "content": FOLLOW_UP}


def test_the_history_snippet_clips_long_answers() -> None:
    runner, _client = _runner(plan=_plan(2), client=_LongAnswerService())

    runner.invoke(QUESTION, session_id="s1")
    runner.invoke(FOLLOW_UP, session_id="s1")

    assistant = [m["content"] for m in runner.region_llm.prompts[1] if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0].endswith("…")
    assert len(assistant[0]) == HISTORY_ANSWER_CHARS + 1


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
    assert state["conversation"] == [{"question": QUESTION, "answer": "答案正文[依据1]"}]


def test_the_runner_fallback_never_touches_the_session() -> None:
    saver = InMemorySaver()
    runner, _client = _runner(
        sessions=saver, llm=AgentStubLLM(model="none", available=False)
    )

    events = list(runner.stream(QUESTION, session_id="s1"))

    assert [kind for kind, _payload in events] == ["answer"]
    assert events[0][1]["session_id"] == "s1"
    assert saver.get_tuple({"configurable": {"thread_id": "s1"}}) is None


def test_a_place_question_pauses_for_clarification_then_resumes_national() -> None:
    runner, _client = _runner(plan=PLAN, clarify=True, region=("深圳", "深圳"))

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


def test_resuming_with_a_region_keeps_national_plus_that_local_law() -> None:
    runner, _client = _runner(plan=_plan(2), clarify=True, region=("深圳", "深圳"))

    list(runner.stream(QUESTION, session_id="s1"))
    list(runner.resume_stream("s1", {"region": "深圳经济特区"}))

    values = _values(runner, "s1")
    assert values["region"] == "深圳经济特区"
    assert set(values["region_scope"]) == {ROAD_ID, PENALTY_ID}


def test_an_unknown_region_resume_falls_back_to_national() -> None:
    runner, _client = _runner(plan=_plan(2), clarify=True, region=("深圳", "深圳"))

    list(runner.stream(QUESTION, session_id="s1"))
    list(runner.resume_stream("s1", {"region": "火星"}))

    values = _values(runner, "s1")
    assert values["region"] == "national"
    assert set(values["region_scope"]) == {ROAD_ID}


def test_a_non_streaming_resume_returns_the_payload_and_keeps_flags_off() -> None:
    runner, client = _runner(plan=PLAN, clarify=True, region=("深圳", "深圳"))

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
    runner, _client = _runner(plan=PLAN, clarify=True, region=("深圳", "深圳"))

    status, payload = runner.ask_payload(QUESTION, session_id="s1")

    assert status == "interrupted"
    assert payload["session_id"] == "s1"
    assert payload["interrupt"]["value"]["type"] == "region_clarify"
    assert "answer" not in payload


def test_parse_region_reads_both_fields_and_survives_garbage() -> None:
    parse = region_mod._parse_region

    assert parse('{"region": "深圳", "place": "深圳龙岗"}') == ("深圳", "深圳龙岗")
    assert parse('开场白 {"region": "深圳"} 收尾') == ("深圳", "")
    assert parse('{"region": "  ", "place": 7}') == (region_mod.REGION_UNKNOWN, "7")
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

    paused, _client = _runner(plan=PLAN, clarify=True, region=("深圳", "深圳"))
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


def test_the_conversation_is_capped_and_truncated_to_n_turns() -> None:
    runner, _client = _runner(
        plan=_plan(6),
        cfg=config.AgentConfig(max_steps=2, history_turns=2),
        review_llm=AgentStubLLM(_review(True) * 6, model="stub-review"),
    )

    for index in range(6):
        state = runner.invoke(f"第 {index} 问", session_id="s1")
        assert len(state["conversation"]) <= 2 * 2

    assert [entry["question"] for entry in state["conversation"]] == [
        "第 3 问",
        "第 4 问",
        "第 5 问",
    ]


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
