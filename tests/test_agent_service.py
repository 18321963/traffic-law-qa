from __future__ import annotations

import json

import pytest

from rag_contracts import config
from rag_contracts.domain.answer import Question
from rag_contracts.domain.errors import QaError, QaTimeout
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
from agent_service.agents.errors import ResumeConflict  # noqa: E402
from agent_service.agents.graph import AgentRunner  # noqa: E402
from agent_service.agents.nodes import make_agent_node, make_tools_node  # noqa: E402
from agent_service.merge import merge_retrievals  # noqa: E402
from agent_service.prompts import (  # noqa: E402
    INTAKE_OUTPUT_GUARD,
    RESUME_UNAVAILABLE_DETAIL,
    REVIEW_DOWNGRADE_ANSWER,
)
from agent_service.tools import handlers  # noqa: E402
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


def _intake(
    switch: str = "continue",
    *,
    intent: str = "other",
    query: str = "",
    topic: str = "",
    summary: str = "",
    region: str = "",
    place: str = "",
) -> list[dict]:
    payload = {
        "switch": switch,
        "intent": intent,
        "query": query,
        "topic_id": topic,
        "summary": summary,
    }
    if region:
        payload["region"] = region
        payload["place"] = place
    return [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]


def _memory(question: str, topic_id: str, turn: int, *, intent: str = "other") -> dict:
    return {
        "question": question,
        "answer": f"答案：{question}",
        "region": "深圳",
        "topic_id": topic_id,
        "intent": intent,
        "turn": turn,
    }


class _BoomLLM(AgentStubLLM):

    def chat(self, messages, **kwargs):
        raise RuntimeError("入口模型炸了")


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
        else AgentStubLLM(
            list(plan),
            model="stub-agent",
            finishes=finishes,
            tokens=tokens,
            intake=_intake(region=region, place=place or ""),
        ),
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
    llm = AgentStubLLM(intake=_intake(region="深圳"))

    update = region_mod.make_region_node(llm, client.laws())(_state())

    assert update["region"] == "深圳"
    assert update["region_scope"] == (ROAD_ID, PENALTY_ID)
    system = llm.intake_prompts[0][0]["content"]
    assert ROAD in system and PENALTY in system


def test_a_law_name_reply_from_the_region_model_reads_as_unknown() -> None:
    client = AgentStubService()
    llm = AgentStubLLM(intake=_intake(region=ROAD))

    update = region_mod.make_region_node(llm, client.laws())(_state())

    assert update["region"] == region_mod.REGION_UNKNOWN
    assert update["region_scope"] == ()


def test_the_topic_id_folding_table_holds_line_by_line() -> None:
    client = AgentStubService()
    conversation = [
        _memory("醉酒驾驶怎么处罚", "t1", 1),
        _memory("那记分呢", "t2", 2, intent="followup"),
    ]
    cases = [
        (_intake("continue", intent="followup", query="深圳醉驾记分", region="深圳"), "t2"),
        (_intake("resume", query="醉驾怎么处罚", topic="t1", region="深圳"), "t1"),
        (_intake("resume", query="旧问题", topic="t9", region="深圳"), "t2"),
        (_intake("new", query="外地车限行", region="深圳"), "t3"),
        (_intake("unsure", region="深圳"), "t2"),
    ]
    for replies, expected in cases:
        node = region_mod.make_region_node(AgentStubLLM(intake=replies), client.laws())
        update = node(_state(conversation=list(conversation), topic_id="t2", turn_no=2))
        assert update["topic_id"] == expected, replies


def test_a_slot_only_restatement_keeps_the_topic_when_new_is_reported() -> None:
    client = AgentStubService()
    conversation = [
        _memory("电动自行车载人怎么处理", "t6", 6),
        _memory("深圳电动自行车上牌需要什么材料", "t7", 7),
    ]
    intake = AgentStubLLM(
        intake=_intake(
            "new",
            query="广州电动自行车上牌需要什么材料",
            summary="问深圳电动自行车上牌材料；按深圳",
            region="?",
        )
    )
    node = region_mod.make_region_node(intake, client.laws())

    update = node(_state(conversation=list(conversation), topic_id="t7", turn_no=7))

    assert update["switch_raw"] == "new"
    assert update["topic_switch"] == "continue", "同事项仅换地区槽位，环节一规则 3 不许换话题"
    assert update["topic_id"] == "t7"
    assert "topic_summaries" not in update, "没离开活跃话题，摘要不落账"
    assert update["history"] == [
        ("user", "深圳电动自行车上牌需要什么材料"),
        ("assistant", "（本轮按「深圳」的规定作答）答案：深圳电动自行车上牌需要什么材料"),
    ]


def test_a_true_new_matter_survives_the_slot_gate() -> None:
    client = AgentStubService()
    conversation = [_memory("深圳电动自行车上牌需要什么材料", "t7", 7)]
    intake = AgentStubLLM(
        intake=_intake(
            "new",
            query="广州电动自行车载人怎么处罚",
            summary="问深圳电动自行车上牌材料；按深圳",
            region="?",
        )
    )
    node = region_mod.make_region_node(intake, client.laws())

    update = node(_state(conversation=list(conversation), topic_id="t7", turn_no=7))

    assert update["switch_raw"] == "new"
    assert update["topic_switch"] == "new", "真换事项不许被闸拉回旧话题"
    assert update["topic_id"] == "t8"
    assert update["topic_summaries"] == {"t7": "问深圳电动自行车上牌材料；按深圳"}


def test_the_slot_gate_reads_the_active_summary_first() -> None:
    client = AgentStubService()
    conversation = [_memory("最早问的那件事", "t7", 7)]
    intake = AgentStubLLM(intake=_intake("new", query="广州电动自行车上牌需要什么材料", region="?"))
    node = region_mod.make_region_node(intake, client.laws())

    update = node(
        _state(
            conversation=list(conversation),
            topic_id="t7",
            turn_no=7,
            topic_summaries={"t7": "深圳电动自行车上牌需要什么材料"},
        )
    )

    assert update["switch_raw"] == "new"
    assert update["topic_switch"] == "continue", "有摘要时以摘要为头，不被旧问题头挡住"
    assert update["topic_id"] == "t7"


def test_the_slot_gate_only_touches_plain_new_turns() -> None:
    client = AgentStubService()
    conversation = [_memory("深圳电动自行车上牌需要什么材料", "t1", 1)]
    resume = AgentStubLLM(
        intake=_intake("resume", query="广州电动自行车上牌需要什么材料", topic="t1", region="?")
    )
    node = region_mod.make_region_node(resume, client.laws())
    update = node(_state(conversation=list(conversation), topic_id="t1", turn_no=1))
    assert update["topic_switch"] == "resume", "闸只认 new，resume 原样穿过"

    short = AgentStubLLM(intake=_intake("new", query="那记分呢", region="深圳"))
    node = region_mod.make_region_node(short, client.laws())
    update = node(_state(conversation=[_memory("那记分呢", "t1", 1)], topic_id="t1", turn_no=1))
    assert update["topic_switch"] == "new", "短头的相似度不稳，不参与闸判"
    assert update["topic_id"] == "t2"

    empty = AgentStubLLM(intake=_intake("new", region="深圳"))
    node = region_mod.make_region_node(empty, client.laws())
    update = node(_state(conversation=list(conversation), topic_id="t1", turn_no=1))
    assert update["topic_switch"] == "new", "没有改写就没有可比对的问句"
    assert update["topic_id"] == "t2"


def test_the_intake_stage_runs_every_turn_with_a_model() -> None:
    client = AgentStubService()
    intake = AgentStubLLM(intake=_intake("new", query="深圳醉驾怎么罚", region="深圳"))
    node = region_mod.make_region_node(intake, client.laws())

    update = node(_state())

    assert len(intake.intake_prompts) == 1, "判地区折入后没有「有没有历史」这条闸：首轮也发"
    assert update["query_rewrite"] == "深圳醉驾怎么罚", "首轮的改写也可能非空"
    assert update["topic_switch"] == "new"
    assert update["intent"] == "other"
    assert update["topic_id"] == "t1"
    assert update["turn_no"] == 1
    assert update["region"] == "深圳"

    off = AgentStubLLM(intake=_intake("new"), available=False)
    node_off = region_mod.make_region_node(off, client.laws())
    update_off = node_off(_state(conversation=[_memory("旧问", "t1", 1)], topic_id="t1"))
    assert off.intake_prompts == []
    assert update_off["topic_id"] == "t1"
    assert update_off["region"] == region_mod.REGION_UNKNOWN
    assert update_off["usage_extra"].value == []


def test_the_intake_stage_rewrites_before_reading_the_region() -> None:
    client = AgentStubService()
    intake = AgentStubLLM(
        intake=_intake("continue", intent="followup", query="深圳醉酒驾驶记多少分", region="深圳")
    )
    node = region_mod.make_region_node(intake, client.laws())

    update = node(
        _state(conversation=[_memory("醉酒驾驶怎么处罚", "t1", 1)], topic_id="t1", turn_no=1)
    )

    assert update["query_rewrite"] == "深圳醉酒驾驶记多少分"
    assert update["topic_switch"] == "continue"
    assert update["intent"] == "followup"
    assert update["topic_id"] == "t1"
    assert update["turn_no"] == 2
    assert update["region"] == "深圳"

    ask = intake.intake_prompts[0]
    assert ask[0]["role"] == "system"
    assert "五、判地区" in ask[0]["content"], "一次调用同时出五件事，判地区折在同一份系统提示里"
    assert {"role": "user", "content": "醉酒驾驶怎么处罚"} in ask
    assert "当前活跃话题：t1。" in ask[-1]["content"]
    assert "本轮用户问题：在深圳，醉酒驾驶机动车怎么处罚？" in ask[-1]["content"]
    assert "较早的其他话题" not in ask[-1]["content"]
    assert ask[-1]["content"].endswith(INTAKE_OUTPUT_GUARD)


def test_a_broken_entry_call_falls_back_to_the_plain_turn() -> None:
    client = AgentStubService()
    junk = [
        _BoomLLM(),
        AgentStubLLM(intake=[{"role": "assistant", "content": "不是 JSON"}]),
    ]
    for llm in junk:
        node = region_mod.make_region_node(llm, client.laws())
        update = node(_state(conversation=[_memory("旧问", "t1", 1)], topic_id="t1", turn_no=1))

        assert update["topic_id"] == "t1", llm
        assert update["turn_no"] == 2
        assert update["query_rewrite"] == ""
        assert update["topic_switch"] == ""
        assert update["intent"] == ""
        assert update["region"] == region_mod.REGION_UNKNOWN, "入口调用一坏，判地区一起降级为 ?"
        assert update["region_scope"] == ()
        assert update["history"] == [
            ("user", "旧问"),
            ("assistant", "（本轮按「深圳」的规定作答）答案：旧问"),
        ]


def test_unknown_enum_values_land_on_unsure_and_other() -> None:
    client = AgentStubService()
    intake = AgentStubLLM(
        intake=[
            {
                "role": "assistant",
                "content": '{"switch": "飞到火星", "intent": "未知", "query": "q", "topic_id": "t9"}',
            }
        ]
    )
    node = region_mod.make_region_node(intake, client.laws())

    update = node(_state(conversation=[_memory("旧问", "t1", 1)], topic_id="t1", turn_no=1))

    assert update["topic_switch"] == "unsure"
    assert update["intent"] == "other"
    assert update["query_rewrite"] == "q"
    assert update["topic_id"] == "t1"


def test_a_missing_model_still_carries_the_window_and_the_topic() -> None:
    client = AgentStubService()
    llm = AgentStubLLM(
        intake=_intake("continue", intent="followup", query="深圳醉驾记多少分", region="深圳"),
        available=False,
    )
    node = region_mod.make_region_node(llm, client.laws())

    update = node(
        _state(conversation=[_memory("醉酒驾驶怎么处罚", "t1", 1)], topic_id="t1", turn_no=1)
    )

    assert update["region"] == region_mod.REGION_UNKNOWN
    assert update["region_scope"] == ()
    assert update["topic_id"] == "t1"
    assert update["query_rewrite"] == ""
    assert update["history"] == [
        ("user", "醉酒驾驶怎么处罚"),
        ("assistant", "（本轮按「深圳」的规定作答）答案：醉酒驾驶怎么处罚"),
    ]
    assert update["usage_extra"].value == []


def test_the_turn_counter_steps_over_seen_topic_numbers() -> None:
    client = AgentStubService()
    node = region_mod.make_region_node(AgentStubLLM(intake=_intake("new", region="深圳")), client.laws())

    update = node(_state(conversation=[_memory("老问", "t5", 5)]))

    assert update["turn_no"] == 6
    assert update["topic_id"] == "t6"
    assert update["region"] == "深圳"


def test_the_memory_view_switches_to_the_active_topic() -> None:
    client = AgentStubService()
    conversation = [
        _memory("醉驾怎么处罚", "t1", 1),
        _memory("记分呢", "t1", 2, intent="followup"),
        _memory("电动车上牌", "t2", 3),
        _memory("电池着火怎么赔", "", 4),
    ]
    intake = AgentStubLLM(intake=_intake("continue", intent="followup", query="上牌要多少钱", region="深圳"))
    node = region_mod.make_region_node(intake, client.laws())

    update = node(_state(conversation=list(conversation), topic_id="t2", turn_no=4))

    assert update["history"] == [
        ("user", "电动车上牌"),
        ("assistant", "（本轮按「深圳」的规定作答）答案：电动车上牌"),
        ("user", "电池着火怎么赔"),
        ("assistant", "（本轮按「深圳」的规定作答）答案：电池着火怎么赔"),
    ]
    ask = intake.intake_prompts[0][-1]["content"]
    assert "较早的其他话题（只有索引，没有原文）：" in ask
    assert "（仅列最近 10 个话题；更早话题未列出、无法回访，指代不到时按 new 处理）" in ask
    assert "[t1] 醉驾怎么处罚（共 2 轮，最近第 2 轮）" in ask
    assert "[?]" not in ask


def test_leaving_the_active_topic_writes_its_summary() -> None:
    client = AgentStubService()
    conversation = [_memory("醉驾怎么处罚", "t1", 1), _memory("记分呢", "t1", 2, intent="followup")]
    intake = AgentStubLLM(
        intake=_intake("new", query="外地车限行", summary="问醉驾处罚与记分；按深圳", region="深圳")
    )
    node = region_mod.make_region_node(intake, client.laws())

    update = node(_state(conversation=list(conversation), topic_id="t1", turn_no=2))

    assert update["topic_id"] == "t3"
    assert update["topic_summaries"] == {"t1": "问醉驾处罚与记分；按深圳"}
    ask = intake.intake_prompts[0][-1]["content"]
    assert "当前活跃话题既有摘要" not in ask, "没有旧摘要时不该出现「既有摘要」行"


def test_a_continuing_turn_keeps_the_summaries_channel_untouched() -> None:
    client = AgentStubService()
    conversation = [_memory("醉驾怎么处罚", "t1", 1)]
    stray = AgentStubLLM(
        intake=_intake("continue", query="记分呢", summary="本轮不是离开，不该落账", region="深圳")
    )
    node = region_mod.make_region_node(stray, client.laws())

    update = node(_state(conversation=list(conversation), topic_id="t1", turn_no=1))

    assert stray.intake_prompts, "入口这一枪该照发"
    assert "topic_summaries" not in update, "continue 轮不写摘要，通道保留旧值"

    held = AgentStubLLM(intake=_intake("continue", query="记分呢", region="深圳"))
    node = region_mod.make_region_node(held, client.laws())
    update = node(
        _state(
            conversation=list(conversation),
            topic_id="t1",
            turn_no=1,
            topic_summaries={"t1": "旧摘要"},
        )
    )
    assert "topic_summaries" not in update, "没写时不能整字典重写：会盖掉其他话题的摘要"


def test_leaving_again_folds_the_old_summary_into_the_prompt() -> None:
    client = AgentStubService()
    conversation = [
        _memory("醉驾怎么处罚", "t1", 1),
        _memory("上牌要什么材料", "t2", 2),
        _memory("那电动自行车呢", "t1", 3, intent="followup"),
    ]
    intake = AgentStubLLM(
        intake=_intake("new", query="大货车限行", summary="问电动自行车上牌；深圳", region="深圳")
    )
    node = region_mod.make_region_node(intake, client.laws())

    update = node(
        _state(
            conversation=list(conversation),
            topic_id="t1",
            turn_no=3,
            topic_summaries={"t1": "问醉驾处罚；深圳"},
        )
    )

    assert update["topic_id"] == "t4"
    assert update["topic_summaries"]["t1"] == "问电动自行车上牌；深圳"
    ask = intake.intake_prompts[0][-1]["content"]
    assert "当前活跃话题既有摘要：问醉驾处罚；深圳（离开本话题时以它为底更新）" in ask


def test_summaries_beyond_the_renderable_face_are_evicted() -> None:
    client = AgentStubService()
    conversation = [_memory(f"第{index}问", f"t{index}", index) for index in range(1, 13)]
    intake = AgentStubLLM(intake=_intake("new", query="全新问题", summary="更新：第12问", region="深圳"))
    node = region_mod.make_region_node(intake, client.laws())

    update = node(
        _state(
            conversation=list(conversation),
            topic_id="t12",
            turn_no=12,
            topic_summaries={f"t{index}": f"摘要{index}" for index in range(1, 13)},
        )
    )

    kept = update["topic_summaries"]
    assert set(kept) == {f"t{index}" for index in range(2, 13)}, "只留可渲染面，最旧的先淘汰"
    assert kept["t12"] == "更新：第12问"


def test_the_plus_one_slot_keeps_the_resumed_and_shown_summaries() -> None:
    client = AgentStubService()
    conversation = [_memory(f"第{index}问", f"t{index}", index) for index in range(1, 12)]
    intake = AgentStubLLM(
        intake=_intake("resume", topic="t1", query="第1问问的啥", summary="更新：第11问", region="深圳")
    )
    node = region_mod.make_region_node(intake, client.laws())

    update = node(
        _state(
            conversation=list(conversation),
            topic_id="t11",
            turn_no=11,
            topic_summaries={f"t{index}": f"摘要{index}" for index in range(1, 12)},
        )
    )

    assert update["topic_id"] == "t1"
    kept = update["topic_summaries"]
    assert set(kept) == {f"t{index}" for index in range(1, 12)}, (
        "回访中的活跃话题不在索引行里、但占摘要槽位（+1）；显示窗口下沿的话题也必须保"
    )
    assert kept["t1"] == "摘要1", "回访话题的旧摘要要留着当增量基底"
    assert kept["t11"] == "更新：第11问"


def test_a_broken_intake_or_an_orphan_topic_writes_no_summary() -> None:
    client = AgentStubService()
    conversation = [_memory("旧问", "t1", 1)]
    junk = [
        _BoomLLM(),
        AgentStubLLM(intake=[{"role": "assistant", "content": "不是 JSON"}]),
    ]
    for llm in junk:
        node = region_mod.make_region_node(llm, client.laws())
        update = node(
            _state(
                conversation=list(conversation),
                topic_id="t1",
                turn_no=1,
                topic_summaries={"t1": "旧摘要"},
            )
        )
        assert "topic_summaries" not in update, "坏了就整个不写，不能清空既有摘要"

    orphan = AgentStubLLM(intake=_intake("new", query="新问题", summary="不该落账", region="深圳"))
    node = region_mod.make_region_node(orphan, client.laws())
    update = node(_state(conversation=list(conversation), topic_id="t9", turn_no=1))

    assert update["topic_id"] == "t2"
    assert "topic_summaries" not in update, "活跃话题没有条目（澄清中断轮残留）时不写孤儿键"


def test_the_index_line_prefers_a_topic_summary() -> None:
    client = AgentStubService()
    conversation = [
        _memory("醉驾怎么处罚", "t1", 1),
        _memory("记分呢", "t1", 2, intent="followup"),
        _memory("电动车上牌", "t2", 3),
    ]
    intake = AgentStubLLM(intake=_intake("continue", query="上牌要多少钱", region="深圳"))
    node = region_mod.make_region_node(intake, client.laws())

    node(
        _state(
            conversation=list(conversation),
            topic_id="t2",
            turn_no=3,
            topic_summaries={"t1": "问深圳醉驾处罚与记分"},
        )
    )

    ask = intake.intake_prompts[0][-1]["content"]
    assert "[t1] 问深圳醉驾处罚与记分（共 2 轮，最近第 2 轮）" in ask
    assert "[t1] 醉驾怎么处罚" not in ask, "有摘要就不再用起点问题截图"


def test_an_overlong_summary_is_clipped_on_write_and_on_render() -> None:
    client = AgentStubService()
    node = region_mod.make_region_node(
        AgentStubLLM(intake=_intake("new", query="外地车限行", summary="问" * 90, region="深圳")),
        client.laws(),
    )

    update = node(_state(conversation=[_memory("旧问", "t1", 1)], topic_id="t1", turn_no=1))

    assert update["topic_summaries"] == {"t1": "问" * 60}

    intake = AgentStubLLM(intake=_intake("continue", query="继续", region="深圳"))
    node = region_mod.make_region_node(intake, client.laws())
    node(
        _state(
            conversation=[
                _memory("旧问", "t1", 1),
                _memory("追问", "t1", 2, intent="followup"),
                _memory("新问", "t2", 3),
            ],
            topic_id="t2",
            turn_no=3,
            topic_summaries={"t1": "题" * 90},
        )
    )
    ask = intake.intake_prompts[0][-1]["content"]
    assert f"[t1] {'题' * 60}（共 2 轮，最近第 2 轮）" in ask
    assert "题" * 61 not in ask, "渲染端要兜底截断，别把整段摘要灌进索引"


def test_the_summary_lands_even_when_the_region_cannot_be_judged() -> None:
    client = AgentStubService()
    intake = AgentStubLLM(intake=_intake("new", query="外地车限行", summary="问醉驾；深圳"))
    node = region_mod.make_region_node(intake, client.laws())

    update = node(_state(conversation=[_memory("醉驾怎么处罚", "t1", 1)], topic_id="t1", turn_no=1))

    assert update["region"] == region_mod.REGION_UNKNOWN
    assert update["topic_summaries"] == {"t1": "问醉驾；深圳"}


def test_stale_unnumbered_turns_collapse_into_a_tail_line() -> None:
    client = AgentStubService()
    conversation = [_memory(f"旧问{index}", "", index + 1) for index in range(4)]
    conversation.append(_memory("新话题", "t1", 5))
    intake = AgentStubLLM(intake=_intake("continue", query="继续问", region="深圳"))
    node = region_mod.make_region_node(intake, client.laws())

    update = node(_state(conversation=list(conversation), topic_id="t1", turn_no=5))

    assert [content for role, content in update["history"] if role == "user"] == [
        "旧问2",
        "旧问3",
        "新话题",
    ]
    ask = intake.intake_prompts[0][-1]["content"]
    assert "[?] 更早 2 轮" in ask
    assert ask.index("[?] 更早 2 轮") > ask.index("仅列最近")


def test_without_an_active_topic_the_view_is_the_plain_window() -> None:
    client = AgentStubService()
    conversation = [_memory(f"第{index}问", "", index) for index in range(1, 5)]
    intake = AgentStubLLM(intake=_intake("new", query="全新问题", region="深圳"))
    node = region_mod.make_region_node(intake, client.laws())

    update = node(_state(conversation=list(conversation)))

    assert [content for role, content in update["history"] if role == "user"] == [
        "第2问",
        "第3问",
        "第4问",
    ]
    assert update["turn_no"] == 5
    assert update["topic_id"] == "t5"
    ask = intake.intake_prompts[0][-1]["content"]
    assert "当前没有活跃话题。" in ask
    assert "较早的其他话题" not in ask


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
    text = update["messages"][0]["content"]
    assert text.startswith("注意：这些段上一轮已经给过你了")
    assert "本轮没有新增材料段" in text
    assert "材料里没有别的相关内容" not in text, (
        "fresh 为空只说明这一查没带出未见过的新段，推不出「材料里没有别的相关内容」"
    )


def test_an_empty_search_gets_the_retry_note_not_the_already_given_note() -> None:
    client = AgentStubService()

    def nothing(question, top_k=None, **kwargs):
        return _retrieval(question, ())

    client.search = nothing
    call = _call("c1", "search_law", query=QUESTION)

    update = _tools_node(client)(_state(call))

    text = update["messages"][0]["content"]
    assert "没有命中的法条" in text
    assert "换一组更接近法条原文的关键词" in text
    assert "已给过你" not in text and "注意" not in text, (
        "0 命中不是「都已给过」：空结果集减 seen 还是空集，这里曾恒真地贴上重复警告，"
        "把「换词重试」这一停一续两句矛盾的话同时下发给模型"
    )


def test_a_repeat_search_warns_exactly_once() -> None:
    client = AgentStubService()
    call = _call("c1", "search_law", query=QUESTION)
    log = [{"articles": [{"parent_id": ARTICLE.parent_id}, {"parent_id": OTHER.parent_id}]}]

    update = _tools_node(client)(_state(call, search_log=log))

    text = update["messages"][0]["content"]
    assert "注意：本轮没有新增法条，与之前的检索重复。换个完全不同的角度，或直接结束检索。" in text
    assert text.count("注意") == 1, (
        "全旧条只该有一条合并文案（重复警告 + 停/换），渲染层与前缀各写一份时模型会收到双份"
    )
    assert "一条新证据都没取到" not in text


def test_a_missing_article_comes_back_as_the_service_note() -> None:
    client = AgentStubService()
    call = _call("c1", "get_article", article_no="第九十九条")

    update = _tools_node(client)(_state(call))

    assert client.lookups == [("第九十九条", None)]
    assert update["messages"][0]["content"] == "库里没有任何一部法规有第 99 条。"
    assert update["search_log"] == []


class _EventRecorder(Tracer):
    def __init__(self) -> None:
        self.events: list[dict] = []

    def observation(self, name: str, as_type: str = "span", **fields):
        self.events.append({"name": name, "as_type": as_type, **fields})
        return self


def test_an_identical_repeat_call_is_skipped_without_executing() -> None:
    client = AgentStubService()
    call = _call("c1", "search_law", query=QUESTION)
    node = _tools_node(client)

    first = node(_state(call))
    second = node(_state(call, call_keys=first["call_keys"]))

    assert len(first["call_keys"]) == 1
    assert len(client.searches) == 1
    assert second["call_keys"] == []
    assert "已跳过执行" in second["messages"][0]["content"]
    assert second["search_log"] == []


def test_web_search_repeats_are_never_skipped(monkeypatch) -> None:
    client = AgentStubService()
    fetched: list[str] = []

    def fake_fetch(query, *, start=1, count=None):
        fetched.append(query)
        return [], f"联网#{start}「{query}」"

    monkeypatch.setattr(handlers, "fetch_web", fake_fetch)
    call = _call("c1", "web_search", query="深圳 电动自行车 新规")
    node = _tools_node(client)

    first = node(_state(call))
    second = node(_state(call, call_keys=first["call_keys"]))

    assert first["call_keys"] == []
    assert fetched == ["深圳 电动自行车 新规", "深圳 电动自行车 新规"]
    assert "已跳过" not in second["messages"][0]["content"]


def test_a_failed_call_is_not_remembered_so_the_retry_still_runs() -> None:
    client = AgentStubService()
    attempts: list[str] = []
    original = client.search

    def flaky(question, top_k=None, **kwargs):
        attempts.append(question)
        if len(attempts) == 1:
            raise QaError("打不通 rag 服务：Connection refused")
        return original(question, top_k, **kwargs)

    client.search = flaky
    call = _call("c1", "search_law", query=QUESTION)
    node = _tools_node(client)

    first = node(_state(call))
    second = node(_state(call, call_keys=first["call_keys"]))

    assert "检索失败" in first["messages"][0]["content"]
    assert first["call_keys"] == []
    assert attempts == [QUESTION, QUESTION]
    assert "已跳过" not in second["messages"][0]["content"]
    assert len(second["search_log"]) == 1


def test_a_timeout_is_classified_apart_from_a_generic_failure() -> None:
    client = AgentStubService()

    def slow(question, top_k=None, **kwargs):
        raise QaTimeout("POST /qa 超时（4.5 秒）：ReadTimeout")

    client.search = slow
    cfg = config.AgentConfig(max_steps=2, tool_timeout=4.5)
    node = make_tools_node(client, cfg, observer=Tracer())

    update = node(_state(_call("c1", "search_law", query=QUESTION)))

    text = update["messages"][0]["content"]
    assert "检索超时（超过 4.5 秒仍未返回）" in text
    assert "检索失败" not in text
    assert update["call_keys"] == []


def test_every_retrieval_tool_carries_the_timeout_budget() -> None:
    client = AgentStubService()
    cfg = config.AgentConfig(max_steps=2, tool_timeout=4.5)
    node = make_tools_node(client, cfg, observer=Tracer())

    node(_state(_call("c1", "search_law", query=QUESTION)))
    node(_state(_call("c2", "get_article", article_no="第九十一条")))
    node(_state(_call("c3", "search_materials", query="培训费")))

    assert client.tool_timeouts == [
        ("search", 4.5),
        ("get_article", 4.5),
        ("search_materials", 4.5),
    ]


def test_blocked_calls_still_leave_one_event_each() -> None:
    client = AgentStubService()
    recorder = _EventRecorder()
    node = make_tools_node(client, config.AgentConfig(max_steps=2), observer=recorder)

    first = node(_state(_call("c1", "search_law", query=QUESTION)))
    node(
        _state(
            _call("c2", "拉格朗日插值"),
            _call("c3", "search_law", query=QUESTION, law_name="深圳经济特区停车管理条例"),
            _call("c4", "get_article"),
            _call("c5", "search_law", query=QUESTION),
            call_keys=first["call_keys"],
        )
    )

    names = [event["name"] for event in recorder.events if event["as_type"] == "span"]
    assert "未知工具" in names
    assert "法规无法确定" in names
    assert "参数不合法" in names
    assert "重复调用已跳过" in names
    unknown = next(event for event in recorder.events if event["name"] == "未知工具")
    assert unknown["input"]["tool"] == "拉格朗日插值"


def test_a_broken_article_lookup_comes_back_as_a_receipt() -> None:
    client = AgentStubService()

    def broken(article_no, law_name=None, *, timeout=None):
        raise QaError("打不通 rag 服务：Connection refused")

    client.get_article = broken

    update = _tools_node(client)(_state(_call("c1", "get_article", article_no="第九十一条")))

    assert "检索失败" in update["messages"][0]["content"]
    assert update["search_log"] == []


def test_a_broken_materials_lookup_comes_back_as_a_receipt() -> None:
    client = AgentStubService()

    def broken(query, doc_ids, *, top_k=5, timeout=None):
        raise QaError("打不通 rag 服务：Connection refused")

    client.search_materials = broken

    update = _tools_node(client)(_state(_call("c1", "search_materials", query="培训费")))

    assert "检索失败" in update["messages"][0]["content"]
    assert update["materials"] == []


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
    assert "stub-agent" in text and "stub-review" in text
    assert "入口理解（每轮 1 次 stub-agent：话题/意图/改写/判地区，判不出则回问或不限地区）" in text


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


def test_a_degraded_resume_refuses_with_a_clear_message() -> None:
    runner, _client = _runner(
        clarify=True, region="?", place="深圳", sessions=InMemorySaver()
    )
    status, _payload = runner.ask_payload(QUESTION, session_id="s1")

    assert status == "interrupted"
    runner.llm.available = False
    with pytest.raises(QaError) as excinfo:
        runner.resume("s1", {"region": "national"})

    assert str(excinfo.value) == RESUME_UNAVAILABLE_DETAIL


def test_a_degraded_resume_stream_refuses_with_the_same_message() -> None:
    runner, _client = _runner(
        clarify=True, region="?", place="深圳", sessions=InMemorySaver()
    )
    status, _payload = runner.ask_payload(QUESTION, session_id="s1")

    assert status == "interrupted"
    runner.llm.available = False
    with pytest.raises(QaError) as excinfo:
        list(runner.resume_stream("s1", {"region": "national"}))

    assert str(excinfo.value) == RESUME_UNAVAILABLE_DETAIL


def test_a_degraded_resume_reports_the_session_conflict_first() -> None:
    runner, _client = _runner(
        llm=AgentStubLLM(model="none", available=False), sessions=InMemorySaver()
    )

    with pytest.raises(ResumeConflict, match="没有等待澄清"):
        runner.resume("never-started", {"region": "national"})


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
        plan=PLAN_THEN_ANSWER, region="?", place="深圳", clarify=True, sessions=InMemorySaver()
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
        plan=PLAN_THEN_ANSWER, region="?", place="深圳", clarify=True, sessions=InMemorySaver()
    )

    list(runner.stream(QUESTION, session_id="s1"))

    status, payload = runner.resume("s1", {"region": "national"})

    assert status == "ok"
    assert NATIONAL_NOTE in payload["notes"]
