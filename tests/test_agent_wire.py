from __future__ import annotations

import json

import pytest
from conftest import PENALTY, AgentStubLLM

from rag_contracts import config
from rag_contracts.domain.answer import Question
from rag_contracts.domain.errors import QaError
from rag_contracts.observability.tracer import Tracer

pytest.importorskip("fastapi", reason="api extra 没装：agent 侧 HTTP 验收跳过")
pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")
pytest.importorskip("langgraph", reason='agent 循环要 pip install -e ".[agent]"')

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent_service.agents import graph as graph_mod  # noqa: E402
from agent_service.prompts import RESUME_UNAVAILABLE_DETAIL  # noqa: E402

QUESTION = "在深圳，醉酒驾驶机动车怎么处罚？"


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _call(call_id: str, name: str, **arguments) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [_tool_call(call_id, name, arguments)]}


def _region(text: str, place: str = "") -> list[dict]:
    payload = {"region": text, "place": place}
    return [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]


def _review(*judgments: bool) -> list[dict]:
    payload = {"judgments": [{"n": n, "supported": ok} for n, ok in enumerate(judgments, start=1)]}
    return [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]


PLAN_THEN_ANSWER = [
    _call("c1", "search_law", query="醉酒驾驶机动车怎么处罚"),
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


def _runner(remote, *, plan=None, region=("深圳", ""), llm=None, sessions=None, clarify=False):
    return graph_mod.AgentRunner(
        remote,
        cfg=config.AgentConfig(max_steps=2, clarify=clarify),
        llm=llm
        if llm is not None
        else AgentStubLLM(list(plan or PLAN_THEN_ANSWER), model="stub-agent", tokens=7),
        region_llm=AgentStubLLM(_region(*region), model="stub-region"),
        review_llm=AgentStubLLM(_review(True), model="stub-review"),
        tracer=Tracer(),
        sessions=sessions,
    )


def _agent_http(runner):
    from fastapi.testclient import TestClient

    from agent_service.api import app as server

    server.app.state.runner = runner
    server.app.state.boot_error = None
    return TestClient(server.app)


def _unbooted_http(boot_error: str):
    from fastapi.testclient import TestClient

    from agent_service.api import app as server

    server.app.state.runner = None
    server.app.state.boot_error = boot_error
    return TestClient(server.app)


def _sse_frames(text: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line[len("event: ") :] for line in lines if line.startswith("event: "))
        data = next(line[len("data: ") :] for line in lines if line.startswith("data: "))
        frames.append((event, json.loads(data)))
    return frames


def _post_stream(http, path: str, body: dict) -> list[tuple[str, dict]]:
    with http.stream("POST", path, json=body) as resp:
        assert resp.status_code == 200, path
        return _sse_frames("".join(f"{line}\n" for line in resp.iter_lines()))


def _nodes(frames: list[tuple[str, dict]]) -> list[str]:
    return [payload["node"] for event, payload in frames if event == "step"]


def test_the_agent_stream_over_http_carries_steps_then_the_real_answer(agent_wire) -> None:
    remote, rt, _rag_app = agent_wire
    http = _agent_http(_runner(remote))

    with http.stream("POST", "/qa/stream", json={"question": QUESTION}) as resp:
        assert resp.status_code == 200
        frames = _sse_frames("".join(f"{line}\n" for line in resp.iter_lines()))

    kinds = [event for event, _payload in frames]
    assert [event for event in kinds if event != "delta"] == ["step"] * 6 + ["done"]
    assert _nodes(frames) == [
        "region",
        "agent",
        "tools",
        "agent",
        "finalize",
        "review",
    ]
    done = frames[-1][1]
    assert done["question"] == QUESTION
    assert done["answer"] == "答案正文[依据1]"
    assert done["citations"] == [
        "《中华人民共和国道路交通安全法》第九十一条",
        "《深圳经济特区道路交通安全违法行为处罚条例》第九十条",
    ]
    assert done["review"]["passed"] is True and done["review"]["total"] == 1
    assert "桩生成器：没花真调用" in done["notes"]
    assert any(note.startswith("Agent：") for note in done["notes"])
    assert "mode" not in done
    assert done["session_id"] is None
    assert done["request_ms"] >= 0
    deltas = [payload["text"] for event, payload in frames if event == "delta"]
    assert deltas == ["答案", "正文[依据1]"]
    assert "".join(deltas) == done["answer"]
    first_delta = kinds.index("delta")
    assert kinds[:first_delta].count("step") == 4
    assert "finalize" not in _nodes(frames[:first_delta])
    assert rt.rag.answers == []
    assert rt.rag.searches[0]["law_filter"] == ("road_traffic_safety", "shenzhen_penalty")
    assert rt.rag.searches[0]["top_k"] == 6


def test_the_answer_behind_the_wire_drops_the_article_but_the_retrieval_keeps_it(agent_wire) -> None:
    remote, rt, _rag_app = agent_wire

    answer = _runner(remote).ask(Question(text=QUESTION, top_k=4), material_ids=["d0"])

    assert [evidence.article for evidence in answer.evidences] == [None, None]
    assert all(evidence.text for evidence in answer.evidences)
    assert answer.retrieval is not None
    article = answer.retrieval.articles[0].article
    assert article is not None and article.article_no == "第九十一条" and len(article.text) > 100
    assert answer.model == "stub-model"
    assert "桩生成器：没花真调用" in answer.notes
    assert any(note.startswith("Agent：") for note in answer.notes)
    assert len(rt.rag.searches) == 1, (
        "作答这一步又去检索了一次 —— /answer 的语义是「我已经检索好了，你只负责生成」"
    )
    assert len(rt.rag.answers) == 1
    assert rt.rag.answers[0]["question"].text == QUESTION
    assert [hit.citation for hit in rt.rag.answers[0]["retrieval"].articles] == [
        "《中华人民共和国道路交通安全法》第九十一条",
        "《深圳经济特区道路交通安全违法行为处罚条例》第九十条",
    ]


def test_the_session_materials_travel_from_the_endpoint_into_the_tool(agent_wire) -> None:
    remote, rt, _rag_app = agent_wire
    http = _agent_http(_runner(remote, plan=list(MATERIAL_THEN_ANSWER)))

    payload = http.post("/qa", json={"question": QUESTION, "doc_ids": ["d7", "d8"]}).json()

    assert rt.rag.material_calls == [("培训费", ("d7", "d8"), 5)]
    assert len(rt.rag.answers[0]["materials"]) == 1
    assert [item["doc_id"] for item in payload["materials"]] == ["d0"]
    assert any("材料检索" in note for note in payload["notes"])
    assert payload["citations"] == [
        "《中华人民共和国道路交通安全法》第九十一条",
        "《深圳经济特区道路交通安全违法行为处罚条例》第九十条",
    ]


def test_the_agent_health_passes_through_the_rag_health(agent_wire) -> None:
    remote, rt, _rag_app = agent_wire
    http = _agent_http(_runner(remote))

    payload = http.get("/health").json()

    assert set(payload) == {"status", "version", "rag"}
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert payload["rag"]["status"] == "ok"
    assert payload["rag"]["milvus"] == "v2.6.24"
    assert payload["rag"]["laws"] == 2
    assert payload["rag"]["llm_ready"] is True
    assert rt.rag.llm_ready is True


def test_every_endpoint_hands_back_the_boot_error_while_the_agent_is_down() -> None:
    boot_error = '未安装 langgraph：agent 这条路要 pip install -e ".[agent]"'
    http = _unbooted_http(boot_error)

    for path in ("/health", "/qa", "/qa/stream", "/qa/resume", "/qa/resume/stream"):
        if path == "/health":
            payload = None
        elif path.startswith("/qa/resume"):
            payload = {"session_id": "s1", "value": {"region": "深圳"}}
        else:
            payload = {"question": QUESTION}
        resp = http.request("GET" if payload is None else "POST", path, json=payload)

        assert resp.status_code == 503, path
        assert resp.json()["detail"] == boot_error, path


def test_the_agent_health_says_degraded_when_the_rag_service_is_unreachable(
    agent_wire, monkeypatch
) -> None:
    remote, _rt, _rag_app = agent_wire
    http = _agent_http(_runner(remote))

    def unreachable():
        raise QaError("打不通 rag 服务：Connection refused")

    monkeypatch.setattr(remote, "health", unreachable)

    payload = http.get("/health").json()

    assert payload["status"] == "degraded"
    assert payload["rag"]["status"] == "unreachable"
    assert payload["rag"]["detail"] == "打不通 rag 服务：Connection refused"


def test_the_fallback_asks_the_service_to_search_and_generate(agent_wire) -> None:
    remote, rt, _rag_app = agent_wire
    runner = _runner(remote, llm=AgentStubLLM(model="none", available=False))

    assert runner.top_k == remote.top_k
    answer = runner.ask(QUESTION)

    assert [call["question"] for call in rt.rag.searches] == [QUESTION]
    assert rt.rag.answers[0]["question"].text == QUESTION
    assert answer.text == "答案正文[依据1]"
    assert answer.retrieval is not None and len(answer.retrieval.articles) == 2


def test_a_place_question_pauses_the_wire_stream_with_an_interrupt(agent_wire) -> None:
    remote, _rt, _rag_app = agent_wire
    http = _agent_http(
        _runner(remote, region=("深圳", "深圳"), clarify=True, sessions=InMemorySaver())
    )

    frames = _post_stream(http, "/qa/stream", {"question": QUESTION, "session_id": "s1"})

    kinds = [event for event, _payload in frames]
    assert [event for event in kinds if event != "delta"] == ["step", "interrupt"]
    assert _nodes(frames) == ["region"]
    payload = frames[-1][1]
    assert payload["session_id"] == "s1"
    assert payload["interrupt"]["value"]["type"] == "region_clarify"
    assert payload["interrupt"]["value"]["place"] == "深圳"
    assert payload["interrupt"]["value"]["laws"] == [PENALTY]
    assert "done" not in kinds


def test_the_wire_resume_stream_closes_the_clarified_session(agent_wire) -> None:
    remote, _rt, _rag_app = agent_wire
    http = _agent_http(
        _runner(remote, region=("深圳", "深圳"), clarify=True, sessions=InMemorySaver())
    )

    primed = _post_stream(http, "/qa/stream", {"question": QUESTION, "session_id": "s1"})
    assert primed[-1][0] == "interrupt"

    frames = _post_stream(
        http, "/qa/resume/stream", {"session_id": "s1", "value": {"region": "national"}}
    )

    kinds = [event for event, _payload in frames]
    assert [event for event in kinds if event != "delta"] == ["step"] * 6 + ["done"]
    assert _nodes(frames) == ["clarify", "agent", "tools", "agent", "finalize", "review"]
    done = frames[-1][1]
    assert done["question"] == QUESTION
    assert done["answer"] == "答案正文[依据1]"
    assert done["session_id"] == "s1"


def test_the_wire_resume_answers_once_and_then_conflicts(agent_wire) -> None:
    remote, _rt, _rag_app = agent_wire
    http = _agent_http(
        _runner(remote, region=("深圳", "深圳"), clarify=True, sessions=InMemorySaver())
    )
    _post_stream(http, "/qa/stream", {"question": QUESTION, "session_id": "s1"})
    body = {"session_id": "s1", "value": {"region": "national"}}

    answered = http.post("/qa/resume", json=body)

    assert answered.status_code == 200
    payload = answered.json()
    assert payload["status"] == "ok"
    assert payload["answer"] == "答案正文[依据1]"
    assert payload["session_id"] == "s1"
    assert payload["request_ms"] >= 0

    conflict = http.post("/qa/resume", json=body)

    assert conflict.status_code == 409
    assert "没有等待澄清" in conflict.json()["detail"]

    frames = _post_stream(http, "/qa/resume/stream", body)

    assert [event for event, _payload in frames] == ["error"]
    assert "没有等待澄清" in frames[-1][1]["message"]


def test_the_wire_resume_refuses_when_the_planning_model_is_missing(agent_wire) -> None:
    remote, _rt, _rag_app = agent_wire
    saver = InMemorySaver()
    http = _agent_http(
        _runner(remote, region=("深圳", "深圳"), clarify=True, sessions=saver)
    )
    primed = _post_stream(http, "/qa/stream", {"question": QUESTION, "session_id": "s1"})
    assert primed[-1][0] == "interrupt"

    http = _agent_http(
        _runner(
            remote,
            llm=AgentStubLLM(model="none", available=False),
            region=("深圳", "深圳"),
            clarify=True,
            sessions=saver,
        )
    )
    body = {"session_id": "s1", "value": {"region": "national"}}

    refused = http.post("/qa/resume", json=body)

    assert refused.status_code == 503
    assert refused.json()["detail"] == RESUME_UNAVAILABLE_DETAIL

    frames = _post_stream(http, "/qa/resume/stream", body)

    assert [event for event, _payload in frames] == ["error"]
    assert RESUME_UNAVAILABLE_DETAIL in frames[-1][1]["message"]


def test_the_resume_body_needs_a_session_and_a_non_empty_region(agent_wire) -> None:
    remote, _rt, _rag_app = agent_wire
    http = _agent_http(_runner(remote))

    for body in (
        {"session_id": "s1", "value": {}},
        {"session_id": "s1", "value": {"region": ""}},
        {"session_id": "s1", "value": None},
        {"session_id": "s1"},
    ):
        resp = http.post("/qa/resume", json=body)
        assert resp.status_code == 422, body


def test_the_wire_remembers_the_first_round_under_one_session_id(agent_wire) -> None:
    remote, _rt, _rag_app = agent_wire
    runner = _runner(remote, plan=list(PLAN_THEN_ANSWER) * 2, sessions=InMemorySaver())
    http = _agent_http(runner)

    first = http.post("/qa", json={"question": QUESTION, "session_id": "s1"})
    second = http.post("/qa", json={"question": "那记分呢？", "session_id": "s1"})

    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["session_id"] == "s1"
    prompt = runner.llm.prompts[2]
    assert {"role": "user", "content": QUESTION} in prompt
    assert {"role": "assistant", "content": "答案正文[依据1]"} in prompt


def test_the_wire_stream_falls_back_to_the_service_without_an_llm(agent_wire) -> None:
    remote, _rt, _rag_app = agent_wire
    http = _agent_http(
        _runner(remote, llm=AgentStubLLM(model="none", available=False), sessions=InMemorySaver())
    )

    frames = _post_stream(http, "/qa/stream", {"question": QUESTION, "session_id": "s1"})

    kinds = [event for event, _payload in frames]
    assert [event for event in kinds if event != "delta"] == ["done"]
    assert frames[-1][1]["session_id"] == "s1"


def test_a_runner_without_sessions_refuses_a_session_id_on_the_wire(agent_wire) -> None:
    remote, _rt, _rag_app = agent_wire
    http = _agent_http(_runner(remote))

    refused = http.post("/qa", json={"question": QUESTION, "session_id": "s1"})

    assert refused.status_code == 400
    assert "没接会话存储" in refused.json()["detail"]

    frames = _post_stream(http, "/qa/stream", {"question": QUESTION, "session_id": "s1"})

    assert [event for event, _payload in frames] == ["error"]
    assert "没接会话存储" in frames[-1][1]["message"]
