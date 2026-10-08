from __future__ import annotations

import json

import pytest
from conftest import AgentStubLLM

from rag_contracts import config
from rag_contracts.domain.answer import Question
from rag_contracts.domain.errors import QaError
from rag_contracts.observability.tracer import Tracer

pytest.importorskip("fastapi", reason="api extra 没装：agent 侧 HTTP 验收跳过")
pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")
pytest.importorskip("langgraph", reason='agent 循环要 pip install -e ".[agent]"')

from agent_service.agents import graph as graph_mod  # noqa: E402

QUESTION = "在深圳，醉酒驾驶机动车怎么处罚？"


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _call(call_id: str, name: str, **arguments) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [_tool_call(call_id, name, arguments)]}


def _region(text: str) -> list[dict]:
    return [{"role": "assistant", "content": json.dumps({"region": text}, ensure_ascii=False)}]


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


def _runner(remote, *, plan=None, region="深圳", llm=None):
    return graph_mod.AgentRunner(
        remote,
        cfg=config.AgentConfig(max_steps=2),
        llm=llm
        if llm is not None
        else AgentStubLLM(list(plan or PLAN_THEN_ANSWER), model="stub-agent", tokens=7),
        region_llm=AgentStubLLM(_region(region), model="stub-region"),
        review_llm=AgentStubLLM(_review(True), model="stub-review"),
        tracer=Tracer(),
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


def test_the_agent_stream_over_http_carries_steps_then_the_real_answer(agent_wire) -> None:
    remote, rt, _rag_app = agent_wire
    http = _agent_http(_runner(remote))

    with http.stream("POST", "/qa/stream", json={"question": QUESTION}) as resp:
        assert resp.status_code == 200
        frames = _sse_frames("".join(f"{line}\n" for line in resp.iter_lines()))

    assert [event for event, _payload in frames] == ["step"] * 6 + ["done"]
    assert [payload["node"] for event, payload in frames if event == "step"] == [
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
    assert done["request_ms"] >= 0
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

    for path in ("/health", "/qa", "/qa/stream"):
        payload = None if path == "/health" else {"question": QUESTION}
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
