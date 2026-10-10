from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="api extra 没装：agent 侧 HTTP 验收跳过")
pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")
pytest.importorskip("langgraph", reason='agent 图要 pip install -e ".[agent]"')

from conftest import AgentStubLLM, AgentStubService  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent_service import storage  # noqa: E402
from agent_service.agents.graph import AgentRunner  # noqa: E402
from agent_service.api import app as server  # noqa: E402
from agent_service.api import budget  # noqa: E402
from rag_contracts import config  # noqa: E402
from rag_contracts.observability.tracer import Tracer  # noqa: E402

QUESTION = "在深圳，醉酒驾驶机动车怎么处罚？"

PLAN = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "search_law",
                    "arguments": json.dumps({"query": "醉酒驾驶机动车怎么处罚"}, ensure_ascii=False),
                },
            }
        ],
    },
    {"role": "assistant", "content": "够了"},
]


def _usage(*tokens: int) -> list[dict]:
    return [{"total_tokens": value} for value in tokens]


def _region_line(region: str, place: str = "") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": json.dumps({"region": region, "place": place}, ensure_ascii=False),
        }
    ]


def _review_line(*judgments: bool) -> list[dict]:
    payload = {"judgments": [{"n": n, "supported": ok} for n, ok in enumerate(judgments, start=1)]}
    return [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]


class _StubAnswer:
    def to_dict(self) -> dict:
        return {"answer": "答案正文"}


class _StubGraph:
    def __init__(self, *, chunks=(), error=None, invoke_result=None, state=None) -> None:
        self.chunks = list(chunks)
        self.error = error
        self.invoke_result = invoke_result
        self.state = state

    def stream(self, run_input, **kwargs):
        yield from self.chunks
        if self.error is not None:
            raise self.error

    def invoke(self, run_input, **kwargs):
        if self.error is not None:
            raise self.error
        return self.invoke_result

    def get_state(self, config_dict):
        return None if self.state is None else SimpleNamespace(values=self.state)


def _graph_runner(graph: _StubGraph) -> AgentRunner:
    runner = AgentRunner(AgentStubService(), llm=AgentStubLLM(), tracer=Tracer())
    runner._graph = graph
    return runner


def _real_runner() -> AgentRunner:
    return AgentRunner(
        AgentStubService(),
        llm=AgentStubLLM([dict(message) for message in PLAN], model="stub-agent"),
        region_llm=AgentStubLLM(_region_line("national"), model="stub-region"),
        review_llm=AgentStubLLM(_review_line(True) * 4, model="stub-review"),
        cfg=config.AgentConfig(max_steps=2),
        tracer=Tracer(),
    )


class _StubRagService:
    def health(self) -> dict:
        return {"status": "ok"}


class _StubRunner:
    def __init__(self) -> None:
        self.rag = _StubRagService()
        self.answer_calls = 0
        self.stream_calls = 0

    def ask_payload(self, question, *, material_ids, session_id):
        self.answer_calls += 1
        return "ok", {"question": question.text, "answer": "答案正文", "session_id": session_id}

    def resume(self, session_id, value):
        self.answer_calls += 1
        return "ok", {"answer": "答案正文", "session_id": session_id}

    def stream(self, question, *, material_ids, session_id):
        self.stream_calls += 1
        return iter(())

    def resume_stream(self, session_id, value):
        self.stream_calls += 1
        return iter(())


@pytest.fixture
def wire(monkeypatch):
    monkeypatch.delenv("AGENT_API_KEYS", raising=False)
    runner = _StubRunner()
    server.app.state.runner = runner
    server.app.state.boot_error = None
    return TestClient(server.app), runner


def test_the_utc_day_key_and_the_retry_window() -> None:
    assert budget.day_key(now=0) == "1970-01-01"
    assert budget.retry_after_seconds(now=0) == 86400
    assert budget.retry_after_seconds(now=86399) == 1
    assert 1 <= budget.retry_after_seconds() <= 86400


def test_budget_reexports_storage_names_and_keeps_the_wrapper_seam() -> None:
    assert budget.DB_FAIL_PREFIX is storage.DB_FAIL_PREFIX
    assert budget.day_key is storage.day_key
    assert budget.used_today is not storage.used_today
    assert budget.record_usage is not storage.record_usage


def test_the_ledger_is_created_on_demand_and_accumulates_one_row_per_day(tmp_path) -> None:
    target = tmp_path / "nested" / "usage.db"

    assert budget.record_state({"usage": _usage(7, 4)}, path=target) is True
    assert target.exists()
    assert budget.used_today(path=target) == (11, 2)
    assert budget.record_state({"usage": _usage(1)}, path=target) is True
    assert budget.used_today(path=target) == (12, 3)
    assert budget.record_state({"usage": []}, path=target) is True
    assert budget.record_state(None, path=target) is True
    assert budget.used_today(path=target) == (12, 3)


def test_budget_off_leaves_requests_and_the_disk_untouched(wire, monkeypatch, tmp_path) -> None:
    http, runner = wire
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", "0")

    resp = http.post("/qa", json={"question": QUESTION})

    assert resp.status_code == 200
    assert runner.answer_calls == 1
    assert not ledger.exists()


def test_the_gate_blocks_the_question_after_the_day_is_full(wire, monkeypatch, tmp_path) -> None:
    http, runner = wire
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", "100")
    budget.record_usage(99, 1, path=ledger)

    assert http.post("/qa", json={"question": QUESTION}).status_code == 200
    budget.record_usage(1, 1, path=ledger)

    blocked = http.post("/qa", json={"question": QUESTION})
    assert blocked.status_code == 429
    assert blocked.json()["detail"] == budget.BLOCKED_DETAIL.format(budget=100, used=100)
    assert blocked.headers["Retry-After"].isdigit()
    assert 1 <= int(blocked.headers["Retry-After"]) <= 86400
    assert runner.answer_calls == 1


def test_every_run_endpoint_shares_the_gate_streams_included(wire, monkeypatch, tmp_path) -> None:
    http, runner = wire
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", "10")
    budget.record_usage(10, 1, path=ledger)
    resume_body = {"session_id": "s1", "value": {"region": "national"}}

    for path, body in (
        ("/qa", {"question": QUESTION}),
        ("/qa/stream", {"question": QUESTION}),
        ("/qa/resume", resume_body),
        ("/qa/resume/stream", resume_body),
    ):
        resp = http.post(path, json=body)
        assert resp.status_code == 429, path
        assert resp.json()["detail"].startswith("日额度已用完"), path
        assert resp.headers["Retry-After"], path

    assert runner.answer_calls == 0
    assert runner.stream_calls == 0


def test_a_real_run_keeps_the_ledger_equal_to_its_own_usage_events(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    runner = _real_runner()

    state = runner.invoke(QUESTION)

    rows = [*state["usage"], *state["usage_extra"]]
    assert len(rows) >= 2
    assert budget.used_today(path=ledger) == (
        sum(int(row["total_tokens"]) for row in rows),
        len(rows),
    )


def _clarify_runner(sessions) -> AgentRunner:
    return AgentRunner(
        AgentStubService(),
        llm=AgentStubLLM([dict(message) for message in PLAN], model="stub-agent"),
        region_llm=AgentStubLLM(_region_line("national", place="深圳"), model="stub-region"),
        review_llm=AgentStubLLM(_review_line(True) * 4, model="stub-review"),
        cfg=config.AgentConfig(max_steps=2, clarify=True),
        tracer=Tracer(),
        sessions=sessions,
    )


def test_a_clarify_interrupt_records_the_region_call(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    runner = _clarify_runner(InMemorySaver())

    kind, _payload = runner.ask_payload(QUESTION, material_ids=(), session_id="s1")

    assert kind == "interrupted"
    assert budget.used_today(path=ledger) == (3, 1)


def test_a_resume_does_not_recount_the_interrupted_turn(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    runner = _clarify_runner(InMemorySaver())

    runner.ask_payload(QUESTION, material_ids=(), session_id="s1")
    assert budget.used_today(path=ledger) == (3, 1)

    status, _payload = runner.resume("s1", {"region": "national"})

    assert status == "ok"
    assert budget.used_today(path=ledger) == (21, 5)


def test_a_finished_run_records_the_sum_of_every_event(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    graph = _StubGraph(
        chunks=[
            ("values", {"usage": _usage(2)}),
            ("values", {"usage": _usage(2, 5, 4), "answer": _StubAnswer()}),
        ]
    )
    runner = _graph_runner(graph)

    events = list(runner.stream(QUESTION))

    assert events[-1][0] == "answer"
    assert budget.used_today(path=ledger) == (11, 3)


def test_an_interrupted_run_records_at_its_exit(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    graph = _StubGraph(
        chunks=[
            ("values", {"usage": _usage(3, 6)}),
            ("updates", {"__interrupt__": (object(),)}),
        ]
    )
    runner = _graph_runner(graph)

    events = list(runner.stream(QUESTION))

    assert [kind for kind, _payload in events] == ["interrupt"]
    assert budget.used_today(path=ledger) == (9, 2)


def test_a_failed_stream_run_records_what_was_already_committed(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    graph = _StubGraph(
        chunks=[("values", {"usage": _usage(4, 4, 1)})], error=RuntimeError("模型炸了")
    )
    runner = _graph_runner(graph)

    with pytest.raises(RuntimeError, match="模型炸了"):
        list(runner.stream(QUESTION))

    assert budget.used_today(path=ledger) == (9, 3)


def test_a_failed_sync_run_records_the_latest_checkpoint(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    graph = _StubGraph(error=RuntimeError("模型炸了"), state={"usage": _usage(8, 2)})
    runner = _graph_runner(graph)

    with pytest.raises(RuntimeError, match="模型炸了"):
        runner.ask_payload(QUESTION)

    assert budget.used_today(path=ledger) == (10, 2)


def test_a_failed_run_without_a_snapshot_writes_nothing(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    runner = _graph_runner(_StubGraph(error=RuntimeError("模型炸了"), state=None))

    with pytest.raises(RuntimeError, match="模型炸了"):
        runner.ask_payload(QUESTION)

    assert not ledger.exists()


def test_a_broken_ledger_is_printed_but_never_blocks(wire, monkeypatch, tmp_path, capsys) -> None:
    http, runner = wire
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    ledger = blocker / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", "50")

    assert budget.record_state({"usage": _usage(5)}) is False
    assert "[usage.db] 记账失败" in capsys.readouterr().err

    resp = http.post("/qa", json={"question": QUESTION})

    assert resp.status_code == 200
    assert runner.answer_calls == 1
    assert "[usage.db] 读取失败" in capsys.readouterr().err


def test_show_prints_the_day_line(monkeypatch, tmp_path, capsys) -> None:
    ledger = tmp_path / "usage.db"
    monkeypatch.setattr(budget, "db_path", lambda: ledger)
    monkeypatch.delenv("AGENT_DAILY_TOKEN_BUDGET", raising=False)
    budget.record_usage(7, 2, path=ledger)

    assert budget.main(["--show"]) == 0

    out = capsys.readouterr().out
    assert f"[usage.db] day={budget.day_key()}" in out
    assert "tokens=7" in out
    assert "calls=2" in out
    assert "budget=0" in out


def test_show_exits_nonzero_on_an_unreadable_ledger(monkeypatch, tmp_path, capsys) -> None:
    ledger = tmp_path / "usage.db"
    ledger.write_bytes(b"\x00garbage not sqlite" * 64)
    monkeypatch.setattr(budget, "db_path", lambda: ledger)

    assert budget.main(["--show"]) == 1
    assert "[usage.db] 读取失败" in capsys.readouterr().err
