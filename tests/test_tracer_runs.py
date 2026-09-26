from __future__ import annotations

import threading

from traffic_law_qa.observability.langfuse import LangfuseTracer
from traffic_law_qa.observability.tracer import Recorder, Tracer


class _FakeObs:
    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.trace_id = f"trace-{index}"
        self.id = f"span-{index}"
        self.updates: list[dict] = []

    def update(self, **fields) -> None:
        self.updates.append(fields)

    def __enter__(self) -> "_FakeObs":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _FakeClient:

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def start_as_current_observation(self, *, name, as_type, trace_context=None, **fields):
        with self._lock:
            obs = _FakeObs(name, len(self.calls))
            self.calls.append(
                {"name": name, "as_type": as_type, "trace_context": trace_context, "obs": obs}
            )
        return obs

    def flush(self) -> None:
        pass

    def of(self, name: str) -> list[dict]:
        return [call for call in self.calls if call["name"] == name]


def _tracer() -> tuple[LangfuseTracer, _FakeClient]:
    client = _FakeClient()
    return LangfuseTracer(client=client), client


def test_each_run_gets_its_own_root():
    tracer, client = _tracer()
    for _ in range(2):
        with tracer.span("invoke", root=True) as span:
            with tracer.span("node.agent"):
                pass
            span.record({"question": "醉驾怎么处罚"}, {"answer": "…"})

    runs, nodes = client.of("invoke"), client.of("node.agent")
    assert len(runs) == 2 and len(nodes) == 2
    assert [run["trace_context"] for run in runs] == [None, None]
    assert runs[0]["obs"].trace_id != runs[1]["obs"].trace_id
    for run, node in zip(runs, nodes):
        assert node["trace_context"] == {"trace_id": run["obs"].trace_id, "parent_span_id": run["obs"].id}
    renamed = [update for update in runs[0]["obs"].updates if "name" in update]
    assert renamed and renamed[0]["name"].startswith("invoke · 醉驾怎么处罚")


def test_concurrent_runs_do_not_share_a_root():
    tracer, client = _tracer()
    both_open = threading.Barrier(2)

    def run() -> None:
        with tracer.span("invoke", root=True):
            both_open.wait(timeout=5)
            with tracer.span("node.agent"):
                pass

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    runs, nodes = client.of("invoke"), client.of("node.agent")
    assert len(runs) == 2 and len(nodes) == 2
    assert {node["trace_context"]["trace_id"] for node in nodes} == {
        run["obs"].trace_id for run in runs
    }


def test_a_node_outside_any_run_does_not_invent_a_root():
    tracer, client = _tracer()
    with tracer.span("node.agent"):
        pass
    assert client.calls[0]["trace_context"] is None


def test_the_root_is_released_even_when_the_run_raises():
    tracer, client = _tracer()
    try:
        with tracer.span("invoke", root=True):
            raise RuntimeError("模型调用失败")
    except RuntimeError:
        pass
    with tracer.span("invoke", root=True):
        with tracer.span("node.agent"):
            pass
    runs, nodes = client.of("invoke"), client.of("node.agent")
    assert [run["trace_context"] for run in runs] == [None, None]
    assert nodes[0]["trace_context"]["trace_id"] == runs[1]["obs"].trace_id


def test_observation_carries_the_as_type_and_never_anchors():
    tracer, client = _tracer()
    with tracer.span("invoke", root=True):
        with tracer.observation("llm.agent", as_type="generation", model="stub"):
            pass
    assert client.of("llm.agent")[0]["as_type"] == "generation"
    assert client.of("llm.agent")[0]["trace_context"] is None


def test_empty_backends_accept_the_boundary_flag():
    with Tracer().span("invoke", root=True) as span:
        span.record({"question": "q"}, {"answer": "a"})
    recorder = Recorder()
    with recorder.span("invoke", root=True):
        pass
    assert [row["name"] for row in recorder.spans] == ["invoke"]
