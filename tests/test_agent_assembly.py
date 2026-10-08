from __future__ import annotations

import importlib
from dataclasses import replace

from conftest import AgentStubLLM

from agent_service import container
from agent_service.agents import graph as graph_mod
from rag_contracts import config
from rag_contracts.domain.errors import QaError
from rag_contracts.observability import langfuse as lt
from rag_contracts.observability.tracer import Tracer


def _attach(**kwargs):
    return graph_mod.AgentRunner.attach(object(), tracer=kwargs.pop("tracer", Tracer()), **kwargs)


class _Client:
    base_url = "http://stub"

    def __init__(self, *, milvus: str = "v2.6.24") -> None:
        self.milvus = milvus
        self.calls: list[str] = []

    def health(self):
        self.calls.append("health")
        return {"milvus": self.milvus}


def _boot(monkeypatch, *, milvus: str = "v2.6.24", llm=None):
    client = _Client(milvus=milvus)
    seen: dict = {}
    monkeypatch.setattr(graph_mod.AgentRunner, "attach", _spy_attach(seen))
    monkeypatch.setattr(graph_mod.AgentRunner, "graph", lambda self: seen.setdefault("graph"))
    monkeypatch.setattr(
        "rag_contracts.llm.build_llm",
        lambda *args, **kwargs: llm if llm is not None else AgentStubLLM(model="stub-agent"),
    )
    try:
        runner = container.boot_agent_runner(client, tracer=Tracer())
    except QaError as exc:
        return client, seen, str(exc)
    return client, seen, runner


def _spy_attach(seen: dict):
    real = graph_mod.AgentRunner.attach.__func__

    def spy(cls, client, **kwargs):
        seen["client"] = client
        return real(cls, client, **kwargs)

    return classmethod(spy)


def test_attach_never_puts_the_main_model_on_review(monkeypatch):
    main_cfg = replace(config.llm_config(), model="stub-main")
    region_cfg = replace(config.region_llm_config(), model="stub-region")
    review_cfg = replace(config.review_llm_config(), model="stub-review")
    assert len({main_cfg, region_cfg, review_cfg}) == 3
    monkeypatch.setattr(graph_mod.config, "llm_config", lambda: main_cfg)
    monkeypatch.setattr(graph_mod.config, "region_llm_config", lambda: region_cfg)
    monkeypatch.setattr(graph_mod.config, "review_llm_config", lambda: review_cfg)

    runner = _attach()

    assert runner.review_llm is not runner.llm
    assert runner.region_llm is not runner.llm
    assert (runner.llm.cfg, runner.region_llm.cfg, runner.review_llm.cfg) == (
        main_cfg,
        region_cfg,
        review_cfg,
    )


def test_attach_asks_the_env_when_tracer_is_omitted(monkeypatch):
    sentinel = Tracer()
    monkeypatch.setattr(lt, "from_env", lambda: sentinel)

    assert graph_mod.AgentRunner.attach(object(), tracer=None).tracer is sentinel
    quiet = Tracer()
    assert graph_mod.AgentRunner.attach(object(), tracer=quiet).tracer is quiet


def test_boot_probes_the_rag_service_then_builds_the_graph(monkeypatch):
    client, seen, runner = _boot(monkeypatch)

    assert client.calls == ["health"]
    assert seen["client"] is runner.rag
    assert "graph" in seen, "装配完不建图：第一条请求才发现模型/工具挂了"
    assert runner.review_llm is not runner.llm


def test_boot_stops_before_attaching_when_the_rag_service_has_no_milvus(monkeypatch):
    client, seen, failure = _boot(monkeypatch, milvus="")

    assert client.calls == ["health"]
    assert "没连上 Milvus" in failure
    assert "client" not in seen


def test_boot_stops_when_langgraph_is_missing(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        container.importlib.util,
        "find_spec",
        lambda name: None if name == "langgraph" else real(name),
    )

    _client, seen, failure = _boot(monkeypatch)

    assert "未安装 langgraph" in failure
    assert "client" not in seen


def test_boot_stops_when_the_model_has_no_key(monkeypatch):
    _client, seen, failure = _boot(monkeypatch, llm=AgentStubLLM(model="none", available=False))

    assert "未配置 LLM_API_KEY" in failure
    assert "client" in seen, "没有 key 是装配之后才发现的，不该连图都不建就退出"
