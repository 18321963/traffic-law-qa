from __future__ import annotations

from dataclasses import replace

from traffic_law_qa import config
from traffic_law_qa.agents import graph as graph_mod
from traffic_law_qa.observability import langfuse as lt
from traffic_law_qa.observability.tracer import Tracer


def _attach(**kwargs):
    return graph_mod.AgentRunner.attach(object(), tracer=kwargs.pop("tracer", Tracer()), **kwargs)


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


def test_boot_agent_runner_is_gate_then_rag_then_attach(monkeypatch):
    from traffic_law_qa import container

    calls: list[str] = []
    seen: dict = {}
    monkeypatch.setattr(container, "readiness", lambda **kw: calls.append("gate") or object())
    monkeypatch.setattr(container, "build_rag", lambda **kw: calls.append("rag") or object())
    real_attach = graph_mod.AgentRunner.attach.__func__

    def spy_attach(cls, rag, **kwargs):
        calls.append("attach")
        seen["rag"] = rag
        seen["tracer"] = kwargs.get("tracer")
        return real_attach(cls, rag, **kwargs)

    monkeypatch.setattr(graph_mod.AgentRunner, "attach", classmethod(spy_attach))
    runner = container.boot_agent_runner(tracer=Tracer())
    assert calls == ["gate", "rag", "attach"]
    assert seen["rag"] is runner.rag
    assert seen["tracer"] is not None
    assert runner.review_llm is not runner.llm
