from __future__ import annotations

from dataclasses import replace

from traffic_law_qa import config
from traffic_law_qa.agents import graph as graph_mod
from traffic_law_qa.agents import langfuse_tracer as lt
from traffic_law_qa.obs import Tracer


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


def test_load_is_attach_after_the_gate(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(graph_mod, "ensure_ready", lambda **kw: calls.append("gate") or object())
    seen: dict = {}

    def fake_rag_load(**kwargs):
        calls.append("rag")
        return object()

    monkeypatch.setattr(graph_mod.LegalRAG, "load", staticmethod(fake_rag_load))
    real_attach = graph_mod.AgentRunner.attach.__func__

    def spy_attach(cls, rag, **kwargs):
        calls.append("attach")
        seen["rag"] = rag
        seen["tracer"] = kwargs.get("tracer")
        return real_attach(cls, rag, **kwargs)

    monkeypatch.setattr(graph_mod.AgentRunner, "attach", classmethod(spy_attach))
    runner = graph_mod.AgentRunner.load(tracer=Tracer())
    assert calls == ["gate", "rag", "attach"]
    assert seen["rag"] is runner.rag
    assert runner.review_llm is not runner.llm
    assert runner.rag is not None
