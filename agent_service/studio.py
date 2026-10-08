from __future__ import annotations

from dataclasses import replace

from agent_service import container
from rag_contracts import config

__all__ = ["graph"]


def graph():
    cfg = replace(config.agent_config(), clarify=True)
    runner = container.build_agent_runner(container.build_client(), cfg=cfg)
    return runner.graph()
