from __future__ import annotations

import importlib.util
import os

from api_contracts import RagClient
from rag_contracts import config
from rag_contracts.domain.errors import QaError

__all__ = ["boot_agent_runner", "build_agent_runner", "build_client", "rag_base_url"]

BASE_URL_ENV = "RAG_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def rag_base_url() -> str:
    return os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL


def build_client(*, base_url: str | None = None, top_k: int | None = None) -> RagClient:
    return RagClient(base_url or rag_base_url(), top_k=top_k)


def build_agent_runner(client: RagClient, *, cfg=None, tracer=None):
    from .agents.graph import AgentRunner

    return AgentRunner.attach(client, cfg=cfg, tracer=tracer)


def boot_agent_runner(client: RagClient, *, cfg=None, tracer=None):
    health = client.health()
    if not health.get("milvus"):
        raise QaError(
            f"rag 服务 {client.base_url} 没连上 Milvus：先 " + config.COMPOSE + " up -d standalone"
        )
    if importlib.util.find_spec("langgraph") is None:
        raise QaError('未安装 langgraph：agent 这条路要 pip install -e ".[agent]"')
    runner = build_agent_runner(client, cfg=cfg, tracer=tracer)
    runner.graph()
    if not runner.llm.available:
        raise QaError("未配置 LLM_API_KEY：agent 这条路要走模型（只检索请打 rag 的 /qa?mode=search）")
    return runner
