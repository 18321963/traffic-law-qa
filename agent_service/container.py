from __future__ import annotations

import importlib.util
import os
import sys

from api_contracts import RagClient
from rag_contracts import config
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.laws import LawInfo

__all__ = ["boot_agent_runner", "build_agent_runner", "build_client", "rag_base_url"]

BASE_URL_ENV = "RAG_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def rag_base_url() -> str:
    return os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL


def build_client(*, base_url: str | None = None, top_k: int | None = None) -> RagClient:
    return RagClient(base_url or rag_base_url(), top_k=top_k)


def build_agent_runner(client: RagClient, *, cfg=None, tracer=None, sessions=None, laws=None):
    from .agents.graph import AgentRunner

    return AgentRunner.attach(client, cfg=cfg, tracer=tracer, sessions=sessions, laws=laws)


def _open_sessions(path: str):
    from .agents.session import open_sessions

    try:
        return open_sessions(path)
    except Exception as exc:  # noqa: BLE001
        raise QaError(f"会话存储 {path} 打不开（{type(exc).__name__}: {exc}）") from exc


def _laws_at_boot(client: RagClient) -> list[LawInfo]:
    try:
        return list(client.laws())
    except Exception as exc:  # noqa: BLE001
        print(
            f"[agent] 取法条清单失败（rag GET /laws）：{type(exc).__name__}: {exc} —— "
            "先按空清单起，入口判地区时法规清单会是空的",
            file=sys.stderr,
        )
        return []


def boot_agent_runner(client: RagClient, *, cfg=None, tracer=None):
    health = client.health()
    if not health.get("milvus"):
        raise QaError(
            f"rag 服务 {client.base_url} 没连上 Milvus：先 " + config.COMPOSE + " up -d standalone"
        )
    if importlib.util.find_spec("langgraph") is None:
        raise QaError('未安装 langgraph：agent 这条路要 pip install -e ".[agent]"')
    agent_cfg = cfg or config.agent_config()
    sessions = _open_sessions(agent_cfg.session_db)
    laws = _laws_at_boot(client)
    runner = build_agent_runner(client, cfg=agent_cfg, tracer=tracer, sessions=sessions, laws=laws)
    runner.graph()
    if not runner.llm.available:
        raise QaError("未配置 LLM_API_KEY：agent 这条路要走模型（只检索请打 rag 的 /qa?mode=search）")
    return runner
