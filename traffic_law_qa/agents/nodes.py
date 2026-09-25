from __future__ import annotations

from dataclasses import replace

from .. import config
from ..contracts import MaterialPassage, ParentChunk, Question, RetrievalResult, WebFinding
from ..obs import Tracer
from ..qa.rag import LegalRAG
from ..services.llm import ToolCallingLLM
from ..tools.arguments import tool_message
from ..tools.documents import load_materials
from ..tools.handlers import HANDLERS, ToolCall, ToolEnv, retrieval_digest
from ..tools.merge import merge_retrievals
from ..tools.schemas import (
    GET_ARTICLE_NAME,
    GET_ARTICLE_TOOL,
    SEARCH_LAW_NAME,
    SEARCH_LAW_TOOL,
    SEARCH_MATERIALS_NAME,
    SEARCH_MATERIALS_TOOL,
    WEB_SEARCH_NAME,
)
from .prompts import AGENT_SYSTEM_PROMPT
from .state import AgentState

__all__ = [
    "TOOLS",
    "make_agent_node",
    "make_tools_node",
    "make_finalize_node",
]

TOOLS = [SEARCH_LAW_TOOL, GET_ARTICLE_TOOL, SEARCH_MATERIALS_TOOL]

_NO_EVIDENCE_STOP_NUDGE = (
    "你这一轮没有调用任何工具，而手上一条证据都还没有 ——「够了」这个判断此刻无凭无据，"
    "循环只会把它读成「模型说够了」直接去作答。请调用 search_law 查一次"
    "（规则 2 的问法：一字不改地用用户的问题原文）。"
)


def _sum_usage(first: dict, second: dict) -> dict:
    if not first or not second:
        return first or second or {}
    return {key: (first.get(key) or 0) + (second.get(key) or 0) for key in set(first) | set(second)}


def _unearned_stop(reply: dict, state: AgentState) -> bool:
    return not reply.get("tool_calls") and not state.get("search_log")


def make_agent_node(llm: ToolCallingLLM, cfg: config.AgentConfig, laws: list[str]):

    laws_block = "\n".join(f"- {name}" for name in laws)
    law_count = len(laws)

    def agent_node(state: AgentState) -> dict:
        max_steps = state.get("max_steps", cfg.max_steps)
        prompt: list[dict] = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT
                % {"max_steps": max_steps, "law_count": law_count, "laws": laws_block},
            }
        ]
        prompt.extend({"role": r, "content": c} for r, c in state.get("history") or ())
        prompt.append({"role": "user", "content": state["question"]})
        prompt.extend(state.get("messages") or ())

        reply, usage = llm.chat(
            prompt, tools=TOOLS, temperature=cfg.temperature, name="llm.agent"
        )

        if _unearned_stop(reply, state):
            prompt.append(reply)
            prompt.append({"role": "user", "content": _NO_EVIDENCE_STOP_NUDGE})
            reply, extra = llm.chat(
                prompt, tools=TOOLS, temperature=cfg.temperature, name="llm.agent"
            )
            usage = _sum_usage(usage, extra)

        return {"messages": [reply], "steps": 1, "usage": [usage]}

    return agent_node


def make_tools_node(
    rag: LegalRAG,
    cfg: config.AgentConfig,
    index: dict[tuple[str, int], ParentChunk],
    observer: Tracer | None = None,
):

    observer = observer or Tracer()

    def tools_node(state: AgentState) -> dict:
        pending = (state.get("messages") or [{}])[-1].get("tool_calls") or []
        default_top_k = state.get("top_k", config.retrieve_config().top_k)
        seen = {
            article["parent_id"]
            for row in state.get("search_log") or ()
            for article in row.get("articles") or ()
        }

        out_messages: list[dict] = []
        out_logs: list[dict] = []
        out_web: list[WebFinding] = []
        out_materials: list[MaterialPassage] = []
        seen_labels = {m.label for m in state.get("materials") or ()}
        loaded: list[MaterialPassage] | None = None

        def materials() -> tuple[MaterialPassage, ...]:
            nonlocal loaded
            if loaded is None:
                loaded = list(load_materials(state.get("material_ids") or ()))
            return tuple(loaded)

        def position() -> int:
            return len(state.get("search_log") or ()) + len(out_logs) + 1

        def web_position() -> int:
            return len(state.get("external") or ()) + len(out_web) + 1

        env = ToolEnv(
            rag=rag, index=index, cfg=cfg, observer=observer, materials=materials
        )

        for call in pending:
            call_id = call.get("id", "")
            function = call.get("function") or {}
            name = function.get("name", "")

            handler = HANDLERS.get(name)
            if handler is None:
                out_messages.append(
                    tool_message(
                        call_id,
                        f"未知工具：{name}（本图挂了 {'、'.join(sorted(HANDLERS))}）",
                    )
                )
                continue

            outcome = handler(
                env,
                ToolCall(
                    arguments=function.get("arguments") or "",
                    retrieval_no=position(),
                    web_no=web_position(),
                    seen=frozenset(seen),
                    seen_labels=frozenset(seen_labels),
                    region_scope=tuple(state.get("region_scope") or ()),
                    default_top_k=default_top_k,
                ),
            )
            out_messages.append(tool_message(call_id, outcome.text))
            if outcome.retrieval is not None:
                row = outcome.retrieval.to_dict()
                out_logs.append(row)
                seen |= {article["parent_id"] for article in row["articles"]}
            out_web.extend(outcome.findings)
            out_materials.extend(outcome.passages)
            seen_labels.update(passage.label for passage in outcome.passages)

        return {
            "messages": out_messages,
            "search_log": out_logs,
            "external": out_web,
            "materials": out_materials,
        }

    return tools_node


def _trajectory_notes(state: AgentState, merged: RetrievalResult, cfg: config.AgentConfig) -> list[str]:
    calls = [
        (call.get("function") or {}).get("name") or ""
        for message in state.get("messages") or ()
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or ()
    ]
    plans = len(state.get("usage") or ())
    searches = calls.count(SEARCH_LAW_NAME)
    lookups = calls.count(GET_ARTICLE_NAME)
    webs = calls.count(WEB_SEARCH_NAME)
    materials = calls.count(SEARCH_MATERIALS_NAME)
    steps = state.get("steps", 0)
    max_steps = state.get("max_steps", cfg.max_steps)

    parts = [f"{plans} 轮 LLM 规划", f"{searches} 次检索"]
    if lookups:
        parts.append(f"{lookups} 次精确取条")
    if webs:
        parts.append(f"{webs} 次联网检索（{len(state.get('external') or ())} 条时效提示）")
    if materials:
        parts.append(f"{materials} 次材料检索（{len(state.get('materials') or ())} 段）")
    parts.append(f"证据 {len(merged.articles)} 条")
    notes = ["Agent：" + " / ".join(parts)]

    messages = state.get("messages") or ()
    last = messages[-1] if messages else {}
    if steps >= max_steps and last.get("role") != "assistant":
        notes.append(f"已达最大轮数 {max_steps}，强制进入作答")
    tokens = sum(int(row.get("total_tokens") or 0) for row in state.get("usage") or ())
    if tokens:
        notes.append(f"规划轮 token 合计 {tokens}")
    return notes


def make_finalize_node(
    rag: LegalRAG,
    cfg: config.AgentConfig,
    observer: Tracer | None = None,
):

    observer = observer or Tracer()

    def finalize_node(state: AgentState) -> dict:
        active_top_k = state.get("top_k", config.retrieve_config().top_k)
        logs = list(state.get("search_log") or ())
        extra: list[dict] = []
        notes: list[str] = []

        if not logs:
            with observer.observation(
                "兜底检索",
                "retriever",
                input={"query": state["question"], "top_k": active_top_k},
            ) as span:
                result = rag.search(state["question"], top_k=active_top_k)
                span.update(output=retrieval_digest(result))
            logs = [result.to_dict()]
            extra = list(logs)
            notes.append("本轮未取到任何证据（未调用工具，或工具调用全部失败），已按单轮管道兜底检索一次")

        merged = merge_retrievals(
            logs,
            question=state["question"],
            parents=rag.parents,
            max_evidence=cfg.max_evidence or active_top_k,
        )
        notes.extend(_trajectory_notes(state, merged, cfg))

        question = Question(
            text=state["question"],
            history=tuple(tuple(pair) for pair in state.get("history") or ()),
            top_k=active_top_k,
        )
        answer = rag.answer(
            question,
            merged,
            timeliness=tuple(state.get("external") or ()),
            materials=tuple(state.get("materials") or ()),
        )
        return {"answer": replace(answer, notes=answer.notes + tuple(notes)), "search_log": extra}

    return finalize_node
