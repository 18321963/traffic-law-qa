from __future__ import annotations

from dataclasses import replace
from typing import Any

from .. import config
from ..contracts import MaterialPassage, ParentChunk, Question, RetrievalResult, WebFinding
from ..obs import Tracer
from ..qa.rag import LegalRAG
from ..services.llm import ToolCallingLLM
from ..tools.arguments import (
    parse_article_arguments,
    parse_material_arguments,
    parse_tool_arguments,
    parse_web_arguments,
    tool_message,
)
from ..tools.articles import lookup_article, resolve_law_id
from ..tools.documents import load_materials, search_materials
from ..tools.merge import merge_retrievals
from ..tools.render import render_tool_result
from ..tools.schemas import (
    GET_ARTICLE_NAME,
    GET_ARTICLE_TOOL,
    MATERIAL_TOP_K_DEFAULT,
    SEARCH_LAW_NAME,
    SEARCH_LAW_TOOL,
    SEARCH_MATERIALS_NAME,
    SEARCH_MATERIALS_TOOL,
    WEB_SEARCH_NAME,
)
from ..tools.web_search import search_web
from .prompts import AGENT_SYSTEM_PROMPT
from .state import AgentState

__all__ = [
    "TOOLS",
    "make_agent_node",
    "make_tools_node",
    "make_finalize_node",
]

TOOLS = [SEARCH_LAW_TOOL, GET_ARTICLE_TOOL, SEARCH_MATERIALS_TOOL]

_NO_FRESH_EVIDENCE = (
    "⚠ 本轮一条新证据都没取到（命中的条上一轮都已经给过你了）。同一个方向换词换不出新东西 ——"
    "要么改问你真正缺的那个要素，要么就此停下作答。\n"
)

_NO_EVIDENCE_STOP_NUDGE = (
    "你这一轮没有调用任何工具，而手上一条证据都还没有 ——「够了」这个判断此刻无凭无据，"
    "循环只会把它读成「模型说够了」直接去作答。请调用 search_law 查一次"
    "（规则 2 的问法：一字不改地用用户的问题原文）。"
)

_NO_FRESH_MATERIAL = (
    "⚠ 这些段上一轮已经给过你了，材料里没有别的相关内容。就此停下作答，或换个说法再试一次。\n"
)


def _sum_usage(first: dict, second: dict) -> dict:
    if not first or not second:
        return first or second or {}
    return {key: (first.get(key) or 0) + (second.get(key) or 0) for key in set(first) | set(second)}


def _unearned_stop(reply: dict, state: AgentState) -> bool:
    return not reply.get("tool_calls") and not state.get("search_log")


def _retrieval_digest(result: RetrievalResult) -> dict:
    return {
        "命中": len(result.articles),
        "向量": result.used_vector,
        "BM25": result.used_bm25,
        "耗时ms": round(result.elapsed_ms, 1),
        "头三条": result.citations()[:3],
    }


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
    parents = rag.parents

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
        seen_materials = {m.label for m in state.get("materials") or ()}
        loaded: list[MaterialPassage] | None = None

        def position() -> int:
            return len(state.get("search_log") or ()) + len(out_logs) + 1

        def web_position() -> int:
            return len(state.get("external") or ()) + len(out_web) + 1

        def materials() -> list[MaterialPassage]:
            nonlocal loaded
            if loaded is None:
                loaded = list(load_materials(state.get("material_ids") or ()))
            return loaded

        def run_search(arguments: str) -> tuple[RetrievalResult | None, str]:
            try:
                query, law_name, top_k = parse_tool_arguments(
                    arguments, default_top_k=default_top_k
                )
            except ValueError as exc:
                return None, f"参数不合法：{exc}。请修正后重试。"

            kwargs: dict[str, Any] = {}
            if law_name:
                law_id, known_names = resolve_law_id(law_name, parents)
                if law_id is None:
                    return None, (
                        f"无法确定法规「{law_name}」。库内只有这几部，请从中选一个"
                        f"（或省略 law_name 检索全部）：{'；'.join(known_names)}"
                    )
                kwargs["law_filter"] = (law_id,)
            elif state.get("region_scope"):
                kwargs["law_filter"] = state["region_scope"]

            try:
                with observer.observation(
                    "检索",
                    "retriever",
                    input={
                        "query": query,
                        "top_k": top_k,
                        "law_filter": kwargs.get("law_filter"),
                    },
                ) as span:
                    result = rag.search(query, top_k=top_k, **kwargs)
                    match_text = rag.expand(query)
                    span.update(output=_retrieval_digest(result))
            except Exception as exc:  # noqa: BLE001
                return None, f"检索失败：{exc}。可以换一组关键词再试，或用更通用的说法。"

            text = render_tool_result(
                result,
                index=position(),
                seen=seen,
                snippet_chars=cfg.snippet_chars,
                match_text=match_text,
            )
            if not {hit.article.parent_id for hit in result.articles} - seen:
                text = _NO_FRESH_EVIDENCE + text
            return result, text

        def run_lookup(arguments: str) -> tuple[RetrievalResult | None, str]:
            try:
                article_no, law_name = parse_article_arguments(arguments)
            except ValueError as exc:
                return None, f"参数不合法：{exc}。请修正后重试。"

            with observer.observation(
                "精确取条",
                "retriever",
                input={"article_no": article_no, "law_name": law_name},
            ) as span:
                result, error = lookup_article(
                    article_no, law_name, parents=parents, index=index
                )
                if result is not None:
                    span.update(output=_retrieval_digest(result))
            if result is None:
                return None, error
            return result, render_tool_result(
                result,
                index=position(),
                seen=seen,
                snippet_chars=cfg.article_chars,
                match_text="",
            )

        def run_web(arguments: str) -> tuple[RetrievalResult | None, str]:
            try:
                query, count = parse_web_arguments(
                    arguments, default_count=config.bocha_config().count
                )
            except ValueError as exc:
                return None, f"参数不合法：{exc}。请修正后重试。"

            with observer.observation(
                "联网检索",
                "retriever",
                input={"query": query, "count": count},
            ) as span:
                findings, text = search_web(query, start=web_position(), count=count)
                span.update(output={"命中": len(findings), "头三条": [w.title for w in findings[:3]]})

            out_web.extend(findings)
            return None, text

        def run_materials(arguments: str) -> tuple[RetrievalResult | None, str]:
            try:
                query, top_k = parse_material_arguments(
                    arguments, default_top_k=MATERIAL_TOP_K_DEFAULT
                )
            except ValueError as exc:
                return None, f"参数不合法：{exc}。请修正后重试。"

            with observer.observation(
                "材料检索", "retriever", input={"query": query, "top_k": top_k}
            ) as span:
                hits, text = search_materials(query, materials(), top_k=top_k)
                span.update(
                    output={
                        "命中": len(hits),
                        "材料份数": len({hit.doc_id for hit in hits}),
                        "段": [hit.citation for hit in hits[:3]],
                    }
                )
            fresh = [hit for hit in hits if hit.label not in seen_materials]
            if hits and not fresh:
                text = _NO_FRESH_MATERIAL + text
            out_materials.extend(fresh)
            seen_materials.update(hit.label for hit in fresh)
            return None, text

        handlers = {
            SEARCH_LAW_NAME: run_search,
            GET_ARTICLE_NAME: run_lookup,
            WEB_SEARCH_NAME: run_web,
            SEARCH_MATERIALS_NAME: run_materials,
        }

        for call in pending:
            call_id = call.get("id", "")
            function = call.get("function") or {}
            name = function.get("name", "")

            handler = handlers.get(name)
            if handler is None:
                out_messages.append(
                    tool_message(
                        call_id,
                        f"未知工具：{name}（本图挂了 {'、'.join(sorted(handlers))}）",
                    )
                )
                continue

            result, text = handler(function.get("arguments") or "")
            out_messages.append(tool_message(call_id, text))
            if result is None:
                continue

            row = result.to_dict()
            out_logs.append(row)
            seen |= {article["parent_id"] for article in row["articles"]}

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
                span.update(output=_retrieval_digest(result))
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
