"""循环本身的三个节点：规划轮、工具执行轮、收尾轮。

循环**外**那两次调用在 `region.py`（入口判地区）与 `review.py`（末端复核）—— 它们只答一道
与循环无关的判断题，各自拿一只便宜的模型、温度钉死 0、**失败即放行**，既不写 `messages`
也不进 `usage`，契约与本模块不同，所以不在这里。

**收尾轮复用既有的 `AnswerGenerator`，提示词一个字都不改** —— 它把 N 次检索合并回一个
`RetrievalResult`，原样交给 `answer()`。所以「Agent 的答案」和「线性管道的答案」是同一段
代码产出的，对照实验比的才真的是架构差异。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .. import config
from ..contracts import ParentChunk, Question, RetrievalResult
from ..obs import Tracer
from ..qa.rag import LegalRAG
from .llm import ToolCallingLLM
from .prompts import AGENT_SYSTEM_PROMPT
from .state import AgentState
from .tools.arguments import parse_article_arguments, parse_tool_arguments, tool_message
from .tools.articles import lookup_article, resolve_law_id
from .tools.merge import merge_retrievals
from .tools.render import render_tool_result
from .tools.schemas import (
    GET_ARTICLE_NAME,
    GET_ARTICLE_TOOL,
    SEARCH_LAW_NAME,
    SEARCH_LAW_TOOL,
)

__all__ = [
    "TOOLS",
    "make_agent_node",
    "make_tools_node",
    "make_finalize_node",
]

TOOLS = [SEARCH_LAW_TOOL, GET_ARTICLE_TOOL]

_NO_FRESH_EVIDENCE = (
    "⚠ 本轮一条新证据都没取到（命中的条上一轮都已经给过你了）。同一个方向换词换不出新东西 ——"
    "要么改问你真正缺的那个要素，要么就此停下作答。\n"
)

_NO_EVIDENCE_STOP_NUDGE = (
    "你这一轮没有调用任何工具，而手上一条证据都还没有 ——「够了」这个判断此刻无凭无据，"
    "循环只会把它读成「模型说够了」直接去作答。请调用 search_law 查一次"
    "（规则 2 的问法：一字不改地用用户的问题原文）。"
)


def _sum_usage(first: dict, second: dict) -> dict:
    """两次调用合成一行 —— `usage` 的行数被 `_trajectory_notes` 当作规划轮数来数。"""
    if not first or not second:
        return first or second or {}
    return {key: (first.get(key) or 0) + (second.get(key) or 0) for key in set(first) | set(second)}


def _unearned_stop(reply: dict, state: AgentState) -> bool:
    """这一次「停」有没有资格：没发工具调用，**且手上一条证据都没有**。

    判「够了」要有证据可判。零证据又停，只有一个解释：模型把「缺什么」写成了正文，
    没有把缺的东西写成那一轮的工具调用。有证据的停（查过一轮之后自己收手）是合法的，
    不在此列 —— 那是这个循环唯一的停止信号。
    """
    return not reply.get("tool_calls") and not state.get("search_log")


def _retrieval_digest(result: RetrievalResult) -> dict:
    """检索观察的输出：命中几条、走了哪条通道、多快、头三条是谁。

    只放**判断用得上的**：命中条数与首选决定「这一轮有没有用」，通道与耗时解释
    「为什么慢/为什么空」。全文不在这里 —— 它已经在节点 span 的 search_log 里了。
    """
    return {
        "命中": len(result.articles),
        "向量": result.used_vector,
        "BM25": result.used_bm25,
        "耗时ms": round(result.elapsed_ms, 1),
        "头三条": result.citations()[:3],
    }


def make_agent_node(llm: ToolCallingLLM, cfg: config.AgentConfig, laws: list[str]):
    """规划轮。**这个节点同时管判断与检索**：它说「够了」的方式就是不调用工具。

    读：question / history / messages / max_steps
    写：{"messages": [助手消息], "steps": 1, "usage": [一行]}

    `laws` 是**库内全部法规名**，写进提示词当「库的边界」。没有它，模型分不清「这一轮没
    检出来」和「库里根本没有」，会把预算全烧在补不上的缺口上。名单纯从 `parents` 现取，
    不写死一份：新增法规时不该有人记得来这里改。

    **预算不在这个节点里判**：进得来就说明还有轮次（由 `_route_after_tools` 保证），
    所以工具照给 —— 模型不必靠「工具被收走」来知道该停。

    **零证据的「停」不成立，会重问一次。** 判「够了」得有证据可判；手上一条都没有时那一轮
    助手消息却不带 `tool_calls`，只有一个解释 —— 模型把「缺什么」写成了正文，而没把缺的东西
    写成那一轮的工具调用（2026-09-22 实测：4 次实跑里 2 次这样，那一轮零检索，6 条证据全是
    收尾节点兜底捞的）。第一次停若手上没证据，就追加一句纠正再问一次，**上限一次**：
    第二次仍不调工具就照旧收尾，走兜底检索那条路，不会在这里转圈。
    """

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
    """执行工具调用。

    读：messages[-1].tool_calls / search_log / top_k / region_scope
    写：{"messages": [每个 tool_call 一条 ToolMessage], "search_log": [新增行]}

    检索的作用域有**两个来源，模型点名的优先**：模型给了 `law_name` 就用它解析出的那一部，
    没给才落到入口判出的 `region_scope`（该地区条例 + 国家法，判不出时为空 = 全库）。
    顺序不能反：模型点名是它在看到上一轮证据后的明确取舍，地区限定只是没人点名时的默认值。

    **不变量：每个 tool_call 恰好产出一条 ToolMessage** —— 无论成功、参数非法、
    工具名未知、还是工具抛异常。闭合发生在产生 tool_call 的那个节点里，
    于是「assistant 的 tool_calls 没有对应的 tool 消息」这个会让端点直接 400 的
    状态，在结构上就不可能出现，不需要任何事后修补。

    `observer` 只用来给每次检索记一笔（观察类型 `retriever`），与检索结果无关；
    不传就是空实现，逐位不变。

    **一条新证据都没取到的那一轮，工具消息开头会加一句警告**（`_NO_FRESH_EVIDENCE`）。这是
    「这轮白查了」唯一可信的信号：拿检索词的字面相似度去判重复实测分不开 —— 同一句话删掉一个
    字的重复对是 0.983，而「罚多少 / 记多少分」这种**合法续查**是 0.927，中间没有安全的阈值。
    只有「返回的条一条都不新」是硬事实，而且它不拦检索、不丢证据，只是把话说破。
    """

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

        def position() -> int:
            """工具结果里的序号，跨轮连续。"""
            return len(state.get("search_log") or ()) + len(out_logs) + 1

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
            except Exception as exc:  # noqa: BLE001 - 工具失败不该打断循环
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

        handlers = {SEARCH_LAW_NAME: run_search, GET_ARTICLE_NAME: run_lookup}

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

        return {"messages": out_messages, "search_log": out_logs}

    return tools_node


def _trajectory_notes(state: AgentState, merged: RetrievalResult, cfg: config.AgentConfig) -> list[str]:
    """把循环本身的成本写进 Answer.notes —— 不摆出来，就没法解释「凭什么是 3 倍」。

    **两处计数刻意来自两个不同的源**（`usage` / `tool_call` 名字）：`search_log` 的行数
    不是检索次数（`get_article` 同形状但零检索），而 `steps` 与 `usage` 的行数今天虽然
    相等，口径却不是一回事 —— 一个是路由器累加的预算，一个是调用处写下的账单。

    收尾成因只写一种：**预算用尽被强制作答**。判据是「最后一条消息不是助手消息」——
    助手消息且没有 `tool_calls` 就是模型自己收的手（它说了「够了」），而预算尽那条路上
    最后一条一定是 `tools` 产出的 tool 消息。
    """
    calls = [
        (call.get("function") or {}).get("name") or ""
        for message in state.get("messages") or ()
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or ()
    ]
    plans = len(state.get("usage") or ())
    searches = calls.count(SEARCH_LAW_NAME)
    lookups = calls.count(GET_ARTICLE_NAME)
    steps = state.get("steps", 0)
    max_steps = state.get("max_steps", cfg.max_steps)

    parts = [f"{plans} 轮 LLM 规划", f"{searches} 次检索"]
    if lookups:
        parts.append(f"{lookups} 次精确取条")
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
    """收尾：把累积的证据合并回一个 RetrievalResult，交给既有生成器。

    读：question / history / messages / search_log / steps / max_steps / usage / top_k
    写：{"answer": Answer, "search_log": [兜底检索那一行，否则 []]}

    **这里不喂工具历史、也不 bind_tools** —— 从零重建一次「问题 + 依据」的提示词。
    这样既绕开了可能残留的悬空 tool_calls，又让提示词与线性管道逐字节相同。

    **不接 `top_k` 参数**：那曾是 runner 的条数，只在 `state` 缺字段时兜底，
    而 `AgentRunner.invoke()` 必然写 `state["top_k"]` —— 它一次都没生效过，
    却让同一个函数里坐着两个 `top_k`（收尾读闭包、工具轮读 state），
    调用方传进来的 `Question` 自带 `top_k` 时两者就分叉。现在兜底直接读 config。

    `observer` 同上：只给那条**兜底检索**记一笔。它不进工具轮，是这张图里唯一一次
    「没人要求、自己做的检索」，看 trace 时正要知道它花了多久、命中了什么。
    """

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
        answer = rag.answer(question, merged)
        return {"answer": replace(answer, notes=answer.notes + tuple(notes)), "search_log": extra}

    return finalize_node
