from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal
from uuid import uuid4

from api_contracts import RagClient
from rag_contracts import config
from rag_contracts.domain.answer import Answer, Question
from rag_contracts.domain.laws import LawInfo
from rag_contracts.observability.tracer import Tracer, traced
from rag_contracts.ports import LLM

from ..api import budget
from .clarify import make_clarify_node
from .errors import ResumeConflict, SessionUnsupported
from .nodes import TOOLS, make_agent_node, make_finalize_node, make_tools_node
from .region import REGION_UNKNOWN, make_region_node
from .review import make_review_node
from .state import AgentState

__all__ = ["route_after_agent", "build_graph", "make_run_config", "AgentRunner"]


def route_after_agent(state: AgentState) -> Literal["tools", "finalize"]:
    messages = state.get("messages") or []
    return "tools" if messages and messages[-1].get("tool_calls") else "finalize"


def _route_after_tools(state: AgentState) -> Literal["agent", "finalize"]:
    return "agent" if state.get("max_steps", 0) - state.get("steps", 0) > 0 else "finalize"


def make_run_config(session_id: str | None, *, sessions) -> dict:
    if sessions is None:
        if session_id is not None:
            raise SessionUnsupported(
                "这个 runner 没接会话存储（sessions=None），不能带 session_id"
            )
        return {"config": {}}
    return {
        "config": {"configurable": {"thread_id": session_id or uuid4().hex}},
        "durability": "sync",
    }


def _interrupt_payload(value: object) -> dict | None:
    for item in value or ():
        return {"id": getattr(item, "id", None), "value": getattr(item, "value", None)}
    return None


def _step_detail(node: str, update: object) -> dict:
    if not isinstance(update, dict):
        return {}
    if node == "region":
        return {"region": update.get("region"), "laws": len(update.get("region_scope") or ())}
    if node == "clarify":
        return {"region": update.get("region"), "laws": len(update.get("region_scope") or ())}
    if node == "agent":
        names = [
            (call.get("function") or {}).get("name") or ""
            for message in update.get("messages") or ()
            for call in (message.get("tool_calls") or ())
        ]
        return {"tools": names, "truncated": bool(update.get("truncated"))}
    if node == "tools":
        rows = update.get("search_log") or []
        return {
            "retrievals": len(rows),
            "hits": sum(len(row.get("articles") or ()) for row in rows),
            "materials": len(update.get("materials") or ()),
            "web": len(update.get("external") or ()),
        }
    if node == "finalize":
        answer = update.get("answer")
        return {
            "evidence": len(answer.evidences) if answer is not None else 0,
            "fallback": bool(update.get("search_log")),
        }
    if node == "review":
        review = getattr(update.get("answer"), "review", None)
        return {} if review is None else {"passed": review.passed, "score": review.score}
    return {}


def build_graph(
    *,
    rag: RagClient,
    llm: LLM,
    cfg: config.AgentConfig,
    region_llm: LLM | None = None,
    review_llm: LLM | None = None,
    tracer: Tracer | None = None,
    checkpointer=None,
    laws: Sequence[LawInfo] | None = None,
):
    from langgraph.graph import END, START, StateGraph

    tracer = tracer or Tracer()

    region_llm = region_llm or llm
    review_llm = review_llm or region_llm

    laws = list(laws) if laws is not None else list(rag.laws())

    graph = StateGraph(AgentState)
    graph.add_node(
        "region",
        traced(
            tracer,
            "node.region",
            make_region_node(region_llm, laws, history_turns=cfg.history_turns),
        ),
    )
    graph.add_node("clarify", traced(tracer, "node.clarify", make_clarify_node(laws)))
    graph.add_node("agent", traced(tracer, "node.agent", make_agent_node(llm, cfg, laws)))
    graph.add_node("tools", traced(tracer, "node.tools", make_tools_node(rag, cfg, observer=tracer)))
    graph.add_node(
        "finalize", traced(tracer, "node.finalize", make_finalize_node(rag, cfg, observer=tracer))
    )
    graph.add_node("review", traced(tracer, "node.review", make_review_node(review_llm, cfg)))

    graph.add_edge(START, "region")
    graph.add_conditional_edges(
        "region",
        lambda state: "clarify" if cfg.clarify and state.get("place") else "agent",
        {"clarify": "clarify", "agent": "agent"},
    )
    graph.add_edge("clarify", "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "tools", _route_after_tools, {"agent": "agent", "finalize": "finalize"}
    )
    graph.add_edge("finalize", "review")
    graph.add_edge("review", END)
    return graph.compile(checkpointer=checkpointer)


class AgentRunner:

    input_desc = "问题 (str) / Question"
    output_desc = "Answer"

    def __init__(
        self,
        rag: RagClient,
        *,
        llm: LLM | None = None,
        region_llm: LLM | None = None,
        review_llm: LLM | None = None,
        cfg: config.AgentConfig | None = None,
        tracer: Tracer | None = None,
        sessions=None,
        laws: Sequence[LawInfo] | None = None,
    ) -> None:
        from rag_contracts.llm import build_llm

        self.rag = rag
        self.cfg = cfg or config.agent_config()
        self.tracer = tracer or Tracer()
        self.llm = llm or build_llm(retries=self.cfg.retries, observer=self.tracer)
        self.region_llm = region_llm or self.llm
        self.review_llm = review_llm or self.region_llm
        self.sessions = sessions
        self._laws = None if laws is None else list(laws)
        self._graph = None

    @property
    def top_k(self) -> int:
        return self.rag.top_k

    @classmethod
    def attach(
        cls,
        rag: RagClient,
        *,
        cfg: config.AgentConfig | None = None,
        tracer: Tracer | None = None,
        sessions=None,
        laws: Sequence[LawInfo] | None = None,
    ) -> "AgentRunner":
        from rag_contracts.llm import build_llm

        agent_cfg = cfg or config.agent_config()
        if tracer is None:
            from rag_contracts.observability.langfuse import from_env

            tracer = from_env()
        observer = tracer or Tracer()
        return cls(
            rag,
            cfg=agent_cfg,
            tracer=observer,
            region_llm=build_llm(
                config.region_llm_config(), retries=agent_cfg.retries, observer=observer
            ),
            review_llm=build_llm(
                config.review_llm_config(), retries=agent_cfg.retries, observer=observer
            ),
            sessions=sessions,
            laws=laws,
        )

    def graph(self):
        if self._graph is None:
            self._graph = build_graph(
                rag=self.rag,
                llm=self.llm,
                cfg=self.cfg,
                region_llm=self.region_llm,
                review_llm=self.review_llm,
                tracer=self.tracer,
                checkpointer=self.sessions,
                laws=self._laws,
            )
        return self._graph

    def _initial(
        self,
        query: Question,
        material_ids: Sequence[str],
        *,
        session_id: str = "",
        stream_tokens: bool = False,
    ) -> AgentState:
        return {
            "question": query.text,
            "history": [tuple(pair) for pair in query.history],
            "session_id": session_id,
            "stream_tokens": stream_tokens,
            "top_k": query.top_k or self.top_k,
            "max_steps": self.cfg.max_steps,
            "region": REGION_UNKNOWN,
            "region_scope": (),
            "place": "",
            "messages": [],
            "search_log": [],
            "external": [],
            "materials": [],
            "material_ids": list(material_ids),
            "steps": 0,
            "usage": [],
            "usage_extra": [],
            "truncated": 0,
            "conversation": [],
        }

    def _config_kwargs(self, session_id: str | None) -> tuple[dict, str | None]:
        kwargs = make_run_config(session_id, sessions=self.sessions)
        config_dict = dict(kwargs["config"])
        config_dict["recursion_limit"] = 2 * self.cfg.max_steps + 4
        kwargs["config"] = config_dict
        return kwargs, (config_dict.get("configurable") or {}).get("thread_id")

    def _latest_state(self, kwargs: dict) -> AgentState | None:
        try:
            snapshot = self.graph().get_state(kwargs["config"])
        except Exception:  # noqa: BLE001
            return None
        return getattr(snapshot, "values", None)

    def _run(self, run_input, kwargs: dict, *, request=None) -> AgentState:
        final: AgentState | None = None
        try:
            with self.tracer.span("invoke", root=True) as span:
                final = self.graph().invoke(run_input, **kwargs)
                span.record(request if request is not None else run_input, final)
        finally:
            budget.record_state(final if final is not None else self._latest_state(kwargs))
        return final

    def invoke(
        self,
        question: str | Question,
        *,
        material_ids: Sequence[str] = (),
        session_id: str | None = None,
    ) -> AgentState:
        query = question if isinstance(question, Question) else Question(text=question)
        kwargs, sid = self._config_kwargs(session_id)
        initial = self._initial(query, material_ids, session_id=sid or "", stream_tokens=False)
        return self._run(initial, kwargs)

    def _stream_run(
        self, run_input, kwargs: dict, sid: str | None, *, request=None
    ) -> Iterator[tuple[str, dict]]:
        final: AgentState = run_input if isinstance(run_input, dict) else {}
        turn = 0
        paused = False
        with self.tracer.span("invoke", root=True) as span:
            try:
                for mode, chunk in self.graph().stream(
                    run_input, **kwargs, stream_mode=["updates", "values", "custom"]
                ):
                    if mode == "values":
                        final = chunk
                        continue
                    if mode == "custom":
                        yield "delta", chunk
                        continue
                    if "__interrupt__" in chunk:
                        paused = True
                        yield "interrupt", {
                            "session_id": sid,
                            "interrupt": _interrupt_payload(chunk.get("__interrupt__")),
                        }
                        continue
                    for node, update in chunk.items():
                        detail = _step_detail(node, update)
                        if node == "agent":
                            turn += 1
                            detail = {"turn": turn, "max_steps": self.cfg.max_steps, **detail}
                        yield "step", {"node": node, **detail}
            finally:
                budget.record_state(final)
            span.record(request if request is not None else run_input, final)
        if paused:
            return
        answer = final.get("answer")
        if answer is None:
            raise RuntimeError("Agent 未产出答案：图执行异常结束")
        yield "answer", {**answer.to_dict(), "session_id": sid}

    def stream(
        self,
        question: str | Question,
        *,
        material_ids: Sequence[str] = (),
        session_id: str | None = None,
    ) -> Iterator[tuple[str, dict]]:
        query = question if isinstance(question, Question) else Question(text=question)
        kwargs, sid = self._config_kwargs(session_id)
        if not self.llm.available:
            yield "answer", {**self.rag.ask(query).to_dict(), "session_id": sid}
            return
        initial = self._initial(query, material_ids, session_id=sid or "", stream_tokens=True)
        yield from self._stream_run(initial, kwargs, sid)

    def ask(
        self,
        question: str | Question,
        *,
        material_ids: Sequence[str] = (),
        session_id: str | None = None,
    ) -> Answer:
        query = question if isinstance(question, Question) else Question(text=question)
        kwargs, sid = self._config_kwargs(session_id)
        if not self.llm.available:
            return self.rag.ask(query)
        final = self._run(
            self._initial(query, material_ids, session_id=sid or "", stream_tokens=False), kwargs
        )
        answer = final.get("answer")
        if answer is None:
            raise RuntimeError("Agent 未产出答案：图执行异常结束")
        return answer

    def ask_payload(
        self,
        question: str | Question,
        *,
        material_ids: Sequence[str] = (),
        session_id: str | None = None,
    ) -> tuple[Literal["ok", "interrupted"], dict]:
        query = question if isinstance(question, Question) else Question(text=question)
        kwargs, sid = self._config_kwargs(session_id)
        if not self.llm.available:
            return "ok", {**self.rag.ask(query).to_dict(), "session_id": sid}
        final = self._run(
            self._initial(query, material_ids, session_id=sid or "", stream_tokens=False), kwargs
        )
        return self._outcome(final, sid)

    @staticmethod
    def _outcome(
        final: AgentState, sid: str | None
    ) -> tuple[Literal["ok", "interrupted"], dict]:
        interrupt = _interrupt_payload(final.get("__interrupt__"))
        if interrupt is not None:
            return "interrupted", {"session_id": sid, "interrupt": interrupt}
        answer = final.get("answer")
        if answer is None:
            raise RuntimeError("Agent 未产出答案：图执行异常结束")
        return "ok", {**answer.to_dict(), "session_id": sid}

    def _resume_input(
        self, session_id: str | None, value: dict, *, stream_tokens: bool
    ) -> tuple[object, dict, str | None]:
        from langgraph.types import Command

        if not session_id:
            raise ResumeConflict("恢复要带上 session_id：这条会话没有别的句柄")
        kwargs, sid = self._config_kwargs(session_id)
        snapshot = self.graph().get_state(kwargs["config"])
        pending = _interrupt_payload(
            [item for task in snapshot.tasks for item in task.interrupts]
        )
        if pending is None or str((pending.get("value") or {}).get("type")) != "region_clarify":
            raise ResumeConflict("这条会话没有等待澄清的地区问题（可能已经答完）")
        return Command(resume=value, update={"stream_tokens": stream_tokens}), kwargs, sid

    def resume(
        self, session_id: str, value: dict
    ) -> tuple[Literal["ok", "interrupted"], dict]:
        command, kwargs, sid = self._resume_input(session_id, value, stream_tokens=False)
        final = self._run(command, kwargs, request={"resume": value, "session_id": sid})
        return self._outcome(final, sid)

    def resume_stream(self, session_id: str, value: dict) -> Iterator[tuple[str, dict]]:
        command, kwargs, sid = self._resume_input(session_id, value, stream_tokens=True)
        yield from self._stream_run(
            command, kwargs, sid, request={"resume": value, "session_id": sid}
        )

    def describe(self) -> str:
        threshold = self.cfg.review_min_score
        review_state = (
            "复核已关闭"
            if threshold < 0
            else (
                f"末端复核（1 次 {self.review_llm.model_name} 调用，"
                + (f"支撑 < {threshold:g} 降级转人工）" if threshold > 0 else "只打分不拦截）")
            )
        )
        return (
            f"Agent：最多 {self.cfg.max_steps} 轮 | "
            f"入口判地区（1 次 {self.region_llm.model_name} 调用，判不出则不限地区） | "
            f"{review_state} | "
            f"工具 {len(TOOLS)} 个（{'、'.join(tool['function']['name'] for tool in TOOLS)}） | "
            f"会话记忆：{'开（thread_id=session_id）' if self.sessions else '关'} | "
            f"地区澄清：{'开（问题沾到地方先回问一次）' if self.cfg.clarify else '关'} | "
            f"LLM {self.llm.model_name}"
        )
