from __future__ import annotations

from typing import Literal, Sequence

from api_contracts import RagClient
from rag_contracts import config
from rag_contracts.domain.answer import Answer, Question
from rag_contracts.observability.tracer import Tracer, traced
from rag_contracts.ports import LLM

from .nodes import TOOLS, make_agent_node, make_finalize_node, make_tools_node
from .region import REGION_UNKNOWN, make_region_node
from .review import make_review_node
from .state import AgentState

__all__ = ["route_after_agent", "build_graph", "AgentRunner"]


def route_after_agent(state: AgentState) -> Literal["tools", "finalize"]:
    messages = state.get("messages") or []
    return "tools" if messages and messages[-1].get("tool_calls") else "finalize"


def _route_after_tools(state: AgentState) -> Literal["agent", "finalize"]:
    return "agent" if state.get("max_steps", 0) - state.get("steps", 0) > 0 else "finalize"


def _step_detail(node: str, update: object) -> dict:
    if not isinstance(update, dict):
        return {}
    if node == "region":
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
):
    from langgraph.graph import END, START, StateGraph

    tracer = tracer or Tracer()

    region_llm = region_llm or llm
    review_llm = review_llm or region_llm

    laws = list(rag.laws())

    graph = StateGraph(AgentState)
    graph.add_node("region", traced(tracer, "node.region", make_region_node(region_llm, laws)))
    graph.add_node("agent", traced(tracer, "node.agent", make_agent_node(llm, cfg, laws)))
    graph.add_node("tools", traced(tracer, "node.tools", make_tools_node(rag, cfg, observer=tracer)))
    graph.add_node(
        "finalize", traced(tracer, "node.finalize", make_finalize_node(rag, cfg, observer=tracer))
    )
    graph.add_node("review", traced(tracer, "node.review", make_review_node(review_llm, cfg)))

    graph.add_edge(START, "region")
    graph.add_edge("region", "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "tools", _route_after_tools, {"agent": "agent", "finalize": "finalize"}
    )
    graph.add_edge("finalize", "review")
    graph.add_edge("review", END)
    return graph.compile()


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
    ) -> None:
        from rag_contracts.llm import build_llm

        self.rag = rag
        self.cfg = cfg or config.agent_config()
        self.tracer = tracer or Tracer()
        self.llm = llm or build_llm(retries=self.cfg.retries, observer=self.tracer)
        self.region_llm = region_llm or self.llm
        self.review_llm = review_llm or self.region_llm
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
            )
        return self._graph

    def _initial(self, query: Question, material_ids: Sequence[str]) -> AgentState:
        return {
            "question": query.text,
            "history": [tuple(pair) for pair in query.history],
            "top_k": query.top_k or self.top_k,
            "max_steps": self.cfg.max_steps,
            "region": REGION_UNKNOWN,
            "region_scope": (),
            "messages": [],
            "search_log": [],
            "external": [],
            "materials": [],
            "material_ids": list(material_ids),
            "steps": 0,
            "usage": [],
            "truncated": 0,
        }

    def invoke(
        self, question: str | Question, *, material_ids: Sequence[str] = ()
    ) -> AgentState:
        query = question if isinstance(question, Question) else Question(text=question)
        initial = self._initial(query, material_ids)
        with self.tracer.span("invoke", root=True) as span:
            final = self.graph().invoke(
                initial, config={"recursion_limit": 2 * self.cfg.max_steps + 4}
            )
            span.record(initial, final)
        return final

    def stream(self, question: str | Question, *, material_ids: Sequence[str] = ()):
        query = question if isinstance(question, Question) else Question(text=question)
        if not self.llm.available:
            yield "answer", self.rag.ask(query).to_dict()
            return

        initial = self._initial(query, material_ids)
        final: AgentState = initial
        turn = 0
        with self.tracer.span("invoke", root=True) as span:
            for mode, chunk in self.graph().stream(
                initial,
                config={"recursion_limit": 2 * self.cfg.max_steps + 4},
                stream_mode=["updates", "values"],
            ):
                if mode == "values":
                    final = chunk
                    continue
                for node, update in chunk.items():
                    detail = _step_detail(node, update)
                    if node == "agent":
                        turn += 1
                        detail = {"turn": turn, "max_steps": self.cfg.max_steps, **detail}
                    yield "step", {"node": node, **detail}
            span.record(initial, final)

        answer = final.get("answer")
        if answer is None:
            raise RuntimeError("Agent 未产出答案：图执行异常结束")
        yield "answer", answer.to_dict()

    def ask(self, question: str | Question, *, material_ids: Sequence[str] = ()) -> Answer:
        query = question if isinstance(question, Question) else Question(text=question)
        if not self.llm.available:
            return self.rag.ask(query)

        state = self.invoke(query, material_ids=material_ids)
        answer = state.get("answer")
        if answer is None:
            raise RuntimeError("Agent 未产出答案：图执行异常结束")
        return answer

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
            f"LLM {self.llm.model_name}"
        )
