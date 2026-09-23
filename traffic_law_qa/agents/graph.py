from __future__ import annotations

from typing import Literal, Sequence

from .. import config
from ..contracts import Answer, Question
from ..obs import Tracer, traced
from ..qa.rag import LegalRAG
from ..ready import ensure_ready
from ..services.llm import ToolCallingLLM
from ..tools.articles import build_article_index
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


def build_graph(
    *,
    rag: LegalRAG,
    llm: ToolCallingLLM,
    cfg: config.AgentConfig,
    region_llm: ToolCallingLLM | None = None,
    review_llm: ToolCallingLLM | None = None,
    tracer: Tracer | None = None,
):
    from langgraph.graph import END, START, StateGraph

    tracer = tracer or Tracer()

    region_llm = region_llm or llm
    review_llm = review_llm or region_llm

    parents = rag.parents
    by_number = build_article_index(parents)
    laws = sorted({parent.law_name for parent in parents.values()})

    graph = StateGraph(AgentState)
    graph.add_node("region", traced(tracer, "node.region", make_region_node(region_llm, parents, laws)))
    graph.add_node("agent", traced(tracer, "node.agent", make_agent_node(llm, cfg, laws)))
    graph.add_node(
        "tools", traced(tracer, "node.tools", make_tools_node(rag, cfg, by_number, observer=tracer))
    )
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
        rag: LegalRAG,
        *,
        llm: ToolCallingLLM | None = None,
        region_llm: ToolCallingLLM | None = None,
        review_llm: ToolCallingLLM | None = None,
        cfg: config.AgentConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.rag = rag
        self.cfg = cfg or config.agent_config()
        self.tracer = tracer or Tracer()
        self.llm = llm or ToolCallingLLM(retries=self.cfg.retries, observer=self.tracer)
        self.region_llm = region_llm or self.llm
        self.review_llm = review_llm or self.region_llm
        self._graph = None

    @property
    def top_k(self) -> int:
        return self.rag.top_k

    @classmethod
    def load(
        cls,
        *,
        with_vector: bool = True,
        cfg: config.AgentConfig | None = None,
        top_k: int | None = None,
        tracer: Tracer | None = None,
    ) -> "AgentRunner":
        ensure_ready(with_vector=with_vector)
        return cls.attach(
            LegalRAG.load(with_vector=with_vector, top_k=top_k), cfg=cfg, tracer=tracer
        )

    @classmethod
    def attach(
        cls,
        rag: LegalRAG,
        *,
        cfg: config.AgentConfig | None = None,
        tracer: Tracer | None = None,
    ) -> "AgentRunner":
        agent_cfg = cfg or config.agent_config()
        if tracer is None:
            from .langfuse_tracer import from_env

            tracer = from_env()
        observer = tracer or Tracer()
        return cls(
            rag,
            cfg=agent_cfg,
            tracer=observer,
            region_llm=ToolCallingLLM(
                config.region_llm_config(), retries=agent_cfg.retries, observer=observer
            ),
            review_llm=ToolCallingLLM(
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

    def invoke(
        self, question: str | Question, *, material_ids: Sequence[str] = ()
    ) -> AgentState:
        query = question if isinstance(question, Question) else Question(text=question)
        initial: AgentState = {
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
        }
        with self.tracer.span("invoke", root=True) as span:
            final = self.graph().invoke(
                initial, config={"recursion_limit": 2 * self.cfg.max_steps + 4}
            )
            span.record(initial, final)
        return final

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
        stats = self.rag.stats()
        if stats.dense is None:
            vector_state = "未知（Milvus 未连接）"
        else:
            vector_state = "已启用" if stats.dense else "未启用（仅 BM25）"
        threshold = self.cfg.review_min_score
        review_state = (
            "复核已关闭"
            if threshold < 0
            else (
                f"末端复核（1 次 {self.review_llm.cfg.model} 调用，"
                + (f"支撑 < {threshold:g} 降级转人工）" if threshold > 0 else "只打分不拦截）")
            )
        )
        return (
            f"Agent：{stats.articles} 条法条 / {stats.chunks} 个子块 | "
            f"稠密通道 {vector_state} | 最多 {self.cfg.max_steps} 轮 | "
            f"入口判地区（1 次 {self.region_llm.cfg.model} 调用，判不出则不限地区） | "
            f"{review_state} | "
            f"工具 {len(TOOLS)} 个（{'、'.join(tool['function']['name'] for tool in TOOLS)}） | "
            f"LLM {self.llm.cfg.model}"
        )
