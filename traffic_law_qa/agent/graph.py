"""图的装配（4 个节点 + 2 个路由）与门面 `AgentRunner`。

    START → region → agent ─┬─(有 tool_calls)→ tools ─(还有预算)→ 回 agent
                            │                 └─(预算尽)──────┐
                            └─(无 tool_calls)─────────────────┴→ finalize → END

**停止由规划轮自己说**：这一轮的助手消息没有 `tool_calls` 就是「证据够了」，直接进作答 ——
标准 ReAct 的形状。代价是「停」要交给一只手上正握着工具的模型（它天然有调用倾向），
提示词里那段停止条件因此写得比别的几段都重。

**回边指向 agent 而不是 region**：地区只在入口判一次。第二轮再判一遍纯属浪费一次调用，
而且可能判出不同地区导致检索作用域抖动 —— 同一道题的证据范围在运行中来回变，就没法解释了。

`AgentRunner` 与 `LegalRAG` 平级，是**第二条编排路径**而非替代品：`qa(mode="ask")` 仍是
默认，走线性管道；Agent 需显式开启。两者产出同一个 `Answer` 契约对象，因此可以直接对照。

**只用 LangGraph 的状态机，不用它的 LLM 抽象层。** 带工具的调用直接走 `openai` SDK，
于是 state 里的 `messages` 就是 OpenAI 线上格式的 `list[dict]`。
"""

from __future__ import annotations

from typing import Literal

from .. import config
from ..contracts import Answer, Question
from ..obs import Tracer, traced
from ..qa.rag import LegalRAG
from .llm import ToolCallingLLM
from .nodes import TOOLS, make_agent_node, make_finalize_node, make_tools_node
from .region import REGION_UNKNOWN, make_region_node
from .state import AgentState
from .tools import build_article_index

__all__ = ["route_after_agent", "build_graph", "AgentRunner"]


def route_after_agent(state: AgentState) -> Literal["tools", "finalize"]:
    """最后一条消息带 tool_calls → 必须先去 tools；不带 → 模型自己说够了，进作答。

    这是全图的停止信号：**「够了」不是一句判断，而是「这一轮没调工具」这个动作本身**。

    绝不能在这里看预算：一旦在预算用尽时直接跳到 finalize，历史里就会留下一条
    带 tool_calls 却没有对应 tool 消息的助手消息，OpenAI 兼容端点会直接返回 400。
    预算的守门在 `_route_after_tools` —— 那里最后一条一定是 tool 消息。
    """
    messages = state.get("messages") or []
    return "tools" if messages and messages[-1].get("tool_calls") else "finalize"


def _route_after_tools(state: AgentState) -> Literal["agent", "finalize"]:
    """工具跑完：还有预算就回规划轮再查一轮，预算用尽就直接收尾作答。

    **这是全图唯一检查预算的地方**，位置由 `route_after_agent` 那条约束反推出来：
    只有在这里跳 finalize 是安全的，因为最后一条消息一定是 `tools` 产出的 tool 消息，
    不存在悬空的 tool_calls。

    预算尽时**不再调一次模型**（从前由审核节点的短路承担同一件事）：那一轮反正要作答，
    问了也是白花一次调用。判据里的 `steps` 是已发生的规划轮数，由 `agent` 节点累加；
    「递减」的字段不落进 state —— 在 `operator.add` reducer 下是读-改-写，写错就是双倍
    消耗，而且路由器是纯函数、本来就没有写权限。
    """
    return "agent" if state.get("max_steps", 0) - state.get("steps", 0) > 0 else "finalize"


def build_graph(
    *,
    rag: LegalRAG,
    llm: ToolCallingLLM,
    cfg: config.AgentConfig,
    region_llm: ToolCallingLLM | None = None,
    tracer: Tracer | None = None,
):
    """装配图。`region_llm` 是**入口判地区那一只**模型 —— 要的是判得准且便宜；
    循环用主模型，它要的是会调工具、也会自己判断什么时候停。"""
    from langgraph.graph import END, START, StateGraph

    tracer = tracer or Tracer()

    region_llm = region_llm or llm

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

    graph.add_edge(START, "region")
    graph.add_edge("region", "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "tools", _route_after_tools, {"agent": "agent", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    return graph.compile()


class AgentRunner:
    """Input : 自然语言问题 (str) / Question；Output: Answer（与单轮管道同一个契约对象）。

    **依赖的是 `LegalRAG`，不是检索器**：生成器与 top_k 都从 rag 上取，不另存一份 ——
    存两份就迟早不一致（`--linear` 退回线性管道时要用的正是同一个 rag）。
    组图这一侧照同一条走：`build_graph` 与各节点工厂都不接 `top_k` 参数，
    条数只从 `state["top_k"]` 取、兜底读 `config`，真源始终只有一处。
    """

    input_desc = "问题 (str) / Question"
    output_desc = "Answer"

    def __init__(
        self,
        rag: LegalRAG,
        *,
        llm: ToolCallingLLM | None = None,
        region_llm: ToolCallingLLM | None = None,
        cfg: config.AgentConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.rag = rag
        self.cfg = cfg or config.agent_config()
        self.tracer = tracer or Tracer()
        self.llm = llm or ToolCallingLLM(retries=self.cfg.retries, observer=self.tracer)
        self.region_llm = region_llm or self.llm
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
        agent_cfg = cfg or config.agent_config()
        observer = tracer or Tracer()
        return cls(
            LegalRAG.load(with_vector=with_vector, top_k=top_k),
            cfg=agent_cfg,
            tracer=observer,
            region_llm=ToolCallingLLM(
                config.region_llm_config(), retries=agent_cfg.retries, observer=observer
            ),
        )

    def graph(self):
        """懒构建 + 缓存：一个 rag 只配一套图。"""
        if self._graph is None:
            self._graph = build_graph(
                rag=self.rag,
                llm=self.llm,
                cfg=self.cfg,
                region_llm=self.region_llm,
                tracer=self.tracer,
            )
        return self._graph

    def invoke(self, question: str | Question) -> AgentState:
        """跑完整图，返回**原始终态**（调试 / 测试 / --json 用）。

        外层 `invoke` span 与内层四个 `node.*` 是平的、不嵌套计时 —— 两者相减就是
        langgraph 自己的调度开销，那恰好是「20 分钟里有多少是框架的」的答案。

        `record(initial, final)` 是给真后端用的：根 span 的入参/出参就是整条 trace 的
        入参/出参（问题进、答案出）。空实现下它什么也不做。
        """
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
            "steps": 0,
            "usage": [],
        }
        with self.tracer.span("invoke") as span:
            final = self.graph().invoke(
                initial, config={"recursion_limit": 2 * self.cfg.max_steps + 4}
            )
            span.record(initial, final)
        return final

    def ask(self, question: str | Question) -> Answer:
        """提问 → 带 [依据N] 标注的答案。未配 LLM 时退回线性管道。"""
        query = question if isinstance(question, Question) else Question(text=question)
        if not self.llm.available:
            return self.rag.ask(query)

        state = self.invoke(query)
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
        return (
            f"Agent：{stats.articles} 条法条 / {stats.chunks} 个子块 | "
            f"稠密通道 {vector_state} | 最多 {self.cfg.max_steps} 轮 | "
            f"入口判地区（1 次 {self.region_llm.cfg.model} 调用，判不出则不限地区） | "
            f"工具 2 个（{'、'.join(tool['function']['name'] for tool in TOOLS)}） | "
            f"LLM {self.llm.cfg.model}"
        )
