"""图的装配（6 个节点 + 4 个路由）与门面 `AgentRunner`。

    START → classify ─(条文定位)→ lookup_plan ─┐
                  └──(法规检索)→ agent ────────┤─(有 tool_calls)→ tools → reflect
                                              │                          │
                                  (无 tool_calls)┘      (不够 且 有预算)──┘
                                              └──→ finalize ←──(够了 / 预算尽)
                                                     │
                                                    END

**回边指向 agent 而不是 classify**：意图只在入口定一次。第二轮再分类一遍纯属浪费，
而且可能分到不同意图导致循环抖动。

`AgentRunner` 与 `LegalRAG` 平级，是**第二条编排路径**而非替代品：`qa(mode="ask")` 仍是
默认，走线性管道；Agent 需显式开启。两者产出同一个 `Answer` 契约对象，因此可以直接对照。

**只用 LangGraph 的状态机，不用它的 LLM 抽象层。** 带工具的调用直接走 `openai` SDK，
于是 state 里的 `messages` 就是 OpenAI 线上格式的 `list[dict]`。

设计论证（为什么 reflect 独立成节点、`retrievable` 是怎么补上的、`recursion_limit` 的
实测最小值、收尾三种成因）见 `docs/DESIGN.md`。
"""

from __future__ import annotations

from typing import Literal

from .. import config
from ..contracts import Answer, Question
from ..obs import Tracer, traced
from ..qa.rag import LegalRAG
from .intent import INTENT_LOOKUP, INTENT_SEARCH, make_classify_node, make_lookup_plan_node
from .llm import ToolCallingLLM
from .nodes import TOOLS, make_agent_node, make_finalize_node, make_tools_node
from .reflect import make_reflect_node
from .state import AgentState
from .tools import build_article_index

__all__ = ["route_after_agent", "build_graph", "AgentRunner"]


def _route_after_classify(state: AgentState) -> Literal["lookup_plan", "agent"]:
    return "lookup_plan" if state.get("intent") == INTENT_LOOKUP else "agent"


def route_after_agent(state: AgentState) -> Literal["tools", "finalize"]:
    """最后一条消息带 tool_calls → 必须先去 tools，**没有任何例外**。

    绝不能在这里看预算：一旦在预算用尽时直接跳到 finalize，历史里就会留下一条
    带 tool_calls 却没有对应 tool 消息的助手消息，OpenAI 兼容端点会直接返回 400。
    """
    messages = state.get("messages") or []
    return "tools" if messages and messages[-1].get("tool_calls") else "finalize"


def _route_after_reflect(state: AgentState) -> Literal["agent", "finalize"]:
    """审核说不够、缺口补得上、且还有预算 → 回规划轮再查一次；否则收尾。

    三个条件是与的关系，缺一不可：**够不够**（`sufficient`）、**补不补得上**
    （`retrievable`，缺口在库外时再检索一百次也检不到，回边只会白烧一轮检索加两次模型
    调用）、**还有没有预算**。

    预算在这里现算（`max_steps - steps`），不落进 state：需要「递减」的字段在
    `operator.add` reducer 下是读-改-写，写错就是双倍消耗，而且路由器是纯函数、
    本来就没有写权限。

    放这里而不是 tools 的出边，是因为这三件事本来就是同一个决策的几半，合在一处判断
    才不会出现「审核说不够、但已经没轮次了」的中间态。
    """
    reflections = state.get("reflections") or []
    verdict = reflections[-1] if reflections else None
    if verdict and not verdict.get("sufficient"):
        if verdict.get("retrievable", True) and state.get("max_steps", 0) - state.get("steps", 0) > 0:
            return "agent"
    return "finalize"


def build_graph(
    *,
    rag: LegalRAG,
    llm: ToolCallingLLM,
    cfg: config.AgentConfig,
    forced_intent: str | None = None,
    reflect_llm: ToolCallingLLM | None = None,
    tracer: Tracer | None = None,
):
    from langgraph.graph import END, START, StateGraph

    # 空 Tracer（默认）下每个 span 只是两次空调用，图的行为与不埋点逐位相同。
    tracer = tracer or Tracer()

    # 审核可以挂另一个模型；不给就用规划轮那个。**默认必须是 `llm` 而不是新建一个
    # 真客户端** —— 测试注入假 LLM 时，审核要是自己造一个，就会当场去打网络。
    reflect_llm = reflect_llm or llm

    # 条号索引只建一次：classify / lookup_plan / tools 三个节点共用同一份。
    # 它是纯函数产物、构造后只读，所以共享是安全的，也避免三份各建一遍。
    parents = rag.parents
    by_number = build_article_index(parents)
    # 库的边界要写进审核提示词，否则审核分不清「这轮没检出来」与「库里根本没有」。
    # **从 parents 现取，不写死一份名单** —— 新增法规时不该有人记得来这里改。
    laws = sorted({parent.law_name for parent in parents.values()})

    graph = StateGraph(AgentState)
    graph.add_node("classify", traced(tracer, "node.classify", make_classify_node(parents, by_number, forced_intent)))
    graph.add_node("lookup_plan", traced(tracer, "node.lookup_plan", make_lookup_plan_node(parents, by_number)))
    graph.add_node("agent", traced(tracer, "node.agent", make_agent_node(llm, cfg)))
    graph.add_node("tools", traced(tracer, "node.tools", make_tools_node(rag, cfg, by_number)))
    graph.add_node("reflect", traced(tracer, "node.reflect", make_reflect_node(reflect_llm, cfg, laws)))
    graph.add_node("finalize", traced(tracer, "node.finalize", make_finalize_node(rag, cfg)))

    # 入口先定意图：条文定位走规则规划（零 LLM），其余才进 LLM 规划轮
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify", _route_after_classify, {"lookup_plan": "lookup_plan", "agent": "agent"}
    )
    # lookup_plan 与 agent 共用一条出边：**判据只有「有没有 tool_calls」，与预算无关**。
    # 共用而不是各写一条，是为了让「有 tool_calls 必去 tools」这条不变量只有一个实现。
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "lookup_plan", route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    # tools → reflect 是**无条件**的：每一次工具调用的结果都要经过一次审核
    graph.add_edge("tools", "reflect")
    graph.add_conditional_edges(
        "reflect", _route_after_reflect, {"agent": "agent", "finalize": "finalize"}
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
        reflect_llm: ToolCallingLLM | None = None,
        cfg: config.AgentConfig | None = None,
        forced_intent: str | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.rag = rag
        self.cfg = cfg or config.agent_config()
        self.forced_intent = forced_intent
        self.tracer = tracer or Tracer()
        self.llm = llm or ToolCallingLLM(retries=self.cfg.retries)
        # 不单独给就**跟着规划轮那个走**（含测试注入的假客户端）。
        # 真实第二个客户端只在 `load()` 里造 —— 那里才知道该读 `AGENT_REFLECT_*`。
        self.reflect_llm = reflect_llm or self.llm
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
        forced_intent: str | None = None,
        tracer: Tracer | None = None,
    ) -> "AgentRunner":
        # top_k 必须一路透到 LegalRAG：它是检索层的默认召回条数，
        # 只存在 AgentRunner 上的话，--top-k 会静默失效（检索仍按配置的 6 条走）。
        agent_cfg = cfg or config.agent_config()
        return cls(
            LegalRAG.load(with_vector=with_vector, top_k=top_k),
            cfg=agent_cfg,
            forced_intent=forced_intent,
            tracer=tracer,
            # 审核单独一个模型。`AGENT_REFLECT_*` 没配时它逐字段回退到 LLM_*，
            # 于是这里多出来的只是**一个配置相同的客户端**，行为与单模型时逐位相同。
            reflect_llm=ToolCallingLLM(
                config.reflect_llm_config(), retries=agent_cfg.retries
            ),
        )

    def graph(self):
        """懒构建 + 缓存：一个 rag 只配一套图。"""
        if self._graph is None:
            self._graph = build_graph(
                rag=self.rag,
                llm=self.llm,
                cfg=self.cfg,
                forced_intent=self.forced_intent,
                tracer=self.tracer,
            )
        return self._graph

    def invoke(self, question: str | Question) -> AgentState:
        """跑完整图，返回**原始终态**（调试 / 测试 / --json 用）。

        外层 `invoke` span 与内层六个 `node.*` 是平的、不嵌套计时 —— 两者相减就是
        langgraph 自己的调度开销，那恰好是「20 分钟里有多少是框架的」的答案。
        """
        query = question if isinstance(question, Question) else Question(text=question)
        initial: AgentState = {
            "question": query.text,
            "history": [tuple(pair) for pair in query.history],
            "top_k": query.top_k or self.top_k,
            "max_steps": self.cfg.max_steps,
            "intent": "",
            "messages": [],
            "search_log": [],
            "steps": 0,
            "usage": [],
            "reflections": [],
        }
        # 显式设：langgraph 1.x 的默认值是 10007，等于没有限制。
        # 这条是「路由写错」时的最后一道保险 —— 宁可报错，不要静默死循环。
        # 它按 superstep 计（不是按轮），所以必须跟着图的形状走。
        #
        # **这个余量在加入意图识别时被吃光过一次**：原值是 3*max_steps + 3，
        # 恰等于新图的最小值 12 —— 一个 superstep 都不能再插，插了就
        # GraphRecursionError。图变了就得重测最小值，别信推导。
        # 现测：检索路 3*max_steps + 3；定位路恒为 6。给到 3*max_steps + 6。
        with self.tracer.span("invoke"):
            return self.graph().invoke(
                initial, config={"recursion_limit": 3 * self.cfg.max_steps + 6}
            )

    def ask(self, question: str | Question) -> Answer:
        """提问 → 带 [依据N] 标注的答案。未配 LLM 时退回线性管道。"""
        query = question if isinstance(question, Question) else Question(text=question)
        if not self.llm.available:
            # 没有 key 就连规划轮都跑不了，直接退回单轮管道。
            # 它同样会拒答（UNAVAILABLE_ANSWER），行为与基线一致。
            return self.rag.ask(query)

        state = self.invoke(query)
        answer = state.get("answer")
        if answer is None:  # 图正常跑完必有 answer；真缺了就说出来，不要静默给空
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
            f"意图 2 类（{'、'.join((INTENT_LOOKUP, INTENT_SEARCH))}，纯规则） | "
            f"工具 2 个（{'、'.join(tool['function']['name'] for tool in TOOLS)}） | "
            f"LLM {self.llm.cfg.model}"
        )
