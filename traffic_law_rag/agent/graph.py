"""Agent 循环：检索 → 自评 → 改写重检 → 收尾（阶段二）。

    AgentRunner.ask : str → Answer

与 `LegalRAG` 平级的**第二条编排路径**，不是替代品：`qa(mode="ask")` 仍是默认，
走线性管道；Agent 需显式开启。两者产出同一个 `Answer` 契约对象，因此可以直接对照。

图结构（6 个节点 + 4 个路由）：

    START → classify ─(条文定位)→ lookup_plan ─┐
                  └──(法规检索)→ agent ────────┤─(有 tool_calls)→ tools → reflect
                                              │                          │
                                  (无 tool_calls)┘      (不够 且 有预算)──┘
                                              └──→ finalize ←──(够了 / 预算尽)
                                                     │
                                                    END

**意图识别放在最前面，而且是纯规则、零 LLM 调用。** 判据只有一个：题面里写没写「第X条」、
且那个条号能不能唯一定位到一条法条 —— 能，就不用检索，用户已经把条号告诉你了，取出来即可
（`get_article`）；不能，就老老实实走混合检索。

这样做不是为了省一次调用（虽然确实省了），而是为了让「这一轮在查什么」变成**可观测、可单测**的
状态：它落在 `state["intent"]` 上，`render_trace` 会打出来，也能用 `--intent` 强行走另一边做对照。
交给模型去隐式决定的话，这两件事都做不到。

**回边指向 agent 而不是 classify**：意图只在入口定一次。第二轮再分类一遍纯属浪费，
而且可能分到不同意图导致循环抖动。这一点与最初的描述不同，是有意为之。

**为什么要独立的 reflect 节点，而不是让 agent 自己判断「够了没」**：
那是标准 ReAct 的做法，本图刻意不这么做。原因是**终止条件不该由一个绑着工具的模型给出** ——
工具可用时模型有很强的调用倾向，让「不产生 tool_call」充当停止信号，等于把停止设计成
最不自然的那条路。reflect 节点有两个别的节点给不了的性质：

1. **不绑工具** —— 没有可调用的东西，调用倾向无从产生；
2. **直接被问一个 yes/no** —— 靠结构化输出表态，不用靠「什么都没做」来暗示。

它还有一个副产品：**循环的终止判断天然落在这里**。「够不够」「再检一次补不补得上」
「还剩几轮」本来就是同一个决策的三半，而 reflect 只在**还有预算可行动时**才花钱调模型 ——
预算已尽就没必要问「够不够」了，答案必然是「进入作答」。

其中「补不补得上」（`retrievable`）是后来实测补上的。原先审核只答「证据够不够回答问题」，
而路由把它当成了「要不要再检索一次」—— 这两件事并不等价。用户问到**库外**的东西时
（《治安管理处罚法》的责任、商业保险的合同约定、紧急避险的免责），审核诚实地答「不够」，
循环就去再检一次，可那个缺口再检一百次也补不上。100 题那次审核一共跑了 **207 轮**，其中
**162 轮**是真的调了模型（另 45 轮预算已尽、reflect 短路，压根没问）—— 白跑就烧在这 162 轮里。
所以审核现在多答一个 `retrievable`（缺口在不在库内），路由只在它为真时才回边 ——
提示词里因此要带上库的法规清单，否则审核无从判断「库内」的边界在哪。

**但一个字段还不足以让它判得准**：审核每轮是从零重判的，手上没有「上一轮补上了没」这个依据。
实测 45 对相邻的「不够」判定里，`missing` 文案相似度中位 0.82、24 对在 0.8 以上 —— 第 2 轮检索
基本没改变它的看法。而同一批数据里规划轮其实很听话（有第 2 轮的 62 道题里，检索词平均 68% 的
字符落在上轮 `missing` 里，48 道覆盖率 ≥50%）：缺口没补上不是没去查，是**查了也没有**。
所以提示词还要它去看对话历史里自己上一轮写的那条「[检索审核] 还缺：」—— 同一个缺口
连着两轮要不到，就是库外最直接的证据。

**只用 LangGraph 的状态机，不用它的 LLM 抽象层。** 带工具的调用直接走 `openai` SDK
（与 qa/generator.py 同一种写法），于是 state 里的 `messages` 就是 OpenAI 线上格式的
`list[dict]`：`json.dumps` 直接可过，也不需要 `add_messages` 那层会改变消息形态的转换。

**生成层的提示词一个字都不改**：收尾节点把 N 次检索合并回一个 `RetrievalResult`，
原样交给既有的 `AnswerGenerator.generate()`。所以「Agent 的答案」和「线性管道的答案」
是同一段代码产出的，对照实验比的才真的是架构差异。
"""

from __future__ import annotations

import json
import operator
import sys
import time
from dataclasses import replace
from typing import Annotated, Any, Literal, TypedDict

from .. import config
from ..contracts import Answer, ParentChunk, Question, RetrievalResult
from ..qa.rag import LegalRAG
from .tools import (
    GET_ARTICLE_NAME,
    GET_ARTICLE_TOOL,
    SEARCH_LAW_NAME,
    SEARCH_LAW_TOOL,
    build_article_index,
    find_article,
    lookup_article,
    merge_retrievals,
    parse_article_arguments,
    parse_tool_arguments,
    render_tool_result,
    resolve_law_id,
    tool_message,
)

# 公开面 = **这个文件之外真的会 import 的名字**。全仓没有一处 `import *`，
# 所以这份表不改变任何运行时行为，它的用处是让「入口」与「零件」一眼可分。
#
# 分层：入口（外面会 import 的）、被测试单独钉住的行为、可调文本。
# `build_graph` 严格说只被测试 import，但它是整个集成测试套件的入口，留着。
# 节点的工厂与路由函数里，只有 `build_graph` 自己用的那几个已经加了 `_` 前缀；
# `make_classify_node` / `make_lookup_plan_node` / `route_after_agent` 不加 ——
# 它们各有测试单独调用，属于真有外部消费者的那一半。
#
# 测试会直接戳 `_parse_reflection` 这类内部件，那是测试的特权，不算公开面。
__all__ = [
    # 入口
    "AgentRunner",
    "build_graph",
    "main",
    "ToolCallingLLM",
    "render_trace",
    "AgentState",
    # 被测试单独钉住的行为
    "classify_intent",
    "make_classify_node",
    "make_lookup_plan_node",
    "route_after_agent",
    "INTENT_LOOKUP",
    "INTENT_SEARCH",
    # 可调文本
    "AGENT_SYSTEM_PROMPT",
    "REFLECT_SYSTEM_PROMPT",
    "TOOLS",
]

AGENT_SYSTEM_PROMPT = """你是交通法规问答的检索规划助手。你的任务是决定**这一轮检索什么**。
判断「够不够、要不要继续」是另一个节点的职责，你只管把这一轮该查的查好。

规则：
1. 每次只发起一次检索（一个 search_law 调用）。
2. **第一次检索原样使用用户的问题，不要改写。** 检索层已内置口语→法条用语的自动对齐
   （醉驾→醉酒驾驶）与混合召回，改写是对一条已经处理好的查询再做一次有损加工。
   100 题实测：原话 73/200 条命中，改写成关键词 72/200 —— **改写没换来收益**，
   却让 agent 与线性管道的对照多出一个混淆变量。要换词的是后续轮次（见第 3 条）。
3. 后续轮次只查**审核指出缺失的那一部分**，不要重检已经命中的内容 ——
   最终送进作答的条文数量是固定的，多余的检索会把真正相关的那几条挤出名额。
4. 查询里带上法规名片段（如「深圳」「智能网联汽车」）能提高对应法规的权重；
   若确定只该看某一部法规，用 law_name 参数限定。
5. 最多 %(max_steps)d 轮。"""

REFLECT_SYSTEM_PROMPT = """你是交通法规问答的检索质量审核员。你的任务**不是作答**，
而是判断「已经检索到的法条够不够回答用户的问题」，并且只输出一个 JSON 对象。

**知识库总共只有这 %(law_count)d 部法规，库里没有别的法：**
%(laws)s

判断标准：
- 问题涉及的每个要素（违法情形、处罚种类与数额、记分、适用法规、适用地域）
  都能在已命中的法条里找到直接依据 → sufficient = true。
- 只要有关键要素在已命中的法条里找不到依据 → sufficient = false，
  并在 missing 里写清楚**还缺哪一类规定**，供下一轮检索使用。

sufficient = false 时，还必须回答第二个独立的问题：**这个缺口，再检索一次补得上吗？**

- 补得上（retrievable = true）：缺的那类规定**属于上面这几部法规**，只是这一轮没检出来。
  例如缺记分条款，而记分就写在《中华人民共和国道路交通安全法》里 —— 换个词再检有意义。
- 补不上（retrievable = false）：缺的东西**根本不在上面这几部法规里**。
  例如用户问的责任写在《治安管理处罚法》、是商业保险合同的约定、是紧急避险的免责，
  这些**再检索一百次也检不到** —— 应当就此停下去作答，不要再浪费一轮。

**判断时先看一个具体事实：这个缺口，上一轮要过吗？**
对话历史里可能有你自己上一轮写下的「[检索审核] 还缺：……」。同一个缺口已经要过一次、
这一轮的新检索仍然没补上 —— 这是「库外」最直接的证据：库里有的东西，换个词通常就出来了；
连着两轮要不到，多半是它根本不在这几部法规里。反过来，若这一轮补上了一部分、只差最后一小块，
那就还值得再检一次。

这一条**不要一律倒向某一边**，按你真实的把握判。判错的代价是双向的：判成 true 而其实
检不到，白花一轮检索加两次模型调用；判成 false 而其实检得到，就少了一次本该做的检索。

两个容易判错的地方：
- **法条里没写的内容就是「缺」**，不要因为它「常识上应该有」就判为够。
  例如只找到罚款条款、没找到记分条款，就应判 false 并把「记分」写进 missing。
- 但**问题本身没问的要素不算缺**。不要因为「还想知道别的」而判 false。

**判出缺口之后，再把它写成一句能直接拿去检索的话**（`next_query`），交给下一轮。
`missing` 是给决策看的（缺哪一类规定），`next_query` 是给检索用的 —— 两者不是一回事：

- 写成**一句完整的问话**，像用户会问的那样。不要写成「记分条款」「超载标准」这种名词短语：
  检索层会把口语自动对齐成法条用语（醉驾→醉酒驾驶），拆成关键词反而稀释信号。
- 只问**缺的那一块**，已经查到的东西不要再问一遍。一句话只问一件事。
- 缺口在库外（`retrievable = false`）时留空 —— 那一轮马上就收尾，这句话没人会读到。

例：`missing` →「缺记分条款」，`next_query` →「驾驶机动车不按规定使用安全带的，记多少分？」

只输出 JSON，不要任何其他文字、不要 markdown 代码块：
{"sufficient": true 或 false, "retrievable": true 或 false,
 "reason": "一句话说明判断依据", "missing": "还缺什么；sufficient 为 true 时留空",
 "next_query": "下一轮拿去检索的那句问话；sufficient 为 true 或缺口在库外时留空"}
"""


def _parse_reflection(text: str) -> dict:
    """从模型输出里抠出判断结果。

    宽容解析：模型可能包了 ``` 代码块、可能前后带解释。**解析失败一律判为
    sufficient=True**（停止）而不是 False —— 失败时该往「少花一次检索的钱、直接作答」
    的方向倒，而不是让一个解析 bug 变成多跑一轮的循环。

    `retrievable` 缺失时默认 **True**（照旧回边再检一次），不是 False。理由：
    这个字段是给「缺口在不在库内」用的，而**不确定时不该替模型做停止的决定** ——
    默认 True 时行为与加这个字段之前逐位相同，默认 False 则会在模型不输出它时
    静默改掉循环行为。那不是「省了一轮」，是换了套逻辑。

    `next_query`（缺口写成的一句问话）缺失时默认 **空串**，空串 = 不给规划轮这句提示，
    也就是与加这个字段之前逐位相同。它只是给下一轮的**补充信息**，缺了不影响回边与否。
    """
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return {
                "sufficient": bool(data.get("sufficient", True)),
                "retrievable": bool(data.get("retrievable", True)),
                "reason": str(data.get("reason") or "").strip(),
                "missing": str(data.get("missing") or "").strip(),
                "next_query": str(data.get("next_query") or "").strip(),
            }
    return {
        "sufficient": True,
        "retrievable": True,
        "reason": f"审核输出无法解析，按已足够处理：{raw[:60]}",
        "missing": "",
        "next_query": "",
    }


# ============================================================ 意图
INTENT_LOOKUP = "条文定位"   # 题面直接给出了条号 → 跳过检索，精确取条
INTENT_SEARCH = "法规检索"   # 其余一切 → 走混合检索

# 规划轮能用的工具。两个一起挂是有意的：`search_law` 每次返回的「相关条：第九十九条」
# 是现有输出里信息量最大、却一直被浪费的字段 —— 有了 get_article，模型才能跟着它去取那一条。
# 这才是真正的多跳：检索负责找到「相关的」，取条负责拿到「就是它」。
TOOLS = [SEARCH_LAW_TOOL, GET_ARTICLE_TOOL]

# 意图节点**不调模型**，纯规则。理由是「条文定位」是纯词法判断：题面里写没写「第X条」
# 是正则一眼能看出来的事，用模型分类只会引入一个新的错误源，还每题多一次调用。
#
# 域外/闲聊**刻意不设意图**：运行时它已经由 `tool_choice="auto"`（模型自己决定不调工具）
# 加上 finalize 的兜底检索处理掉了 —— 模型不调工具就直接进作答，反而是最干净的处理。
# （曾经想过复用 eval 那套「域外判据」，但那是**语料标注启发式**（以「事故叙述」开头
# 或含 3 个以上连续拉丁字母），只对那份语料成立，而且本身还有假阳性。那份语料里的
# 美国题已于 2026-09 删净，判据也一并拆掉了；这里记一笔，免得有人再想去把它捡回来。）
#
# **歧义即回退**：没说哪部法规、而多部法规都有这个条号时，判为「法规检索」，
# 交给检索层（它的法名线索加成还能救一把）。与 resolve_law_id 同一个取舍：
# 宁可走通用路径，也不要猜错法规。


def classify_intent(
    question: str,
    *,
    parents: dict[str, ParentChunk],
    index: dict[tuple[str, int], ParentChunk],
) -> str:
    """题面 → 意图。纯函数，可脱离图单测。

    判据只有一个：**能不能唯一定位到一条法条**。定位得到 → 这条题不需要检索，
    用户已经把条号告诉我们了，取出来就行；定位不到 → 老老实实检索。
    """
    if find_article(question, parents=parents, index=index) is not None:
        return INTENT_LOOKUP
    return INTENT_SEARCH


def _route_after_classify(state: AgentState) -> Literal["lookup_plan", "agent"]:
    return "lookup_plan" if state.get("intent") == INTENT_LOOKUP else "agent"


def make_classify_node(
    parents: dict[str, ParentChunk],
    index: dict[tuple[str, int], ParentChunk],
    forced: str | None = None,
):
    """意图识别。

    读：question
    写：{"intent": str}（无 reducer 的通道，写一次）

    `forced` 是给 A/B 对照用的：把同一道题强行按另一种意图跑一遍，
    看「规则判断」这一层到底贡献了什么。为 None 时才是真实规则。
    """

    def classify_node(state: AgentState) -> dict:
        return {
            "intent": forced
            or classify_intent(state["question"], parents=parents, index=index)
        }

    return classify_node


def make_lookup_plan_node(
    parents: dict[str, ParentChunk], index: dict[tuple[str, int], ParentChunk]
):
    """「条文定位」的规划轮 —— 不调模型，直接构造一条 get_article 调用。

    读：question
    写：{"messages": [带 tool_calls 的助手消息], "steps": 1}

    **刻意不在这里取法条，只产出一个 tool_call。** 让既有的 `tools` 节点去执行，
    换来的是四件事全部零改动：

    - 「每个 tool_call 恰好一条 tool 消息」的不变量由既有代码保证，不需要新的消息形状；
    - `reflect` 看到的仍是普通的工具问答；
    - `search_log` 里仍是普通的 `RetrievalResult.to_dict()` 行 → `merge_retrievals` 零改动；
    - 路由沿用 `route_after_agent`（有 tool_calls 就去 tools），它绝不看预算。

    写 `steps: 1` 是有意的：**由规则驱动的规划也是规划**，它消耗了一轮预算。
    代价是 `steps` 从「只有 agent 写」变成「agent 与 lookup_plan 都写」——
    在 `operator.add` 下每次 +1 仍然正确，只是单一写者的说法不再成立，所以写在这里。
    """

    def lookup_plan_node(state: AgentState) -> dict:
        target = find_article(state["question"], parents=parents, index=index)
        if target is None:
            # classify 已经校验过，正常到不了这里。真到了就退化成一次普通检索，
            # 不要抛异常 —— 意图判断失手不该让整个图崩掉。
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_lookup_1",
                                "type": "function",
                                "function": {
                                    "name": SEARCH_LAW_NAME,
                                    "arguments": json.dumps(
                                        {"query": state["question"]}, ensure_ascii=False
                                    ),
                                },
                            }
                        ],
                    }
                ],
                "steps": 1,
            }

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            # id 固定即可：这条路每个会话只走一次，
                            # 而 agent 那边的 call id 由端点生成，不会撞。
                            "id": "call_lookup_1",
                            "type": "function",
                            "function": {
                                "name": GET_ARTICLE_NAME,
                                "arguments": json.dumps(
                                    {
                                        "article_no": target.article_no,
                                        "law_name": target.law_name,
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            ],
            "steps": 1,
        }

    return lookup_plan_node


# ============================================================ State
class AgentState(TypedDict, total=False):
    """Agent 循环的全部状态。

    `total=False`：节点返回的本来就是**增量**（只带自己写的那几个字段）。
    """

    # ---------- 输入：调用方写一次，所有节点只读，永不返回 ----------
    question: str
    history: list[tuple[str, str]]   # 本轮恒为空，为多轮对话预留
    top_k: int
    max_steps: int                   # 放进 state 而非闭包：轨迹能自描述、可复现

    # ---------- 循环累积：带 reducer ----------
    # 每个 reducer 都是 operator.add，语义统一为「节点返回增量」。
    # messages 不用 add_messages：那会把 dict 强转成 BaseMessage，
    # 正好丢掉「消息就是 OpenAI 线上格式」这个我们刻意选择的性质。
    messages: Annotated[list[dict], operator.add]
    search_log: Annotated[list[dict], operator.add]   # 每次检索一行 RetrievalResult.to_dict()
    steps: Annotated[int, operator.add]               # 已花掉的规划轮数（agent 与 lookup_plan 都写）
    usage: Annotated[list[dict], operator.add]
    reflections: Annotated[list[dict], operator.add]  # 每轮审核的结论；路由读 [-1]

    # ---------- 意图：classify 写一次，无 reducer（LangGraph 默认 LastValue 通道）----------
    # 意图只在**入口**定一次，回边指向 agent 而不是 classify：第二轮再分类一遍纯属浪费，
    # 而且可能分到不同意图、导致循环在两套工具之间抖动。
    intent: str

    # ---------- 输出：finalize 写一次 ----------
    answer: Answer | None


# ============================================================ LLM
def _normalize(response: Any) -> tuple[dict, dict]:
    """SDK 返回值 → (线上格式的助手消息, usage)。"""
    message = response.choices[0].message
    reply: dict[str, Any] = {"role": "assistant", "content": message.content or ""}

    calls = getattr(message, "tool_calls", None) or []
    if calls:
        reply["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ]

    usage: dict[str, int] = {}
    raw = getattr(response, "usage", None)
    if raw is not None:
        usage = {
            "prompt_tokens": raw.prompt_tokens,
            "completion_tokens": raw.completion_tokens,
            "total_tokens": raw.total_tokens,
        }
    return reply, usage


class ToolCallingLLM:
    """带工具调用能力的 LLM 客户端（规划轮专用）。

    与 `AnswerGenerator` 分开，是因为两者的**职责与层契约**不同：
    generate 层的契约是 `Question + RetrievalResult → Answer`，不该被工具调用污染；
    这里只负责「给定消息历史，返回下一条助手消息」。
    """

    def __init__(self, cfg: config.LLMConfig | None = None, *, retries: int = 2) -> None:
        self.cfg = cfg or config.llm_config()
        self.retries = retries
        self._client = None

    @property
    def available(self) -> bool:
        return self.cfg.ready

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
    ) -> tuple[dict, dict]:
        """调一次模型，返回 (助手消息, usage)。重试策略与 generator 一致。"""
        from openai import OpenAI  # 延迟导入

        if self._client is None:
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)

        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
            # 刻意用 "auto" 而不是 "required"：一是不是所有兼容端点都支持 required
            # （百炼就不支持），二是本设计本来就需要模型能自主停下来。
            extra["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature if temperature is None else temperature,
                    **extra,
                )
                return _normalize(response)
            except Exception as exc:  # noqa: BLE001 - 统一重试
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"调用 {self.cfg.model} 失败：{last_error}") from last_error


# ============================================================ 节点
def _make_agent_node(llm: ToolCallingLLM, cfg: config.AgentConfig):
    """规划轮。

    读：question / history / messages / max_steps
    写：{"messages": [助手消息], "steps": 1, "usage": [一行]}
    """

    def agent_node(state: AgentState) -> dict:
        max_steps = state.get("max_steps", cfg.max_steps)
        prompt: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT % {"max_steps": max_steps}}
        ]
        prompt.extend({"role": r, "content": c} for r, c in state.get("history") or ())
        prompt.append({"role": "user", "content": state["question"]})
        prompt.extend(state.get("messages") or ())

        reply, usage = llm.chat(prompt, tools=TOOLS, temperature=cfg.temperature)
        return {"messages": [reply], "steps": 1, "usage": [usage]}

    return agent_node


def _make_tools_node(
    rag: LegalRAG,
    cfg: config.AgentConfig,
    index: dict[tuple[str, int], ParentChunk],
):
    """执行工具调用。

    读：messages[-1].tool_calls / search_log / top_k
    写：{"messages": [每个 tool_call 一条 ToolMessage], "search_log": [新增行]}

    **不变量：每个 tool_call 恰好产出一条 ToolMessage** —— 无论成功、参数非法、
    工具名未知、还是工具抛异常。闭合发生在产生 tool_call 的那个节点里，
    于是「assistant 的 tool_calls 没有对应的 tool 消息」这个会让端点直接 400 的
    状态，在结构上就不可能出现，不需要任何事后修补。
    """

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

            try:
                result = rag.search(query, top_k=top_k, **kwargs)
                # 摘要窗口按检索层实际用的词定位（含口语对齐），不按模型原话
                match_text = rag.expand(query)
            except Exception as exc:  # noqa: BLE001 - 工具失败不该打断循环
                return None, f"检索失败：{exc}。可以换一组关键词再试，或用更通用的说法。"

            return result, render_tool_result(
                result,
                index=position(),
                seen=seen,
                snippet_chars=cfg.snippet_chars,
                match_text=match_text,
            )

        def run_lookup(arguments: str) -> tuple[RetrievalResult | None, str]:
            try:
                article_no, law_name = parse_article_arguments(arguments)
            except ValueError as exc:
                return None, f"参数不合法：{exc}。请修正后重试。"

            result, error = lookup_article(
                article_no, law_name, parents=parents, index=index
            )
            if result is None:
                return None, error
            # 精确取条没有查询词可用来定位窗口，match_text 留空 → 摘要落在条文开头
            # （也就是写着处罚的那句「帽子」），这正是用户点名要看的那一段。
            # 摘要上限用 article_chars 而不是 snippet_chars：只取一条，给得起更大的窗口。
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
                continue  # 失败：已经回了一条说明，这一轮不进证据

            row = result.to_dict()
            out_logs.append(row)
            seen |= {article["parent_id"] for article in row["articles"]}

        return {"messages": out_messages, "search_log": out_logs}

    return tools_node


def _make_reflect_node(llm: ToolCallingLLM, cfg: config.AgentConfig, laws: list[str]):
    """审核：现有证据够不够回答问题，以及不够的那部分补不补得上。

    读：question / history / messages / steps / max_steps / search_log
    写：{"reflections": [结论], "messages": [给规划轮的补充说明]}

    **不绑工具**，只问两个判断题（够不够 / 补不补得上）—— 这是它和 `agent` 节点的根本区别，
    也是它存在的理由（见模块文档）。温度用 0：这是判断题，不要它发挥。

    `laws` 是**库内全部法规名**，作为「库的边界」写进提示词。没有它，审核分不清
    「这一轮没检出来」和「库里根本没有」这两件事，于是一律判「不够」，把预算全烧在
    补不上的缺口上 —— 实测 100 题里有 64/162 次「不够」判定属于这一类（缺的责任写在
    《治安管理处罚法》、是商业保险的合同约定、是紧急避险的免责，都不是库内这 6 部法的事）。
    名单纯从 `parents` 现取，不写死一份：新增法规时不该有人记得来这里改。

    预算已尽时**不调模型**直接短路：没有剩余轮次可行动，「够不够」的答案不影响任何后续决策，
    问它纯属浪费一次调用。
    """

    # 前缀只算一次：库边界在一次运行内不变，而这是每次审核都要带上的固定开销。
    prompt_head = REFLECT_SYSTEM_PROMPT % {
        "law_count": len(laws),
        "laws": "\n".join(f"- {name}" for name in laws),
    }

    def reflect_node(state: AgentState) -> dict:
        max_steps = state.get("max_steps", cfg.max_steps)
        if max_steps - state.get("steps", 0) <= 0:
            return {
                "reflections": [
                    {
                        "sufficient": False,
                        "retrievable": True,
                        "reason": f"已达最大轮数 {max_steps}",
                        "missing": "",
                        "next_query": "",
                    }
                ]
            }

        prompt: list[dict] = [{"role": "system", "content": prompt_head}]
        prompt.extend({"role": r, "content": c} for r, c in state.get("history") or ())
        prompt.append({"role": "user", "content": state["question"]})
        prompt.extend(state.get("messages") or ())

        reply, _usage = llm.chat(prompt, temperature=0.0)
        reflection = _parse_reflection(reply.get("content") or "")

        update: dict = {"reflections": [reflection]}
        # 把「还缺什么」回灌给规划轮。用 user 角色：工具消息之后任何角色都合法，
        # 而 user 是各家兼容端点支持得最稳的那个。
        #
        # **只在真要回边时才回灌**：判成「库外」的那条缺口马上就收尾了，这条消息没人会读到，
        # 留在历史里反而与「就此停止」的决定自相矛盾。
        if not reflection["sufficient"] and reflection["retrievable"] and reflection["missing"]:
            content = f"[检索审核] 还缺：{reflection['missing']}"
            # 审核顺手把这个缺口写成了**一句问话**，规划轮可以直接拿它当检索词。
            # 没写（旧模型/解析不出来）就只发上面那半句 —— 与加这个字段之前逐位相同。
            if reflection.get("next_query"):
                content += f"\n[检索审核] 建议查：{reflection['next_query']}"
            update["messages"] = [{"role": "user", "content": content}]
        return update

    return reflect_node


def _trajectory_notes(state: AgentState, merged: RetrievalResult, cfg: config.AgentConfig) -> list[str]:
    """把循环本身的成本写进 Answer.notes —— 不摆出来，就没法解释「凭什么是 3 倍」。

    **计数从消息历史里数，不从 `steps` 和 `search_log` 反推。** 两个都曾经是准确的，
    加了规则取条之后都不是了：

    - `steps` 是**预算**，规则取条也消耗一轮（那是有意的）。拿它当「LLM 规划了几轮」
      会虚报 —— 条文定位那条路一次 LLM 都没调，却报「1 轮规划」。
    - `search_log` 的行数不是检索次数：`get_article` 的产物同形状，但它一次检索都没做，
      embedding 与 BM25 都没碰。用「1 次检索」解释一次零检索的运行，成本归因就错了。

    所以 LLM 规划轮数取 `usage` 的行数（只有 `agent` 节点写它），检索/取条次数从
    assistant 消息里的 tool_call 名字数。三者互不同源，各自准确。
    """
    calls = [
        (call.get("function") or {}).get("name") or ""
        for message in state.get("messages") or ()
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or ()
    ]
    plans = len(state.get("usage") or ())          # agent 每调一次模型写一行
    searches = calls.count(SEARCH_LAW_NAME)
    lookups = calls.count(GET_ARTICLE_NAME)
    reviews = len(state.get("reflections") or ())
    steps = state.get("steps", 0)
    max_steps = state.get("max_steps", cfg.max_steps)

    parts = [f"{plans} 轮 LLM 规划", f"{searches} 次检索"]
    if lookups:
        parts.append(f"{lookups} 次精确取条")   # 只在真发生过时才出现，别占版面
    parts += [f"{reviews} 轮审核", f"证据 {len(merged.articles)} 条"]
    notes = ["Agent：" + " / ".join(parts)]

    # 「被迫收尾」的判据要与 _route_after_reflect 逐字同源：审核说了不够、且预算真的见底。
    # 光看 steps >= max_steps 不够 —— 规则取条那条路根本不缺轮次，它是**按设计**一轮结束的。
    #
    # 收尾有三种成因，轨迹里要分得开，否则「为什么 2 轮就停了」没法解释：
    # 审核说够了 / 审核说缺口在库外 / 预算耗尽。
    reflections = state.get("reflections") or []
    verdict = reflections[-1] if reflections else None
    if verdict and not verdict.get("sufficient"):
        if steps >= max_steps:
            notes.append(f"已达最大轮数 {max_steps}，强制进入作答")
        elif not verdict.get("retrievable", True):
            notes.append("审核判定缺口不在库内（再检索也补不上），提前进入作答")
    tokens = sum(int(row.get("total_tokens") or 0) for row in state.get("usage") or ())
    if tokens:
        notes.append(f"规划轮 token 合计 {tokens}")
    return notes


def _make_finalize_node(
    rag: LegalRAG,
    cfg: config.AgentConfig,
):
    """收尾：把累积的证据合并回一个 RetrievalResult，交给既有生成器。

    读：question / history / search_log / steps / max_steps / usage / top_k
    写：{"answer": Answer, "search_log": [兜底检索那一行，否则 []]}

    **这里不喂工具历史、也不 bind_tools** —— 从零重建一次「问题 + 依据」的提示词。
    这样既绕开了可能残留的悬空 tool_calls，又让提示词与线性管道逐字节相同。

    **不接 `top_k` 参数**：那曾是 runner 的条数，只在 `state` 缺字段时兜底，
    而 `AgentRunner.invoke()` 必然写 `state["top_k"]` —— 它一次都没生效过，
    却让同一个函数里坐着两个 `top_k`（收尾读闭包、工具轮读 state），
    调用方传进来的 `Question` 自带 `top_k` 时两者就分叉。现在兜底直接读 config。
    """

    def finalize_node(state: AgentState) -> dict:
        # `state["top_k"]` 由 invoke 写成 `query.top_k or runner.top_k`，单题覆盖落在它上面。
        # 兜底读 config 而不另接参数：这个值的真源只有 config 一处，与 tools_node 同款。
        active_top_k = state.get("top_k", config.retrieve_config().top_k)
        logs = list(state.get("search_log") or ())
        extra: list[dict] = []
        notes: list[str] = []

        if not logs:
            # 模型一次都没检索 → 按单轮管道兜底。这一条保证 Agent 严格增量，不比基线差。
            result = rag.search(state["question"], top_k=active_top_k)
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
            # 生成器今天并不读 `Question.top_k`（`build_evidence` 用的是 `show_top`），
            # 所以这一行不改变任何行为；跟着 `active_top_k` 走只为不再留两个 top_k ——
            # 哪天生成器开始读它，不会把这个坑悄悄带回来。
            top_k=active_top_k,
        )
        # 既有入口，零改动：Agent 的答案与线性管道的答案是同一段代码产出的
        answer = rag.answer(question, merged)
        return {"answer": replace(answer, notes=answer.notes + tuple(notes)), "search_log": extra}

    return finalize_node


# ============================================================ 路由
def route_after_agent(state: AgentState) -> Literal["tools", "finalize"]:
    """最后一条消息带 tool_calls → 必须先去 tools，**没有任何例外**。

    绝不能在这里看预算：一旦在预算用尽时直接跳到 finalize，历史里就会留下一条
    带 tool_calls 却没有对应 tool 消息的助手消息，OpenAI 兼容端点会直接返回 400。
    """
    messages = state.get("messages") or []
    return "tools" if messages and messages[-1].get("tool_calls") else "finalize"


def _route_after_reflect(state: AgentState) -> Literal["agent", "finalize"]:
    """审核说不够、缺口补得上、且还有预算 → 回规划轮再查一次；否则收尾。

    三个条件是与的关系，缺一不可：

    1. **够不够**（`sufficient`）：证据已覆盖问题要素就没必要再查。
    2. **补不补得上**（`retrievable`）：缺口若落在库外（用户问的责任写在
       《治安管理处罚法》、是商业保险的合同约定……），再检索一百次也检不到，
       回边只会白烧一轮检索加两次模型调用。100 题那次审核跑了 207 轮，其中 162 轮真的调了
       模型（另 45 轮预算已尽、reflect 短路），白跑就烧在这 162 轮里。
    3. **还有没有预算**：预算在这里现算（`max_steps - steps`），不落进 state：需要
       「递减」的字段在 operator.add reducer 下是读-改-写，写错就是双倍消耗，
       而且路由器是纯函数、本来就没有写权限。

    放这里而不是 tools 的出边，是因为这三件事本来就是同一个决策的几半，合在一处判断
    才不会出现「审核说不够、但已经没轮次了」的中间态。
    """
    reflections = state.get("reflections") or []
    verdict = reflections[-1] if reflections else None
    if verdict and not verdict.get("sufficient"):
        if verdict.get("retrievable", True) and state.get("max_steps", 0) - state.get("steps", 0) > 0:
            return "agent"
    return "finalize"


# ============================================================ 组图
def build_graph(
    *,
    rag: LegalRAG,
    llm: ToolCallingLLM,
    cfg: config.AgentConfig,
    forced_intent: str | None = None,
    reflect_llm: ToolCallingLLM | None = None,
):
    from langgraph.graph import END, START, StateGraph

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
    graph.add_node("classify", make_classify_node(parents, by_number, forced_intent))
    graph.add_node("lookup_plan", make_lookup_plan_node(parents, by_number))
    graph.add_node("agent", _make_agent_node(llm, cfg))
    graph.add_node("tools", _make_tools_node(rag, cfg, by_number))
    graph.add_node("reflect", _make_reflect_node(reflect_llm, cfg, laws))
    graph.add_node("finalize", _make_finalize_node(rag, cfg))

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
    # 回边指向 agent 而不是 classify：意图只在入口定一次。第二轮再分类一遍纯属浪费，
    # 而且可能分到不同意图导致循环抖动。
    graph.add_conditional_edges(
        "reflect", _route_after_reflect, {"agent": "agent", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    return graph.compile()


# ============================================================ 门面
class AgentRunner:
    """Agentic RAG 门面（与 `LegalRAG` 平级）。

    Input : 自然语言问题 (str) / Question
    Output: Answer（与单轮管道同一个契约对象，可直接对照）

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
    ) -> None:
        self.rag = rag
        self.cfg = cfg or config.agent_config()
        self.forced_intent = forced_intent
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
    ) -> "AgentRunner":
        # top_k 必须一路透到 LegalRAG：它是检索层的默认召回条数，
        # 只存在 AgentRunner 上的话，--top-k 会静默失效（检索仍按配置的 6 条走）。
        agent_cfg = cfg or config.agent_config()
        return cls(
            LegalRAG.load(with_vector=with_vector, top_k=top_k),
            cfg=agent_cfg,
            forced_intent=forced_intent,
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
            )
        return self._graph

    def invoke(self, question: str | Question) -> AgentState:
        """跑完整图，返回**原始终态**（调试 / 测试 / --json 用）。"""
        query = question if isinstance(question, Question) else Question(text=question)
        return self.graph().invoke(
            {
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
            },
            # 显式设：langgraph 1.x 的默认值是 10007，等于没有限制。
            # 这条是「路由写错」时的最后一道保险 —— 宁可报错，不要静默死循环。
            # 它按 superstep 计（不是按轮），所以必须跟着图的形状走。
            #
            # 实测各条路能跑完的最小值（二分出来的，不是推出来的）：
            #   检索路 max_steps=1/2/3 → 6 / 9 / 12，即 3*max_steps + 3；
            #   定位路 → 恒为 6（classify→lookup_plan→tools→reflect→finalize）。
            # 给到 3*max_steps + 6，比最小值宽 3 个 superstep。
            #
            # **这个余量在加入意图识别时被吃光过一次**：原值是 3*max_steps + 3，
            # 恰等于新图的最小值 12 —— 一个 superstep 都不能再插，插了就
            # GraphRecursionError。图变了就得重测最小值，别信推导。
            config={"recursion_limit": 3 * self.cfg.max_steps + 6},
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


# ============================================================ 轨迹
def render_trace(state: AgentState) -> str:
    """把终态渲染成人读的决策链：意图 → 每轮「调了什么工具 → 命中几条」→ 审核说够不够。"""
    messages = state.get("messages") or ()
    logs = state.get("search_log") or []
    reflections = state.get("reflections") or []
    max_steps = state.get("max_steps", 0)

    intent = state.get("intent") or INTENT_SEARCH
    route = (
        "规则直接取条（零 LLM、零向量/BM25）"
        if intent == INTENT_LOOKUP
        else "进入 LLM 规划轮"
    )
    lines = [
        f"[agent] 提问：{state.get('question', '')}",
        f"[agent] 意图：{intent}（纯规则）→ {route}",
    ]

    step = logged = 0
    cursor = 0
    while cursor < len(messages):
        message = messages[cursor]
        cursor += 1
        if message.get("role") != "assistant":
            continue
        step += 1
        calls = message.get("tool_calls") or []
        if not calls:
            lines.append(f"[agent] 轮 {step}/{max_steps} → 未再调用工具，进入作答")
            continue

        # tool 消息与 tool_call 一一对应、紧跟在助手消息之后（tools 节点的不变量），
        # 所以这里按数量切片配对，而不是靠扫描。
        replies = messages[cursor : cursor + len(calls)]
        cursor += len(calls)

        for call, reply in zip(calls, replies):
            function = call.get("function") or {}
            name = function.get("name") or "?"
            text = reply.get("content") or ""
            args = function.get("arguments") or "{}"
            # **只有成功的调用才进 search_log**（失败的只回一条说明）。
            # 所以命中数不能按「第几个调用」去索引日志 —— 那样一旦前面有失败，
            # 后面每一轮都会错位，把 A 轮的命中数安到 B 轮头上。
            if not text.startswith("检索#"):
                lines.append(f"[agent] 轮 {step}/{max_steps} → {name}({args}) → 未取到：{text[:60]}")
                continue
            logged += 1
            row = logs[logged - 1] if logged <= len(logs) else None
            hits = len(row.get("articles") or ()) if row else 0
            lines.append(f"[agent] 轮 {step}/{max_steps} → {name}({args}) → 命中 {hits} 条")

        # 这一轮检索之后的审核结论（reflect 每轮至多写一条）
        if step - 1 < len(reflections):
            verdict = reflections[step - 1]
            tail = verdict.get("missing") or verdict.get("reason") or ""
            if verdict.get("sufficient"):
                lines.append(f"[agent]         审核：够了 —— {verdict.get('reason') or '证据已覆盖问题要素'}")
            elif not verdict.get("retrievable", True):
                # 「补不上」是个决定性结论，和「还不够」不是一回事，轨迹里要分得开
                lines.append(f"[agent]         审核：不够，但缺口在库外、再检也补不上 → 收尾 —— {tail}")
            else:
                # 一并印出审核拟的检索问句：紧下一行就是规划轮实际发出去的检索词，
                # 两者挨着才好看出规划轮有没有照办。
                hint = f" →「{verdict['next_query']}」" if verdict.get("next_query") else ""
                lines.append(f"[agent]         审核：不够 —— {tail}{hint}")

    answer = state.get("answer")
    if answer is not None:
        # 只挑 Agent 自己加的那几条，把生成层的既有 note 滤掉（它们不属于决策链）。
        # 「本轮未取到任何证据」必须在列：它解释了「0 次检索」为什么还能有证据 ——
        # 没有这句话，轨迹就成了自相矛盾的两行。
        for note in answer.notes:
            if note.startswith(("Agent：", "已达", "规划轮", "本轮未取到", "共 ", "证据按")):
                lines.append(f"[agent] {note}")
    return "\n".join(lines)


# ============================================================ 命令行
USAGE = """用法：python -m traffic_law_rag.agent "问题" [选项]

  --trace          打印完整决策链（默认打印）
  --json           输出 JSON（Answer.to_dict）
  --max-steps N    规划轮数上限，默认 2。设 1 可做近似基线的 A/B
  --top-k N        证据条数上限，默认取 RAG_TOP_K
  --no-vector      只用 BM25 检索
  --linear         走单轮管道作对照（不发规划轮请求）
  --intent NAME    强制意图，用于对照：lookup/条文定位 或 search/法规检索。
                   默认按规则判。给含条号的问题加 --intent search，
                   就能看到「不做意图识别」时同一道题会怎样。
"""

# --intent 的取值别名；中文原值也直接收，省得记两套词
INTENT_ALIASES = {
    "lookup": INTENT_LOOKUP,
    "条文定位": INTENT_LOOKUP,
    "search": INTENT_SEARCH,
    "法规检索": INTENT_SEARCH,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args or not args:
        print(USAGE)
        return 0

    question = args[0]

    def option(name: str, cast=int):
        if name in args and args.index(name) + 1 < len(args):
            return cast(args[args.index(name) + 1])
        return None

    overrides = {}
    if option("--max-steps"):
        overrides["max_steps"] = option("--max-steps")
    cfg = replace(config.agent_config(), **overrides) if overrides else None

    forced_intent = None
    if option("--intent", str) is not None:
        raw = option("--intent", str)
        forced_intent = INTENT_ALIASES.get(raw, "")
        if not forced_intent:
            print(f"--intent 只认 {'、'.join(INTENT_ALIASES)}，收到「{raw}」")
            return 1

    try:
        runner = AgentRunner.load(
            with_vector="--no-vector" not in args,
            cfg=cfg,
            top_k=option("--top-k"),
            forced_intent=forced_intent,
        )
    except Exception as exc:  # noqa: BLE001 - Milvus 未起、索引未建等
        print(f"装配失败：{exc}")
        print("提示：先 docker compose up -d standalone，并确认索引已建（python -m traffic_law_rag.kb.indexer）")
        return 1

    print(f"[agent] {runner.describe()}")

    if "--linear" in args:
        # 同一个 rag：对照实验要的是「同一套检索 + 同一个生成器，只是不走图」
        answer = runner.rag.ask(question)
    else:
        try:
            state = runner.invoke(question)
        except RuntimeError as exc:
            print(str(exc))
            return 1
        if "--json" not in args:
            print(render_trace(state))
        answer = state.get("answer")

    if answer is None:
        print("未产出答案")
        return 1

    if "--json" in args:
        import json

        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print()
    print(answer.text)
    if answer.evidences:
        print()
        print("依据：")
        for evidence in answer.evidences:
            print(f"  {evidence.label} {evidence.citation}")
    if answer.usage:
        print(f"\n[tokens] {answer.usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
