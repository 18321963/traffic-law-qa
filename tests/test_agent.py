"""Agent 循环：意图识别、两条路径、悬空工具调用、预算、提示词一致性。

**全部离线**：不连 Milvus、不发网络请求。Milvus 用替身换掉，LLM 用按脚本弹消息的
替身换掉，生成器是真的 `AnswerGenerator` 只是把它的 `_client` 换成记录器 ——
所以「Agent 的答案与线性管道的答案是同一段代码产出的」这句话是被**断言**过的，
不是被相信的。

父块来自 `chunk_set` fixture（现场跑切块），所以证据里的法名、条号、正文全是真货。

`langgraph` 在 `[agent]` 可选依赖组里，没装就整体跳过而不是失败。
"""

from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("langgraph")   # 可选依赖：没装就跳过这一整个文件

from traffic_law_rag import config
from traffic_law_rag.agent import graph as A
from traffic_law_rag.agent.tools import (
    GET_ARTICLE_NAME,
    SEARCH_LAW_NAME,
    SEARCH_LAW_TOOL,
    build_article_index,
    merge_retrievals,
)
from traffic_law_rag.contracts import Question, RetrievalResult, RetrievedArticle
from traffic_law_rag.qa.generator import AnswerGenerator
from traffic_law_rag.qa.rag import LegalRAG

FAKE_LLM_CFG = config.LLMConfig(base_url="http://fake", api_key="fake", model="fake-model")


# ================================================================== 替身
class ScriptedLLM:
    """按脚本弹助手消息的假规划器；记录每次收到的请求供逐字断言。"""

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.calls: list[dict] = []
        self.cfg = FAKE_LLM_CFG

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages, *, tools=None, temperature=None):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools})
        if not self.script:
            raise AssertionError(f"脚本用完了，但第 {len(self.calls)} 次调用还在继续")
        return self.script.pop(0), {"total_tokens": 1}

    # 便捷断言
    def tool_names_seen(self) -> list[list[str]]:
        return [[t["function"]["name"] for t in (c["tools"] or [])] for c in self.calls]


class FakeRetriever:
    """只实现 agent 用到的三样：`parents`、`search`、`expand`。

    **故意不提供 `_store` / `_chunks` / `_rewriter`** —— 替身越薄，越能证明
    `LegalRAG` 没有偷偷穿透到检索器内部（真检索器上那三个是私有属性，
    替身上干脆不存在，穿透会当场 AttributeError 而不是悄悄拿到 None）。

    `fail_on` 让**指定的查询词**炸掉，而不是让整个检索器一直坏 ——
    否则收尾节点的兜底检索也会炸，就测不到「失败回灌之后模型还能继续」了。
    """

    def __init__(self, parents: dict, *, hits: int = 2, fail_on: set[str] | None = None) -> None:
        self.parents = parents
        self.hits = hits
        self.fail_on = fail_on or set()
        self.queries: list[str] = []
        self.expanded: list[str] = []

    def expand(self, text: str) -> str:
        """真检索器上这里会做口语对齐；替身原样返回，摘要退化成改动前的行为。"""
        self.expanded.append(text)
        return text

    def search(self, text: str, top_k: int | None = None, **kwargs) -> RetrievalResult:
        self.queries.append(text)
        if text in self.fail_on:
            raise RuntimeError("检索端点不可用")
        picks = list(self.parents.values())[: self.hits]
        return RetrievalResult(
            query=text,
            articles=tuple(
                RetrievedArticle(article=p, score=0.5, hit_chunks=(f"{p.parent_id}#c1",))
                for p in picks
            ),
            used_vector=True,
            used_bm25=True,
            elapsed_ms=1.0,
        )


class RecordingClient:
    """顶掉生成器的 openai 客户端，把发出去的提示词录下来。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.chat = _Namespace(completions=_Namespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return _Namespace(
            choices=[_Namespace(message=_Namespace(content="答案 [依据1]", tool_calls=None))],
            usage=_Namespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _assistant(content: str = "", calls: list[tuple[str, str, dict]] | None = None) -> dict:
    """构造一条线上格式的助手消息；calls 是 (id, 工具名, 参数字典)。"""
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
            for call_id, name, args in calls
        ]
    return message


def _reflection(sufficient: bool, missing: str = "", *, retrievable: bool | None = None) -> dict:
    """构造一条审核输出。

    `retrievable` **不给就不写这个键** —— 现有用例因此走的都是「模型没输出该字段」
    那条路，默认值 True 的行为才一直有人钉着，而不是被显式传参掩盖过去。
    """
    payload: dict = {"sufficient": sufficient, "reason": "r", "missing": missing}
    if retrievable is not None:
        payload["retrievable"] = retrievable
    return _assistant(json.dumps(payload, ensure_ascii=False))


# ================================================================== 夹具
@pytest.fixture
def parents(chunk_set) -> dict:
    return chunk_set.parent_map()


@pytest.fixture
def by_number(parents):
    return build_article_index(parents)


@pytest.fixture
def generator():
    gen = AnswerGenerator(FAKE_LLM_CFG)
    gen._client = RecordingClient()
    return gen


def run(
    question, script, parents, generator, *,
    forced=None, max_steps=None, retriever=None, reflect_llm=None,
):
    """跑一次完整图，返回 (终态, 假 LLM, 假检索器, 提示词记录器)。

    **不传 `max_steps` 就取 `AgentConfig` 的出厂默认值**（现在是 2，实测定的）。
    这里曾经写死一个字面量 `2`，注释却声称「跟着 AgentConfig 的默认走」—— 承诺和实现
    是两回事：默认值改成 3，红的只有下面那条钉默认值的守卫 `test_默认轮数上限是2`，
    跑图的测试一条都不会红 —— 它们仍然全在测 2 轮那条路。现在改默认值会连带震动一批
    跑图的测试（实测把默认压到 1，本文件 9 条图测试立刻红），那才是「跟着默认走」。
    取的是 `config.AgentConfig()` 的 dataclass 默认，**不是 `agent_config()`** ——
    后者读环境变量，本机配了 `AGENT_MAX_STEPS` 就该听本机的，测试不该被它左右。

    第三个返回值仍然是**那个假检索器**（不是 rag），这样各处
    `retriever.queries` / `retriever.expanded` 的断言一行都不用动。

    `top_k=6` 必须显式传给 `LegalRAG`：不给的话它取 `config.retrieve_config().top_k`，
    本机恰好也是 6，于是「提示词与线性管道逐字节相同」那条测试会**靠运气**通过 ——
    换台机器上配了别的值就红，而且红得莫名其妙。
    """
    llm = ScriptedLLM(script)
    retriever = retriever or FakeRetriever(parents)
    rag = LegalRAG(retriever, generator, top_k=6)
    if max_steps is None:
        max_steps = config.AgentConfig().max_steps
    cfg = config.AgentConfig(max_steps=max_steps)
    graph = A.build_graph(
        rag=rag,
        llm=llm,
        cfg=cfg,
        forced_intent=forced,
        reflect_llm=reflect_llm,
    )
    state = graph.invoke(
        {
            "question": question,
            "history": [],
            "top_k": 6,
            "max_steps": max_steps,
            "intent": "",
            "messages": [],
            "search_log": [],
            "steps": 0,
            "usage": [],
            "reflections": [],
        },
        config={"recursion_limit": 3 * max_steps + 6},
    )
    return state, llm, retriever, generator._client


ARTICLE_Q = "《深圳经济特区道路交通安全违法行为处罚条例》第十三条怎么规定的"


# ================================================================== 1. 头条：提示词逐字一致
def test_agent_prompt_is_byte_identical_to_linear_pipeline(parents, generator):
    """**这是整个设计的验收标准**：Agent 的答案与线性管道的答案是同一段代码产出的。

    做法不是「读代码确认」，而是把两边真正发给模型的 `messages` 抓出来逐字比。
    如果收尾节点偷偷往证据里加了一句、或换了证据顺序，这条会红。
    """
    state, *_ = run(ARTICLE_Q, [_reflection(True)], parents, generator)
    agent_prompt = generator._client.requests[-1]["messages"]

    # 用同一个 search_log 独立重算一遍合并结果，再走一次同一个生成器
    merged = merge_retrievals(
        state["search_log"], question=ARTICLE_Q, parents=parents, max_evidence=6
    )
    generator.generate(Question(text=ARTICLE_Q, top_k=6), merged)
    linear_prompt = generator._client.requests[-1]["messages"]

    assert agent_prompt == linear_prompt


def test_prompt_carries_the_retrieved_article_text(parents, generator):
    """逐字一致的对比不能是「两边都空」——顺带确认提示词里真有法条正文。"""
    state, *_ = run(ARTICLE_Q, [_reflection(True)], parents, generator)
    prompt = generator._client.requests[-1]["messages"]
    blob = "\n".join(m.get("content") or "" for m in prompt)
    assert "第十三条" in blob
    assert state["answer"].evidences, "证据不该为空"


# ================================================================== 2. 意图分类
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("《深圳经济特区道路交通安全违法行为处罚条例》第十三条怎么规定的", A.INTENT_LOOKUP),
        ("根据道路交通安全法实施条例第六十六条，警车有什么限制", A.INTENT_LOOKUP),
        ("醉驾怎么处罚", A.INTENT_SEARCH),
        # 跨法歧义：「第一条」6 部法都有，不指名 → 回退检索
        ("第一条怎么规定的", A.INTENT_SEARCH),
        # 越界 → 回退检索
        ("《中华人民共和国道路交通安全法》第九千条", A.INTENT_SEARCH),
        # 没说条号，即使指名了法规也要检索
        ("《中华人民共和国道路交通安全法》里罚款收据怎么理解", A.INTENT_SEARCH),
    ],
)
def test_classify_intent(parents, by_number, question, expected):
    assert A.classify_intent(question, parents=parents, index=by_number) == expected


def test_classify_intent_is_pure(parents, by_number):
    """零 LLM 调用、无副作用 —— 同一输入两次结果相同。"""
    first = A.classify_intent(ARTICLE_Q, parents=parents, index=by_number)
    assert first == A.classify_intent(ARTICLE_Q, parents=parents, index=by_number)


# ================================================================== 3. 条文定位全链路
def test_lookup_path_skips_retrieval_entirely(parents, generator):
    """条文定位这条路：零检索调用、零 LLM 规划调用，证据就是点名的那一条。"""
    state, llm, retriever, _ = run(ARTICLE_Q, [_reflection(True)], parents, generator)

    assert state["intent"] == A.INTENT_LOOKUP
    assert retriever.queries == [], "不该走向量/BM25 检索"
    assert len(state["search_log"]) == 1
    assert state["answer"].evidences[0].citation.endswith("第十三条")
    assert state["steps"] == 1

    # 唯一一次 LLM 调用是 reflect 的审核，不是规划 —— 审核不绑工具，所以 tools 是 None
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is None


def test_lookup_plan_shape(parents, generator):
    """lookup_plan 产出恰好一条带 tool_calls 的助手消息，参数是字符串。"""
    state, llm, *_ = run(ARTICLE_Q, [_reflection(True)], parents, generator)
    assistant = [m for m in state["messages"] if m.get("role") == "assistant"]
    assert len(assistant) == 1
    calls = assistant[0]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == GET_ARTICLE_NAME
    assert isinstance(calls[0]["function"]["arguments"], str)
    assert json.loads(calls[0]["function"]["arguments"])["article_no"] == "第十三条"


def test_intent_is_actually_written_to_state(parents, generator):
    """`intent` 必须真的落在 state 上。

    LangGraph 对 TypedDict 里**没声明**的键是**静默丢弃**（不报错、不警告）。
    所以「忘了把 intent 加进 AgentState」这种错不会让任何测试变红，
    只会让分类永远读到 None、永远走 agent 分支 —— 功能假死。
    这条测试是那种错误的唯一防线。
    """
    state, *_ = run(ARTICLE_Q, [_reflection(True)], parents, generator)
    assert state["intent"] == A.INTENT_LOOKUP


def test_forced_intent_overrides_the_rule(parents, generator):
    """`--intent` 对照用：强行按「法规检索」跑同一道题，就必须真的走检索。"""
    state, llm, retriever, _ = run(
        ARTICLE_Q, [_reflection(True)], parents, generator, forced=A.INTENT_SEARCH
    )
    assert state["intent"] == A.INTENT_SEARCH
    assert retriever.queries, "强制走检索时应当真的检索了"


# ================================================================== 4. 悬空工具调用
def test_every_tool_call_gets_exactly_one_tool_message(parents, generator):
    """**不变量**：每个 tool_call 恰好一条 tool 消息。

    违反它的后果不是「答案差一点」，而是 OpenAI 兼容端点直接 400 ——
    历史里留下一条带 tool_calls 却没有对应 tool 消息的助手消息就炸。
    """
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "玩手机"})]),
        _reflection(False, "缺数额"),
        _assistant(calls=[("c2", GET_ARTICLE_NAME, {"article_no": "第十三条"})]),
        _reflection(True),
    ]
    state, *_ = run("深圳开车玩手机罚多少", script, parents, generator)
    asked = [
        call["id"]
        for message in state["messages"]
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or ()
    ]
    answered = [m.get("tool_call_id") for m in state["messages"] if m.get("role") == "tool"]
    assert asked == answered
    assert len(asked) == 2


def test_budget_exhausted_still_goes_through_tools():
    """预算已满时 `route_after_agent` 仍必须返回 tools。

    绝不能在这里看预算直接跳 finalize —— 那样就留下悬空的 tool_call。
    """
    state = {
        "messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]}],
        "steps": 99,
        "max_steps": 3,
    }
    assert A.route_after_agent(state) == "tools"


def test_route_after_agent_without_tool_calls():
    assert A.route_after_agent({"messages": [{"role": "assistant", "content": "够了"}]}) == "finalize"


# ================================================================== 5. 预算
def test_budget_exhaustion_still_produces_an_answer(parents, generator):
    """模型永远要调工具 → agent 恰好执行 max_steps 次，仍产出 Answer。"""
    script = []
    for index in range(3):
        script.append(_assistant(calls=[(f"c{index}", SEARCH_LAW_NAME, {"query": f"词{index}"})]))
        script.append(_reflection(False, "还不够"))

    state, llm, retriever, _ = run("永远查不够的问题", script, parents, generator, max_steps=3)

    assert state["steps"] == 3
    assert state["answer"] is not None
    assert "已达最大轮数 3" in " ".join(state["answer"].notes)
    # 3 轮规划 + 2 轮审核：第 3 轮的 reflect 看到预算已尽、直接短路，不再调模型
    assert len(llm.calls) == 5
    assert len(retriever.queries) == 3


def test_reflect_does_not_call_the_model_when_budget_is_gone(parents, generator):
    """预算已尽时 reflect 短路，不花钱问一个不影响决策的问题。"""
    script = [_assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "q"})])]
    state, llm, *_ = run("问题", script, parents, generator, max_steps=1)
    assert state["steps"] == 1
    assert len(llm.calls) == 1, "只该有规划那一次，审核不该调模型"


# ================================================================== 6. 工具失败
def test_search_failure_is_fed_back_and_the_model_retries(parents, generator):
    """工具报错 → 回一条中文说明 → 模型再次被调用；失败那次不进证据。"""
    retriever = FakeRetriever(parents, fail_on={"会炸的查询"})
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "会炸的查询"})]),
        _reflection(False, "没查到"),
        _assistant(content="那我不查了"),      # 第 2 轮模型被再次调用 —— 这就是「回灌生效」
    ]
    state, llm, *_ = run("问题", script, parents, generator, max_steps=2, retriever=retriever)

    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "检索失败" in tool_messages[0]["content"]
    assert len(llm.calls) == 3                  # 规划 → 审核 → 再规划

    # 失败那次不进证据。注意 search_log 不是空的：finalize 见一条日志都没有，
    # 会按单轮管道兜底检索一次 —— 那是设计，但兜底那行的 query 是原问题，
    # 不是那次炸掉的查询词。
    assert all(row["query"] != "会炸的查询" for row in state["search_log"])
    assert any("兜底" in note for note in state["answer"].notes)


def test_unknown_tool_still_gets_a_tool_message(parents, generator):
    """未知工具也**必须**回一条 tool 消息，否则悬空 → 400。"""
    script = [
        _assistant(calls=[("c1", "no_such_tool", {})]),
        _reflection(True),
    ]
    state, *_ = run("问题", script, parents, generator)
    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "未知工具" in tool_messages[0]["content"]
    assert tool_messages[0]["tool_call_id"] == "c1"


def test_get_article_failure_is_fed_back(parents, generator):
    """get_article 取不到时回中文说明，同样不产生证据。"""
    script = [
        _assistant(
            calls=[("c1", GET_ARTICLE_NAME, {"article_no": "第九千条", "law_name": "中华人民共和国道路交通安全法"})]
        ),
        _reflection(True),
    ]
    state, *_ = run("问题", script, parents, generator, forced=A.INTENT_SEARCH)
    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert "没有第" in tool_messages[0]["content"]
    # 取条失败没有产出证据；search_log 里那一行是 finalize 的兜底检索，不是这条
    assert all("第九千条" not in row["query"] for row in state["search_log"])


# ================================================================== 7. 空检索兜底
def test_no_evidence_falls_back_to_the_linear_pipeline(parents, generator):
    """模型一次工具都没调 → 按单轮管道兜底检索，保证 Agent 严格增量。"""
    script = [_assistant(content="我直接回答"), _reflection(True)]
    state, llm, retriever, _ = run("域外问题", script, parents, generator)

    assert retriever.queries, "兜底应当真的检索一次"
    assert len(state["search_log"]) == 1
    assert any("兜底" in note for note in state["answer"].notes)


# ================================================================== 8. 回灌一致性
def test_merged_articles_match_the_logged_articles(parents, generator):
    """合并出来的 `Article` 与 `search_log` 里记的必须是同一条（两边独立算出）。"""
    state, *_ = run(ARTICLE_Q, [_reflection(True)], parents, generator)
    merged = merge_retrievals(
        state["search_log"], question=ARTICLE_Q, parents=parents, max_evidence=6
    )
    logged = {a["parent_id"] for row in state["search_log"] for a in row["articles"]}
    assert {a.article.parent_id for a in merged.articles} <= logged
    for evidence in state["answer"].evidences:
        assert evidence.article.parent_id in logged


def test_merge_dedupes_and_truncates(parents):
    """同一父块被两次检索命中只留一条；超过 max_evidence 截断并留 note。"""
    picks = list(parents.values())[:8]
    row_a = RetrievalResult(
        query="a",
        articles=tuple(RetrievedArticle(article=p, score=1.0) for p in picks[:4]),
        used_vector=True, used_bm25=True, elapsed_ms=0.0,
    ).to_dict()
    row_b = RetrievalResult(
        query="b",
        articles=tuple(RetrievedArticle(article=p, score=1.0) for p in picks[2:6]),
        used_vector=True, used_bm25=True, elapsed_ms=0.0,
    ).to_dict()

    merged = merge_retrievals([row_a, row_b], question="q", parents=parents, max_evidence=3)
    ids = [a.article.parent_id for a in merged.articles]
    assert len(ids) == 3
    assert len(set(ids)) == 3                      # 去重
    assert any("截断" in note for note in merged.notes)


def test_merge_interleaves_rather_than_sorts(parents):
    """轮转交错：query1#1、query2#1、query1#2…（跨查询的 RRF 分数不可比）。"""
    picks = list(parents.values())[:4]
    rows = [
        RetrievalResult(
            query=q,
            articles=tuple(RetrievedArticle(article=p, score=1.0) for p in pair),
            used_vector=True, used_bm25=True, elapsed_ms=0.0,
        ).to_dict()
        for q, pair in (("a", picks[0:2]), ("b", picks[2:4]))
    ]
    merged = merge_retrievals(rows, question="q", parents=parents, max_evidence=4)
    assert [a.article.parent_id for a in merged.articles] == [
        picks[0].parent_id, picks[2].parent_id, picks[1].parent_id, picks[3].parent_id,
    ]


# ================================================================== 9. 节点不就地修改 state
def test_nodes_do_not_mutate_the_input_state(parents, by_number):
    """节点返回的是增量，不该就地改传进来的 state（LangGraph 会合并返回值）。"""
    state = {
        "question": ARTICLE_Q,
        "messages": [],
        "search_log": [],
        "steps": 0,
        "max_steps": 3,
        "intent": "",
    }
    before = copy.deepcopy(state)

    A.make_classify_node(parents, by_number)(state)
    A.make_lookup_plan_node(parents, by_number)(state)

    assert state == before


# ================================================================== 10. 失败报告诚实性
def test_trajectory_counts_do_not_count_lookups_as_searches(parents, generator):
    """报告层不能把「精确取条」算成「检索」，也不能把规则取条算成 LLM 规划轮。

    这里钉住的是**成本归因的正确性**：用「1 次检索 / 1 轮规划」解释一次
    零检索、零规划的运行，会让人以为这条路也很贵。
    """
    state, *_ = run(ARTICLE_Q, [_reflection(True)], parents, generator)
    note = next(n for n in state["answer"].notes if n.startswith("Agent："))
    assert "0 轮 LLM 规划" in note
    assert "0 次检索" in note
    assert "1 次精确取条" in note
    assert "已达最大轮数" not in " ".join(state["answer"].notes)


def test_trajectory_reports_forced_stop_only_when_budget_really_ran_out(parents, generator):
    """审核说不够 + 预算见底，才叫「被迫收尾」；预算没花完的路径不该这么说。"""
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "q"})]),
        _reflection(False, "还不够"),
    ]
    state, *_ = run("问题", script, parents, generator, max_steps=1)
    assert "已达最大轮数 1" in " ".join(state["answer"].notes)


# ================================================================== 11. 轨迹渲染
def test_render_trace_shows_intent_and_real_tool_names(parents, generator):
    """轨迹要打出意图，且工具名必须来自 tool_call 本身，不能写死。"""
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "玩手机"})]),
        _reflection(True),
    ]
    state, *_ = run("深圳开车玩手机罚多少", script, parents, generator)
    trace = A.render_trace(state)
    assert f"意图：{A.INTENT_SEARCH}" in trace
    assert f"{SEARCH_LAW_NAME}(" in trace
    assert "命中 2 条" in trace


def test_render_trace_reports_failed_calls_without_misnumbering(parents, generator):
    """失败的调用不产生日志行 —— 命中数不能按「第几个调用」去索引，否则会错位。"""
    script = [
        _assistant(calls=[("c1", "no_such_tool", {})]),        # 失败，不产日志行
        _reflection(False, "再来"),
        _assistant(calls=[("c2", SEARCH_LAW_NAME, {"query": "q"})]),   # 成功，第 1 行日志
        _reflection(True),
    ]
    state, *_ = run("问题", script, parents, generator)
    assert len(state["search_log"]) == 1
    trace = A.render_trace(state)
    assert "未取到" in trace
    assert "命中 2 条" in trace       # 第 2 轮的命中数是它自己那次的，不是错位来的


def test_render_trace_shows_forced_stop(parents, generator):
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "q"})]),
        _reflection(False, "还不够"),
    ]
    state, *_ = run("问题", script, parents, generator, max_steps=1)
    assert "已达最大轮数 1" in A.render_trace(state)


def test_render_trace_explains_evidence_without_retrieval(parents, generator):
    """模型一次工具都没调时，轨迹会同时出现「0 次检索」和「证据 N 条」—— 必须给出解释。

    `finalize` 有个兜底：一次检索都没有时按单轮管道补一次，保证 Agent 不比基线差。
    兜底那句话若不打进轨迹，这两行就成了自相矛盾的输出，读起来像 bug。渲染层按前缀
    白名单挑 note，新加的 note 很容易被漏掉 —— 所以这条钉住的是「兜底那句必须在」。
    """
    script = [_reflection(True)]      # 第一轮就不调工具
    state, *_ = run("怎么做红烧肉", script, parents, generator)
    trace = A.render_trace(state)

    assert "0 次检索" in trace
    assert "证据" in trace
    assert "本轮未取到任何证据" in trace    # 那个「为什么有证据」的答案
    assert "兜底检索一次" in trace


# ================================================================== 12. 惰性导入
def test_importing_the_package_does_not_pull_langgraph():
    """`import traffic_law_rag` 永远不触发 langgraph 导入。

    这是「Agent 是增量能力」在依赖层面的兑现：没装 `[agent]` extra 的人
    照常走线性管道，不该因为一个可选能力而 import 失败。
    """
    import subprocess
    import sys

    code = "import sys, traffic_law_rag; sys.exit(1 if 'langgraph' in sys.modules else 0)"
    assert subprocess.call([sys.executable, "-c", code]) == 0


# ================================================================== 13. 库外缺口不空跑
def test_库外缺口直接收尾不回边(parents, generator):
    """审核说「不够，但库里没有」→ 收尾，不回边。

    回边只会白烧一轮检索加两次模型调用 —— 缺口在库外（这里是《治安管理处罚法》，
    不在库内那 6 部法里），再检一百次也检不到。100 题那次审核跑了 207 轮，其中 162 轮真的调了
    模型（另 45 轮预算已尽短路），白跑就烧在这 162 轮里。
    """
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "阻碍执行职务"})]),
        _reflection(False, "缺《治安管理处罚法》的责任条款", retrievable=False),
    ]
    state, llm, retriever, _ = run("阻碍交警执法怎么罚", script, parents, generator)

    assert len(retriever.queries) == 1, "库外缺口不该触发第二次检索"
    assert state["steps"] == 1
    assert len(llm.calls) == 2, "规划 + 审核各一次，不该有第二轮规划"
    assert "缺口不在库内" in " ".join(state["answer"].notes)


@pytest.mark.parametrize("retrievable", [True, None], ids=["显式true", "字段缺席"])
def test_库内缺口照旧回边(parents, generator, retrievable):
    """缺口在库内（只是这轮没检出来）→ 必须回边再查一次。

    参数化把「显式 true」和「字段整个缺席」钉成**同一条路**：`retrievable` 是后加的字段，
    模型不输出它时必须逐位等于加它之前的行为。默认成 False 会在模型不配合时静默改掉
    循环 —— 那不是「省了一轮」，是换了套逻辑，而且没人会收到告警。
    """
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "罚款"})]),
        _reflection(False, "缺记分条款", retrievable=retrievable),
        _assistant(calls=[("c2", SEARCH_LAW_NAME, {"query": "记分"})]),
        _reflection(True),
    ]
    # max_steps 显式给 3：这里要的是**跑满两轮且第二轮审核真的调了模型**。
    # 用默认的 2 的话，第二轮末尾预算已尽，reflect 会短路不调模型 ——
    # 那样就分不清「回边生效了」和「只是没轮到短路」，测不到想测的东西。
    state, llm, retriever, _ = run("超员怎么罚", script, parents, generator, max_steps=3)

    assert len(retriever.queries) == 2
    assert state["steps"] == 2
    assert len(llm.calls) == 4


def test_parse_reflection_的三种输入():
    """解析层的默认值单独钉一遍 —— 图跑一整套代价太高，而这里是最容易被改错的一行。"""
    assert A._parse_reflection('{"sufficient": false, "retrievable": false}')["retrievable"] is False
    assert A._parse_reflection('{"sufficient": false, "retrievable": true}')["retrievable"] is True
    assert A._parse_reflection('{"sufficient": false}')["retrievable"] is True
    # 解析不出来 → 按已足够处理（停止），此时也谈不上回边
    assert A._parse_reflection("我不知道")["sufficient"] is True


def test_审核提示词带着库的边界(parents, generator):
    """审核要分清「这轮没检出来」和「库里根本没有」，前提是它知道库里有哪几部法。

    名单纯从 `parents` 现取，不写死一份 —— 这条同时钉住「新增法规时不用记得来改提示词」。
    """
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "q"})]),
        _reflection(True),
    ]
    state, llm, *_ = run("问题", script, parents, generator)
    system = llm.calls[1]["messages"][0]["content"]      # 第 2 次调用是审核

    for name in {parent.law_name for parent in parents.values()}:
        assert name in system, f"库内法规 {name} 没进审核提示词"
    assert "retrievable" in system


def test_轨迹区分库外收尾与预算耗尽(parents, generator):
    """两种收尾成因必须分得开：一个是「再检也没用」，一个是「没轮次了」。"""
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "q"})]),
        _reflection(False, "缺《治安管理处罚法》", retrievable=False),
    ]
    state, *_ = run("问题", script, parents, generator)     # 默认 max_steps 没花完
    trace = A.render_trace(state)

    assert "缺口在库外" in trace
    assert "已达最大轮数" not in trace


def test_默认轮数上限是2():
    """3 → 2 是实测定的，不是拍的：100 题多跳集上第 3 轮只多 2/200 条命中，
    却多花 35 次检索 + 35 次规划 + 35 次审核。这条钉住的是「别把它悄悄改回 3」。

    只断言 dataclass 默认值，不看 `agent_config()`：后者读环境变量，
    本机配了 AGENT_MAX_STEPS 就该听本机的，那不是这条测试该管的事。
    """
    assert config.AgentConfig().max_steps == 2


# ================================================================== 14. 提示词措辞
def test_检索工具的措辞不许和第一轮用原话那条打架():
    """第 1 轮照搬率曾经只有 **2/100**：系统提示词写着「不要改写」，而工具 schema 在
    模型真正填参数的那一个字段上写着「尽量靠近法条用语」、在外层描述里写着「换用不同
    关键词」—— 位置更靠近动作，schema 赢了。

    钉的是**两处口径一致**，不是某一句原文：谁再往参数描述里加回改写引导，
    那个 2/100 就会跟着回来。
    """
    fn = SEARCH_LAW_TOOL["function"]
    query_desc = fn["parameters"]["properties"]["query"]["description"]

    assert "原话" in query_desc, "参数描述得说清第一轮用原话"
    assert "法条用语" not in query_desc, "这里回过一次「尽量靠近法条用语」，与系统提示词正面冲突"
    assert "换用不同关键词" not in fn["description"]


def test_审核提示词让它拿上一轮的要过没当证据(parents, generator):
    """`retrievable` 光有字段不够：审核每轮从零重判，手上没有「上轮补上了没」这个依据，
    于是实测 45 对相邻的「不够」判定里，`missing` 相似度中位 0.82、24 对在 0.8 以上。

    对话历史里那条「[检索审核] 还缺：」是它自己写的，提示词必须点明去看它 ——
    连着两轮要不到，才是「库外」最直接的证据。引用要逐字对上 `reflect_node` 注入的写法，
    否则审核在历史里认不出自己那条。
    """
    script = [
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "q"})]),
        _reflection(True),
    ]
    _, llm, *_ = run("问题", script, parents, generator)
    system = llm.calls[1]["messages"][0]["content"]      # 第 2 次调用是审核

    assert "[检索审核]" in system
    assert "上一轮" in system


# ================================================================== 15. 审核单独挂模型
def test_审核模型不配时逐字段回退(monkeypatch):
    """不写 `AGENT_REFLECT_*` = 与单模型时**逐位相同**。

    这是这次改动的安全绳：多出来的只是「一个配置相同的客户端」，不是一套新行为。
    """
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TEMPERATURE",
                "AGENT_REFLECT_BASE_URL", "AGENT_REFLECT_API_KEY", "AGENT_REFLECT_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://base.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "big-model")

    assert config.reflect_llm_config() == config.llm_config()


def test_审核模型只写模型名时只覆盖那一项(monkeypatch):
    """**逐字段**回退，不是「有一项没配就整体回退」。

    同一家只想换个免费额度没用完的模型时（百炼的额度按模型算），写一项就够，
    不必把 key 和 base_url 再抄一遍 —— 抄一遍就迟早抄不一致。

    **没设的那两项必须先 delenv**：`config` 在 import 时把本机 `.env` 读进了
    `os.environ`，不清掉的话「沿用」断言的其实是**本机 .env 的值**，装了审核模型
    的机器上这条会红（这条测试第一次就是这么做出来的）。
    """
    monkeypatch.delenv("AGENT_REFLECT_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_REFLECT_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "qwen3-max")
    monkeypatch.setenv("AGENT_REFLECT_MODEL", "glm-4.7-flash")

    cfg = config.reflect_llm_config()
    assert cfg.model == "glm-4.7-flash"                   # 写了的覆盖
    assert cfg.base_url == "https://dashscope.example/v1"  # 没写的沿用
    assert cfg.api_key == "k"


def test_审核真的走了另一个客户端(parents, generator):
    """规划轮与审核**确实是两个客户端**，不是只多了一个没人调用的属性。"""
    reviewer = ScriptedLLM([_reflection(True)])
    state, planner, *_ = run(
        "问题",
        [_assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "q"})])],
        parents, generator, reflect_llm=reviewer,
    )

    assert len(planner.calls) == 1, "规划轮只该被问一次"
    assert len(reviewer.calls) == 1, "审核走的是另一个客户端"
    assert "审核" in reviewer.calls[0]["messages"][0]["content"]
    assert state["reflections"][0]["sufficient"] is True


# ================================================================== 16. top_k 同源
def test_单题自带的_top_k_管到收尾(parents, generator):
    """单题自带的 `top_k` 必须一路管到收尾 —— 两个节点同源。

    `finalize_node` 曾经读闭包（runner 的条数），`tools_node` 读 state：调用方传进来的
    `Question` 自带 `top_k` 时，循环按查询自己的条数召回、收尾却按 runner 的条数合并 ——
    而收尾这个还决定最终证据条数，正是最不该错的地方。那个闭包参数现在删掉了，
    两边都读 `state["top_k"]`、兜底都读 config（见 `_make_finalize_node`）。

    两个真实调用点（CLI、multihop）都传字符串，所以这条是**潜伏**的，不是活的；
    但 `invoke()` 是公开入口，`Question(text=..., top_k=N)` 就触发。

    **问题里不能带条号**：那会被 `classify_intent` 判成条文定位，走零检索的
    `get_article` 路径，证据恒为 1 条，截断根本没机会发生（这条测试第一版就是这么
    写错的 —— 顺手拿了 `ARTICLE_Q`）。
    """
    llm = ScriptedLLM([
        _assistant(calls=[("c1", SEARCH_LAW_NAME, {"query": "q"})]),
        _reflection(True),
    ])
    # 候选(4) 必须比最终证据(2) 多 —— 不截断就看不出 max_evidence 取的是哪一份
    retriever = FakeRetriever(parents, hits=4)
    rag = LegalRAG(retriever, generator, top_k=6)   # runner 的条数；单题那份要盖过它
    runner = A.AgentRunner(rag, llm=llm, cfg=config.AgentConfig(max_steps=2))

    state = runner.invoke(Question(text="醉驾怎么处罚", top_k=2))   # ← 单题覆盖

    assert state["top_k"] == 2
    assert len(state["answer"].retrieval.articles) == 2, "收尾该跟着单题的 top_k 走，不是 runner 的"
