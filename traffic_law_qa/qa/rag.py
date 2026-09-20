"""RAG 工具门面：把「检索」与「生成」包装成一个可被管道或外部直接调用的工具。

    LegalRAG.search : str → RetrievalResult          （只检索，不花 LLM 的钱）
    LegalRAG.ask    : str → Answer                   （检索 + 生成，带引用）
    LegalRAG.answer : (问题, 检索结果) → Answer        （只生成，不重检）
    LegalRAG.stream : (问题, 检索结果) → 迭代器        （逐块产出，SSE 用）

这一层是管道里唯一对外的"RAG 工具"，上层（pipeline / FastAPI / Agent）只依赖它，
不关心底层是 BM25 还是向量、是 Chroma 还是别的库。

**这句话以前是假的，现在是结构性的**：`HybridRetriever` 只在本模块被 import
（`tests/test_rag_boundary.py` 用 AST 盯着这条），它的 `store` / `chunks` /
`rewriter` 都是私有属性，外面伸手进来会立刻 `AttributeError`。
所以上层需要什么能力，就得在这里变成一个方法 —— `expand` 与 `stats` 就是这么来的：
它们从前分别是 `agent_tools.expand_query` 里的 `getattr(retriever, "rewriter")`
与两处各写一遍的 try/except。

同一条规矩也管 `_generator`。它从前是公开属性，于是 FastAPI 层与 Agent 收尾节点
各自伸手（`rag.generator.generate(...)`、`rag.generator.cfg.model`，7 处）。
那也不是随手乱写，是**门面缺口**：`stream` 根本没有，`available` 与模型名也拿不到，
api.py 还为了透传 `channel_debug` 不得不绕过 `ask()`。缺口补上之后，那 7 处都改走门面。

**换掉检索或生成实现时，要动的只有这一个文件。**
"""

from __future__ import annotations

from .. import config
from ..contracts import Answer, CorpusStats, ParentChunk, Question, RetrievalResult
from .generator import AnswerGenerator
from .retriever import HybridRetriever


class LegalRAG:
    """交通法规 RAG 工具。

    Input : 自然语言问题 (str)
    Output: RetrievalResult（检索）/ Answer（答问）
    """

    input_desc = "问题 (str)"
    output_desc = "RetrievalResult | Answer"

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator | None = None,
        *,
        top_k: int | None = None,
    ) -> None:
        # 零件私有、配置公开：`top_k` 留在公开面上，与 `HybridRetriever` 保留
        # `parents` / `embedder` / `cfg` 是同一个取舍。构造参数名仍是 `retriever=` /
        # `generator=` —— 那是**注入点**（测试要塞替身），与「构造之后谁能看见」是两回事。
        self._retriever = retriever
        self._generator = generator or AnswerGenerator()
        self.top_k = top_k or config.retrieve_config().top_k

    # -------------------------------------------------------------- 装配
    @classmethod
    def load(
        cls,
        *,
        with_vector: bool = True,
        generator: AnswerGenerator | None = None,
        top_k: int | None = None,
    ) -> "LegalRAG":
        """从磁盘上已建好的索引装配工具。"""
        return cls(
            HybridRetriever.load(with_vector=with_vector), generator=generator, top_k=top_k
        )

    # -------------------------------------------------------------- 工具接口
    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        channel_debug: bool = False,
        law_filter: tuple[str, ...] = (),
    ) -> RetrievalResult:
        """只做检索：返回召回的父块（整条法条）与排名信息。

        `law_filter` 是**只在这些 law_id 内检索**，空元组 = 全库。
        默认值必须是 `()` 而不是 `None` —— `MilvusStore.law_filter_expr`
        直接迭代入参，给 `None` 会当场 `TypeError`；而 `Query.law_filter`
        的默认值本来就是 `()`，两边得对上。
        """
        return self._retriever.search(
            question,
            top_k=top_k or self.top_k,
            channel_debug=channel_debug,
            law_filter=law_filter,
        )

    def ask(
        self,
        question: str | Question,
        top_k: int | None = None,
        *,
        channel_debug: bool = False,
    ) -> Answer:
        """检索 + 生成：返回带 [依据N] 标注的答案。

        `channel_debug` 一路透到检索层。从前它没在这里，api.py 的 debug 模式
        只能自己 `search()` 之后再伸手调生成器 —— 一个参数逼出一条绕过门面的路。
        """
        query = question if isinstance(question, Question) else Question(text=question)
        return self.answer(query, self.search(query.text, top_k or query.top_k, channel_debug=channel_debug))

    def answer(self, question: str | Question, retrieval: RetrievalResult) -> Answer:
        """用**已经拿到的**检索结果生成答案，不重新检索。

        给「先推依据、再逐块推 token」的调用方用（FastAPI 的 SSE），以及 Agent
        收尾节点那种检索轮与生成轮分开的流程：它们手上的 retrieval 是刚花过钱、
        带过 `channel_debug` 的，再走一次 `ask()` 等于白检一遍。
        """
        query = question if isinstance(question, Question) else Question(text=question)
        return self._generator.generate(query, retrieval)

    def stream(self, question: str | Question, retrieval: RetrievalResult):
        """流式生成：产出 `(kind, payload)` 二元组，kind 是 "delta" / "usage"。

        门面只转发、不解释 payload —— 事件名（`token` / `done`）与帧格式是 HTTP 层
        的事，见 `server.py`。
        """
        query = question if isinstance(question, Question) else Question(text=question)
        return self._generator.stream(query, retrieval)

    def expand(self, text: str) -> str:
        """口语对齐后的检索词 —— 摘要在原文里开窗口时按它定位。"""
        return self._retriever.expand(text)

    def warm(self, *, probe: str = "预热") -> float | None:
        """预热稠密通道，返回耗时毫秒；没得预热或失败时返回 None。"""
        return self._retriever.warm(probe=probe)

    def stats(self) -> CorpusStats:
        """语料规模与通道状态。连不上 Milvus 时 `dense` 是 None（未知），不是异常。"""
        return self._retriever.stats()

    @property
    def model_name(self) -> str:
        """生成模型名（未配置 LLM 时是空串）。"""
        return self._generator.cfg.model

    @property
    def llm_ready(self) -> bool:
        """生成层是否可用。为 False 时答问会拒答（并说明原因），检索不受影响。"""
        return bool(getattr(self._generator, "available", False))

    @property
    def parents(self) -> dict[str, ParentChunk]:
        """全部父块（整条法条）的只读视图，键是 parent_id。

        名字叫 `parents` 而不是 `articles`：`RetrievalResult.articles` 是另一个
        类型（`RetrievedArticle` 带分数），同名会让「回灌用的语料」与「一次检索的
        结果」看起来是一回事。消费者那边的参数本来也都叫 `parents`。
        """
        return self._retriever.parents

    def describe(self) -> str:
        stats = self.stats()
        if stats.dense is None:
            vector_state = "未知（Milvus 未连接）"
        else:
            vector_state = "已启用" if stats.dense else "未启用（仅 BM25）"
        return (
            f"RAG 工具：{stats.articles} 条法条 / {stats.chunks} 个子块 | "
            f"稠密通道 {vector_state} | 集合 {stats.collection} | "
            f"默认 top_k={self.top_k} | LLM {self.model_name}"
        )


__all__ = ["LegalRAG"]
