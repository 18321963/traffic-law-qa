"""RAG 工具门面：把「检索」与「生成」包装成一个可被管道或外部直接调用的工具。

    LegalRAG.search : str → RetrievalResult   （只检索，不花 LLM 的钱）
    LegalRAG.ask    : str → Answer            （检索 + 生成，带引用）

这一层是管道里唯一对外的"RAG 工具"，上层（pipeline / FastAPI / Agent）只依赖它，
不关心底层是 BM25 还是向量、是 Chroma 还是别的库。
"""

from __future__ import annotations

from . import config
from .contracts import Answer, Question, RetrievalResult
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
        self.retriever = retriever
        self.generator = generator or AnswerGenerator()
        self.top_k = top_k or config.retrieve_config().top_k

    # -------------------------------------------------------------- 装配
    @classmethod
    def load(cls, *, with_vector: bool = True, top_k: int | None = None) -> "LegalRAG":
        """从磁盘上已建好的索引装配工具。"""
        return cls(HybridRetriever.load(with_vector=with_vector), top_k=top_k)

    # -------------------------------------------------------------- 工具接口
    def search(self, question: str, top_k: int | None = None) -> RetrievalResult:
        """只做检索：返回召回的父块（整条法条）与排名信息。"""
        return self.retriever.search(question, top_k=top_k or self.top_k)

    def ask(self, question: str | Question, top_k: int | None = None) -> Answer:
        """检索 + 生成：返回带 [依据N] 标注的答案。"""
        query = question if isinstance(question, Question) else Question(text=question)
        retrieval = self.retriever.search(
            query.text, top_k=top_k or query.top_k or self.top_k
        )
        return self.generator.generate(query, retrieval)

    def context(self, retrieval: RetrievalResult) -> str:
        """把召回结果拼成一段可直接塞进 prompt 的上下文（给外部编排使用）。"""
        blocks = []
        for index, hit in enumerate(retrieval.articles, start=1):
            blocks.append(f"【依据{index}】{hit.citation}\n{hit.article.text}")
        return "\n\n".join(blocks)

    def describe(self) -> str:
        vector_state = "已启用" if self.retriever.vector is not None else "未启用（仅 BM25）"
        return (
            f"RAG 工具：{len(self.retriever.parents)} 条法条 / {len(self.retriever.chunks)} 个子块 | "
            f"向量通道 {vector_state} | 默认 top_k={self.top_k} | LLM {self.generator.cfg.model}"
        )


__all__ = ["LegalRAG"]
