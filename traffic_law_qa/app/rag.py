from __future__ import annotations

from .. import config
from ..contracts.answer import Answer, Question
from ..contracts.disk import ParentChunk
from ..contracts.reports import CorpusStats, channel_state
from ..contracts.retrieval import MaterialPassage, RetrievalResult, WebFinding
from ..generation.generator import AnswerGenerator
from ..ports import RagService
from ..search.retriever import HybridRetriever


class LegalRAG(RagService):

    input_desc = "问题 (str)"
    output_desc = "RetrievalResult | Answer"

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator | None = None,
        *,
        top_k: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator or AnswerGenerator()
        self.top_k = top_k or config.retrieve_config().top_k

    @classmethod
    def load(
        cls,
        *,
        with_vector: bool = True,
        generator: AnswerGenerator | None = None,
        top_k: int | None = None,
        retriever: HybridRetriever | None = None,
    ) -> "LegalRAG":
        from ..container import build_generator, build_retriever

        retriever = retriever or build_retriever(with_vector=with_vector)
        return cls(retriever, generator or build_generator(), top_k=top_k)

    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        channel_debug: bool = False,
        law_filter: tuple[str, ...] = (),
    ) -> RetrievalResult:
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
        query = question if isinstance(question, Question) else Question(text=question)
        return self.answer(query, self.search(query.text, top_k or query.top_k, channel_debug=channel_debug))

    def answer(
        self,
        question: str | Question,
        retrieval: RetrievalResult,
        *,
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ) -> Answer:
        query = question if isinstance(question, Question) else Question(text=question)
        return self._generator.generate(
            query, retrieval, timeliness=timeliness, materials=materials
        )

    def stream(self, question: str | Question, retrieval: RetrievalResult):
        query = question if isinstance(question, Question) else Question(text=question)
        return self._generator.stream(query, retrieval)

    def expand(self, text: str) -> str:
        return self._retriever.expand(text)

    def warm(self, *, probe: str = "预热") -> float | None:
        return self._retriever.warm(probe=probe)

    def stats(self) -> CorpusStats:
        return self._retriever.stats()

    @property
    def model_name(self) -> str:
        return self._generator.model_name

    @property
    def llm_ready(self) -> bool:
        return bool(getattr(self._generator, "available", False))

    @property
    def parents(self) -> dict[str, ParentChunk]:
        return self._retriever.parents

    def describe(self) -> str:
        stats = self.stats()
        return (
            f"RAG 工具：{stats.articles} 条法条 / {stats.chunks} 个子块 | "
            f"稠密通道 {channel_state(stats.dense)} | 集合 {stats.collection} | "
            f"默认 top_k={self.top_k} | LLM {self.model_name}"
        )


__all__ = ["LegalRAG"]
