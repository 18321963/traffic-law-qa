from __future__ import annotations

from collections.abc import Sequence

from rag_contracts import config
from rag_contracts.domain.answer import Answer, Question
from rag_contracts.domain.disk import ParentChunk
from rag_contracts.domain.laws import LawInfo, laws_of
from rag_contracts.domain.reports import CorpusStats, channel_state
from rag_contracts.domain.retrieval import MaterialPassage, RetrievalResult, WebFinding
from rag_contracts.ports import RagService

from .articles import build_article_index, lookup_article
from .generator import AnswerGenerator
from .materials import load_materials
from .materials import search_materials as score_materials
from .retriever import HybridRetriever


class LegalRAG(RagService):

    input_desc = "问题 (str)"
    output_desc = "RetrievalResult | Answer"

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        *,
        top_k: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
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
        candidates: int | None = None,
    ) -> RetrievalResult:
        return self._retriever.search(
            question,
            top_k=top_k or self.top_k,
            candidates=candidates,
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

    def stream(
        self,
        question: str | Question,
        retrieval: RetrievalResult,
        *,
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ):
        query = question if isinstance(question, Question) else Question(text=question)
        return self._generator.stream(query, retrieval, timeliness=timeliness, materials=materials)

    def expand(self, text: str) -> str:
        return self._retriever.expand(text)

    def warm(self, *, probe: str = "预热") -> float | None:
        return self._retriever.warm(probe=probe)

    def stats(self) -> CorpusStats:
        return self._retriever.stats()

    def laws(self) -> tuple[LawInfo, ...]:
        return laws_of(self.parents.values())

    def get_article(
        self, article_no: str, law_name: str | None = None
    ) -> tuple[RetrievalResult | None, str]:
        return lookup_article(
            article_no,
            law_name,
            parents=self.parents,
            index=build_article_index(self.parents),
        )

    def materials(self, doc_ids: Sequence[str]) -> tuple[MaterialPassage, ...]:
        return load_materials(doc_ids)

    def search_materials(
        self, query: str, doc_ids: Sequence[str], *, top_k: int = 5
    ) -> tuple[tuple[MaterialPassage, ...], str]:
        passages, text = score_materials(query, self.materials(doc_ids), top_k=top_k)
        return tuple(passages), text

    @property
    def model_name(self) -> str:
        return self._generator.model_name

    @property
    def llm_ready(self) -> bool:
        return bool(getattr(self._generator, "available", False))

    def milvus_live(self) -> bool:
        return self._retriever.milvus_live()

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
