from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Iterator

from .contracts.answer import Answer, Question
from .contracts.disk import IndexStats, ParentChunk
from .contracts.reports import CorpusStats, PipelineReport
from .contracts.retrieval import MaterialPassage, RetrievalResult, WebFinding

__all__ = [
    "Embedder",
    "IndexBuilder",
    "IndexStatus",
    "LLM",
    "RagService",
    "Reranker",
    "VectorStore",
]


class Embedder(ABC):

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @property
    @abstractmethod
    def unavailable_reason(self) -> str: ...

    @property
    @abstractmethod
    def model_label(self) -> str: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class Reranker(ABC):

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @property
    @abstractmethod
    def unavailable_reason(self) -> str: ...

    @property
    @abstractmethod
    def top_n(self) -> int: ...

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class LLM(ABC):

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        name: str = "llm.chat",
    ) -> tuple[dict, dict]: ...

    @abstractmethod
    def stream(
        self, messages: list[dict], *, temperature: float | None = None
    ) -> Iterator[tuple[str, Any]]: ...


class VectorStore(ABC):

    collection: str
    uri: str

    @abstractmethod
    def ping(self) -> str: ...

    @abstractmethod
    def has_collection(self) -> bool: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def has_dense_field(self) -> bool: ...

    @abstractmethod
    def recreate(self, *, dim: int | None = None) -> dict: ...

    @abstractmethod
    def insert(self, rows: list[dict], *, batch_size: int = 200) -> int: ...

    @abstractmethod
    def hybrid_search(
        self,
        *,
        query_text: str,
        dense_vector: list[float] | None,
        limit: int,
        candidates: int,
        filter_expr: str | None = None,
        rrf_k: int = 60,
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def dense_search(
        self, vector: list[float], *, limit: int, filter_expr: str | None = None
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def sparse_search(
        self, query_text: str, *, limit: int, filter_expr: str | None = None
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def law_filter(self, law_ids: Iterable[str]) -> str | None: ...


class RagService(ABC):

    parents: dict[str, ParentChunk]
    top_k: int

    @property
    @abstractmethod
    def llm_ready(self) -> bool: ...

    @abstractmethod
    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        channel_debug: bool = False,
        law_filter: tuple[str, ...] = (),
    ) -> RetrievalResult: ...

    @abstractmethod
    def answer(
        self,
        question: Question,
        retrieval: RetrievalResult,
        *,
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ) -> Answer: ...

    @abstractmethod
    def expand(self, text: str) -> str: ...

    @abstractmethod
    def stats(self) -> CorpusStats: ...


class IndexStatus(ABC):

    uri: str

    @abstractmethod
    def ping(self) -> str: ...

    @abstractmethod
    def has_collection(self) -> bool: ...

    @abstractmethod
    def rows(self) -> int: ...

    @abstractmethod
    def snapshot(self) -> IndexStats | None: ...

    @abstractmethod
    def notes(self) -> list[str]: ...

    @abstractmethod
    def counts(self) -> tuple[int, int]: ...

    @abstractmethod
    def desired_embedding_label(self) -> str: ...

    @abstractmethod
    def stale_reason(self, *, want_dense: bool) -> str | None: ...


class IndexBuilder(ABC):

    @abstractmethod
    def build(self, *, force: bool = False, with_vector: bool = True) -> PipelineReport: ...
