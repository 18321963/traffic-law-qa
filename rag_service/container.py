from __future__ import annotations

from rag_contracts.domain.reports import ReadyState

__all__ = [
    "build_chunk_set",
    "build_embedder",
    "build_generator",
    "build_indexer",
    "build_library",
    "build_pipeline",
    "build_rag",
    "build_reranker",
    "build_retriever",
    "build_status",
    "build_store",
    "readiness",
]


def build_store():
    from .adapters.milvus import MilvusStore

    return MilvusStore()


def build_embedder():
    from .adapters.embedding import LocalEmbedder

    return LocalEmbedder()


def build_reranker():
    from .adapters.reranking import LocalReranker

    return LocalReranker()


def build_chunk_set():
    from .indexing.chunker import ChunkStage

    return ChunkStage(verbose=False).load()


def build_indexer(*, verbose: bool = True, store=None, embedder=None):
    from .indexing.indexer import Indexer

    return Indexer(
        store=store or build_store(),
        embedder=embedder or build_embedder(),
        verbose=verbose,
    )


def build_library():
    from .indexing.parser import LawLibrary

    return LawLibrary()


def build_status(*, store=None, indexer=None, embedder=None, library=None):
    from .indexing.status import CorpusStatus

    return CorpusStatus(
        store=store or build_store(),
        indexer=indexer or build_indexer(verbose=False),
        embedder=embedder or build_embedder(),
        library=library or build_library(),
    )


def build_retriever(
    *,
    with_vector: bool = True,
    store=None,
    embedder=None,
    reranker=None,
    chunk_set=None,
):
    from .query.retriever import HybridRetriever

    chunk_set = chunk_set or build_chunk_set()
    return HybridRetriever(
        store=store or build_store(),
        parents=chunk_set.parent_map(),
        chunks={chunk.chunk_id: chunk for chunk in chunk_set.chunks},
        embedder=(embedder or build_embedder()) if with_vector else None,
        reranker=reranker or build_reranker(),
    )


def build_generator(cfg=None, *, llm=None):
    from .query.generator import AnswerGenerator

    return AnswerGenerator(cfg, llm=llm)


def build_rag(
    *,
    with_vector: bool = True,
    top_k: int | None = None,
    generator=None,
    retriever=None,
):
    from .query.rag import LegalRAG

    return LegalRAG.load(
        with_vector=with_vector, top_k=top_k, generator=generator, retriever=retriever
    )


def build_pipeline(*, verbose: bool = True, indexer=None, library=None):
    from .indexing.build import RagPipeline

    return RagPipeline(
        indexer=indexer or build_indexer(verbose=verbose),
        library=library or build_library(),
        verbose=verbose,
    )


def readiness(*, with_vector: bool = True, rebuild: bool = False) -> ReadyState:
    from .indexing.readiness import ensure_ready

    return ensure_ready(with_vector=with_vector, rebuild=rebuild)
