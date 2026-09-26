from __future__ import annotations

from typing import Any

from .contracts.reports import ReadyState

__all__ = [
    "boot_agent_runner",
    "build_agent_runner",
    "build_chunk_set",
    "build_embedder",
    "build_generator",
    "build_indexer",
    "build_llm",
    "build_pipeline",
    "build_rag",
    "build_reranker",
    "build_retriever",
    "build_status",
    "build_store",
    "readiness",
]


def build_store(*, verbose: bool = True):
    from .infra.milvus import MilvusStore

    return MilvusStore(verbose=verbose)


def build_embedder():
    from .infra.embedding import LocalEmbedder

    return LocalEmbedder()


def build_reranker():
    from .infra.reranking import LocalReranker

    return LocalReranker()


def build_llm(
    cfg=None,
    *,
    retries: int = 2,
    timeout: float = 30.0,
    max_tokens: int = 1024,
    observer: Any = None,
    client: Any = None,
):
    from .infra.llm import OpenAILLM

    return OpenAILLM(
        cfg,
        retries=retries,
        timeout=timeout,
        max_tokens=max_tokens,
        observer=observer,
        client=client,
    )


def build_chunk_set():
    from .indexing.chunker import ChunkStage

    return ChunkStage(verbose=False).load()


def build_indexer(*, verbose: bool = True):
    from .indexing.indexer import Indexer

    return Indexer(verbose=verbose)


def build_status(*, store=None, indexer=None, embedder=None):
    from .indexing.status import CorpusStatus

    return CorpusStatus(
        store=store or build_store(verbose=False),
        indexer=indexer or build_indexer(verbose=False),
        embedder=embedder or build_embedder(),
    )


def build_retriever(
    *,
    with_vector: bool = True,
    store=None,
    embedder=None,
    reranker=None,
    chunk_set=None,
):
    from .search.retriever import HybridRetriever

    chunk_set = chunk_set or build_chunk_set()
    return HybridRetriever(
        store=store or build_store(verbose=False),
        parents=chunk_set.parent_map(),
        chunks={chunk.chunk_id: chunk for chunk in chunk_set.chunks},
        embedder=(embedder or build_embedder()) if with_vector else None,
        reranker=reranker or build_reranker(),
    )


def build_generator(cfg=None, *, llm=None):
    from .generation.generator import AnswerGenerator

    return AnswerGenerator(cfg, llm=llm)


def build_rag(
    *,
    with_vector: bool = True,
    top_k: int | None = None,
    generator=None,
    retriever=None,
):
    from .app.rag import LegalRAG

    return LegalRAG.load(
        with_vector=with_vector, top_k=top_k, generator=generator, retriever=retriever
    )


def build_pipeline(*, verbose: bool = True):
    from .indexing.build import RagPipeline

    return RagPipeline(verbose=verbose)


def build_agent_runner(rag, *, cfg=None, tracer=None):
    from .agents.graph import AgentRunner

    return AgentRunner.attach(rag, cfg=cfg, tracer=tracer)


def boot_agent_runner(
    *, with_vector: bool = True, top_k: int | None = None, cfg=None, tracer=None
):
    readiness(with_vector=with_vector)
    return build_agent_runner(
        build_rag(with_vector=with_vector, top_k=top_k), cfg=cfg, tracer=tracer
    )


def readiness(*, with_vector: bool = True, rebuild: bool = False) -> ReadyState:
    from .app.readiness import ensure_ready

    return ensure_ready(with_vector=with_vector, rebuild=rebuild)
