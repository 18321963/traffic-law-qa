from __future__ import annotations

import ast
from pathlib import Path

from traffic_law_qa import config
from traffic_law_qa.app.rag import LegalRAG
from traffic_law_qa.container import build_retriever
from traffic_law_qa.contracts.answer import Question
from traffic_law_qa.contracts.disk import Chunk, ChunkSet, ParentChunk
from traffic_law_qa.contracts.retrieval import Query, RetrievalResult, RetrievedArticle
from traffic_law_qa.generation.generator import AnswerGenerator
from traffic_law_qa.ports import LLM, Embedder, RagService, Reranker, VectorStore

PACKAGE = Path(__file__).resolve().parent.parent / "traffic_law_qa"

PORTS = (
    "Embedder",
    "IndexBuilder",
    "IndexStatus",
    "LLM",
    "RagService",
    "Reranker",
    "VectorStore",
)

LAW_NAME = "中华人民共和国道路交通安全法"
ARTICLE_NO = "第九十一条"
ARTICLE_TEXT = "饮酒后驾驶营运机动车的，处十五日拘留，并处五千元罚款。"
QUESTION = "饮酒后驾驶营运机动车怎么处罚"


class _EchoEmbedder(Embedder):

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str:
        return ""

    @property
    def model_label(self) -> str:
        return "内存嵌入"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


class _OffReranker(Reranker):

    @property
    def available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return "内存演练：不重排"

    @property
    def top_n(self) -> int:
        return 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.0 for _ in texts]


class _ScriptedLLM(LLM):

    @property
    def available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "内存模型"

    def chat(self, messages, *, tools=None, temperature=None, name="llm.chat"):
        return {"role": "assistant", "content": f"【依据1】{ARTICLE_TEXT}"}, {"total_tokens": 3}

    def stream(self, messages, *, temperature=None):
        yield "delta", ARTICLE_TEXT


class _InMemoryStore(VectorStore):

    collection = "内存集合"
    uri = "memory://演练"

    def __init__(self, hits) -> None:
        self._hits = list(hits)
        self.searches: list[tuple[str, str | None]] = []

    def ping(self) -> str:
        return "memory-1.0"

    def has_collection(self) -> bool:
        return True

    def count(self) -> int:
        return len(self._hits)

    def has_dense_field(self) -> bool:
        return False

    def recreate(self, *, dim: int | None = None) -> dict:
        return {"collection": self.collection, "dim": dim}

    def insert(self, rows: list[dict], *, batch_size: int = 200) -> int:
        self._hits.extend((row["chunk_id"], 0.0) for row in rows)
        return len(rows)

    def hybrid_search(
        self,
        *,
        query_text: str,
        dense_vector: list[float] | None,
        limit: int,
        candidates: int,
        filter_expr: str | None = None,
        rrf_k: int = 60,
    ) -> list[tuple[str, float]]:
        self.searches.append((query_text, filter_expr))
        return self._hits[:limit]

    def dense_search(
        self, vector: list[float], *, limit: int, filter_expr: str | None = None
    ) -> list[tuple[str, float]]:
        self.searches.append(("dense", filter_expr))
        return self._hits[:limit]

    def sparse_search(
        self, query_text: str, *, limit: int, filter_expr: str | None = None
    ) -> list[tuple[str, float]]:
        self.searches.append((query_text, filter_expr))
        return self._hits[:limit]

    def law_filter(self, law_ids) -> str | None:
        values = [law_id for law_id in law_ids if law_id]
        return None if not values else f"law_id in [{', '.join(values)}]"


def _chunk_set() -> ChunkSet:
    parent = ParentChunk(
        parent_id=f"{LAW_NAME}#{ARTICLE_NO}",
        law_id="road_traffic_safety",
        law_name=LAW_NAME,
        version="2021",
        citation=f"《{LAW_NAME}》",
        article_no=ARTICLE_NO,
        article_index=91,
        chapter="第七章 法律责任",
        section=None,
        text=ARTICLE_TEXT,
    )
    chunk = Chunk(
        chunk_id="c1",
        parent_id=parent.parent_id,
        law_id=parent.law_id,
        law_name=parent.law_name,
        version=parent.version,
        citation=parent.citation,
        article_no=parent.article_no,
        article_index=parent.article_index,
        part_index=0,
        part_total=1,
        text=ARTICLE_TEXT,
        embed_text=ARTICLE_TEXT,
    )
    return ChunkSet(parents=(parent,), chunks=(chunk,))


def test_every_port_has_an_implementation() -> None:
    found: dict[str, list[str]] = {name: [] for name in PORTS}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "ports.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in found:
                    found[base.id].append(
                        f"{path.relative_to(PACKAGE.parent)}:{node.lineno} {node.name}"
                    )
    missing = sorted(name for name, homes in found.items() if not homes)
    assert not missing, f"这些端口只有签名、没有实现（端口是承诺，不是摆设）：{missing}"


def test_legalrag_declares_the_rag_service_port() -> None:
    assert issubclass(LegalRAG, RagService)


def test_search_runs_on_an_injected_in_memory_store() -> None:
    store = _InMemoryStore([("c1", 0.9)])
    retriever = build_retriever(
        with_vector=True,
        store=store,
        embedder=_EchoEmbedder(),
        reranker=_OffReranker(),
        chunk_set=_chunk_set(),
    )

    result = retriever.retrieve(Query(text=QUESTION, top_k=3, candidates=5))

    assert [hit.article.article_no for hit in result.articles] == [ARTICLE_NO]
    assert [hit.article.text for hit in result.articles] == [ARTICLE_TEXT]
    assert store.searches == [(QUESTION, None)], "检索没走到注入的存储上，或问句原话没带上"


def test_the_answer_comes_from_the_injected_llm() -> None:
    parent = _chunk_set().parents[0]
    retrieval = RetrievalResult(
        query=QUESTION,
        articles=(RetrievedArticle(article=parent, score=0.9),),
        used_vector=False,
        used_bm25=True,
        elapsed_ms=1.0,
    )
    generator = AnswerGenerator(
        config.LLMConfig(base_url="http://stub", api_key="stub", model="内存模型"),
        llm=_ScriptedLLM(),
    )

    answer = generator.generate(Question(text=QUESTION), retrieval)

    assert answer.text == f"【依据1】{ARTICLE_TEXT}"
    assert answer.model == "内存模型"
    assert answer.usage == {"total_tokens": 3}
