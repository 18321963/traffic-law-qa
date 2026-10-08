from __future__ import annotations

import ast
from pathlib import Path

import rag_contracts
import rag_service
from rag_contracts import config
from rag_contracts.domain.answer import Question
from rag_contracts.domain.disk import Chunk, ChunkSet, ParentChunk
from rag_contracts.domain.retrieval import Query, RetrievalResult, RetrievedArticle
from rag_contracts.ports import LLM, Embedder, RagService, Reranker, VectorStore
from rag_service.container import build_retriever
from rag_service.query.generator import AnswerGenerator
from rag_service.query.rag import LegalRAG

PACKAGE = Path(rag_service.__file__).resolve().parent
SOURCES = sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)
assert len(SOURCES) > 20, f"扫描根不成立：{PACKAGE} 下只有 {len(SOURCES)} 个 .py，端口门会报成「全都没实现」"

CONTRACTS_PACKAGE = Path(rag_contracts.__file__).resolve().parent
CONTRACT_SOURCES = sorted(
    path for path in CONTRACTS_PACKAGE.rglob("*.py") if "__pycache__" not in path.parts
)
assert len(CONTRACT_SOURCES) > 10, (
    f"扫描根不成立：{CONTRACTS_PACKAGE} 下只有 {len(CONTRACT_SOURCES)} 个 .py，端口门会报成「全都没实现」"
)

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

    @property
    def fingerprint(self) -> str:
        return ""

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
        return (
            {"role": "assistant", "content": f"【依据1】{ARTICLE_TEXT}"},
            {"total_tokens": 3},
            "stop",
        )

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
    for path in list(SOURCES) + list(CONTRACT_SOURCES):
        if path.name == "ports.py":
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
    assert LegalRAG.__abstractmethods__ == frozenset(), (
        "端口加了成员、实现没跟上（issubclass 会绿，一构造就 TypeError）："
        f"{sorted(LegalRAG.__abstractmethods__)}"
    )


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


def _legal_rag() -> LegalRAG:
    retriever = build_retriever(
        with_vector=True,
        store=_InMemoryStore([("c1", 0.9)]),
        embedder=_EchoEmbedder(),
        reranker=_OffReranker(),
        chunk_set=_chunk_set(),
    )
    generator = AnswerGenerator(
        config.LLMConfig(base_url="http://stub", api_key="stub", model="内存模型"),
        llm=_ScriptedLLM(),
    )
    return LegalRAG(retriever, generator)


def test_the_tool_face_needs_nothing_but_the_port() -> None:
    rag = _legal_rag()

    laws = rag.laws()
    assert [(law.law_name, law.articles) for law in laws] == [(LAW_NAME, 1)]

    hit, error = rag.get_article(ARTICLE_NO, LAW_NAME)
    assert error == "" and hit is not None
    assert [item.article.text for item in hit.articles] == [ARTICLE_TEXT]

    missing, why = rag.get_article("第九十九条", LAW_NAME)
    assert missing is None and "没有第 99 条" in why
    assert rag.get_article(ARTICLE_NO, "不存在的法")[0] is None

    assert rag.materials(()) == ()
    passages, note = rag.search_materials(QUESTION, ())
    assert passages == () and "没有上传材料" in note


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
