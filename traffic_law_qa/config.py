from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:

    def _load_dotenv(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return False


load_dotenv = _load_dotenv
load_dotenv(ROOT / ".env", override=False)

KB_DIR = ROOT / "法规知识库"
SOURCE_DIR = KB_DIR / "docx"
TEXT_DIR = KB_DIR / "text"
PARSED_DIR = KB_DIR / "parsed"
CHUNK_DIR = KB_DIR / "chunks"
INDEX_DIR = KB_DIR / "index"
PDF_DIR = KB_DIR / "pdf"

MANIFEST_PATH = PARSED_DIR / "manifest.json"
CHUNKS_PATH = CHUNK_DIR / "chunks.jsonl"
PARENTS_PATH = CHUNK_DIR / "parents.jsonl"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"

DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "documents.db"


def upload_dir(doc_id: str) -> Path:
    return UPLOAD_DIR / doc_id

EVAL_CORPUS_PATH = DATA_DIR / "eval_corpus.json"

EVAL_RETRIEVAL_PATH = DATA_DIR / "eval_retrieval.json"
EVAL_REFERENCE_PATH = DATA_DIR / "eval_reference.json"
EVAL_NOGOLD_PATH = DATA_DIR / "eval_nogold.json"
EVAL_MULTIHOP_PATH = DATA_DIR / "eval_multihop.json"

ALL_DIRS = (SOURCE_DIR, TEXT_DIR, PARSED_DIR, CHUNK_DIR, INDEX_DIR)

SOURCE_SUFFIXES = (".docx", ".md", ".txt")


def source_files(base: Path | None = None) -> list[Path]:
    root = base or SOURCE_DIR
    found = [p for suffix in SOURCE_SUFFIXES for p in root.glob(f"*{suffix}")]
    return sorted(found)


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2

    @property
    def ready(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class EmbedConfig:
    base_url: str
    api_key: str
    model: str
    dim: int | None = None
    batch: int = 10
    query_prefix: str = ""

    @property
    def ready(self) -> bool:
        return bool(self.api_key)


def llm_config() -> LLMConfig:
    return LLMConfig(
        base_url=_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=_env("LLM_API_KEY"),
        model=_env("LLM_MODEL", "deepseek-chat"),
        temperature=_env_float("LLM_TEMPERATURE", 0.2),
    )


def region_llm_config() -> LLMConfig:
    base = llm_config()
    return LLMConfig(
        base_url=_env("AGENT_REGION_BASE_URL") or base.base_url,
        api_key=_env("AGENT_REGION_API_KEY") or base.api_key,
        model=_env("AGENT_REGION_MODEL") or base.model,
        temperature=base.temperature,
    )


def review_llm_config() -> LLMConfig:
    base = llm_config()
    return LLMConfig(
        base_url=_env("AGENT_REVIEW_BASE_URL") or base.base_url,
        api_key=_env("AGENT_REVIEW_API_KEY") or base.api_key,
        model=_env("AGENT_REVIEW_MODEL") or base.model,
        temperature=base.temperature,
    )


def embed_config() -> EmbedConfig:
    base_url = _env("EMBED_BASE_URL") or _env("LLM_BASE_URL", "https://api.deepseek.com/v1")
    api_key = _env("EMBED_API_KEY") or _env("LLM_API_KEY")
    dim_raw = _env("EMBED_DIM")
    return EmbedConfig(
        base_url=base_url,
        api_key=api_key,
        model=_env("EMBED_MODEL", "text-embedding-v4"),
        dim=int(dim_raw) if dim_raw.isdigit() else None,
        batch=_env_int("EMBED_BATCH", 10),
        query_prefix=_env("EMBED_QUERY_PREFIX"),
    )


@dataclass(frozen=True)
class RetrieveConfig:
    top_k: int = 6
    candidates: int = 20
    rrf_k: int = 60
    law_hint_boost: float = 1.5


def retrieve_config() -> RetrieveConfig:
    return RetrieveConfig(
        top_k=_env_int("RAG_TOP_K", 6),
        candidates=_env_int("RAG_CANDIDATES", 20),
        rrf_k=_env_int("RAG_RRF_K", 60),
        law_hint_boost=_env_float("RAG_LAW_HINT_BOOST", 1.5),
    )


@dataclass(frozen=True)
class AgentConfig:

    max_steps: int = 3
    max_evidence: int = 0
    snippet_chars: int = 120
    article_chars: int = 400
    temperature: float = 0.0
    retries: int = 2
    review_min_score: float = 0.6
    """末端复核的拦截阈值。`< 0` 关掉复核节点、`0` 只打分不拦截、`> 0` 才拦（`score < 阈值`）。

    **0.6 是占位值，不是标定值** —— 「正确率 / 拒答率」两条曲线还没跑（要跑批，烧 LLM 额度）。
    今天能说的是它的量级：120 条真答案里引用编号个数 ≤3 的占 70%，也就是多数答案只需要
    3 条判据里错 1 条就会被拦下来。真跑完标定再回来改这个数。
    """


def agent_config() -> AgentConfig:
    return AgentConfig(
        max_steps=_env_int("AGENT_MAX_STEPS", 3),
        max_evidence=_env_int("AGENT_MAX_EVIDENCE", 0),
        snippet_chars=_env_int("AGENT_SNIPPET_CHARS", 120),
        article_chars=_env_int("AGENT_ARTICLE_CHARS", 400),
        temperature=_env_float("AGENT_TEMPERATURE", 0.0),
        retries=_env_int("AGENT_RETRIES", 2),
        review_min_score=_env_float("AGENT_REVIEW_MIN_SCORE", 0.6),
    )


@dataclass(frozen=True)
class BochaConfig:

    api_key: str
    base_url: str = "https://api.bochaai.com/v1"
    count: int = 5
    freshness: str = "noLimit"
    timeout: float = 15.0
    summary: bool = True

    @property
    def ready(self) -> bool:
        return bool(self.api_key)


def bocha_config() -> BochaConfig:
    return BochaConfig(
        api_key=_env("BOCHA_API_KEY"),
        base_url=_env("BOCHA_BASE_URL", "https://api.bochaai.com/v1"),
        count=_env_int("BOCHA_COUNT", 5),
        freshness=_env("BOCHA_FRESHNESS", "noLimit"),
        timeout=_env_float("BOCHA_TIMEOUT", 15.0),
    )


@dataclass(frozen=True)
class LangfuseConfig:

    public_key: str
    secret_key: str
    host: str = "https://cloud.langfuse.com"

    @property
    def ready(self) -> bool:
        return bool(self.public_key and self.secret_key)


def langfuse_config() -> LangfuseConfig:
    return LangfuseConfig(
        public_key=_env("LANGFUSE_PUBLIC_KEY"),
        secret_key=_env("LANGFUSE_SECRET_KEY"),
        host=_env("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


@dataclass(frozen=True)
class MilvusConfig:
    uri: str
    token: str
    collection: str
    analyzer_params: dict
    bm25_k1: float
    bm25_b: float


def milvus_config() -> MilvusConfig:
    tokenizer = _env("MILVUS_ANALYZER", "jieba")
    return MilvusConfig(
        uri=_env("MILVUS_URI", "http://localhost:19530"),
        token=_env("MILVUS_TOKEN", "root:Milvus"),
        collection=_env("MILVUS_COLLECTION", "traffic_law"),
        analyzer_params={"tokenizer": tokenizer} if tokenizer else {},
        bm25_k1=_env_float("MILVUS_BM25_K1", 1.2),
        bm25_b=_env_float("MILVUS_BM25_B", 0.75),
    )
