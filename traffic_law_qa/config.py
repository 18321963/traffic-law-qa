"""集中式配置：目录布局、模型端点、检索参数。

所有可变项都从环境变量读取（写在项目根目录的 .env 里），代码中不出现任何密钥。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover

    def _load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """python-dotenv 缺失时的空实现，直接读进程环境变量。"""
        del args, kwargs
        return False


load_dotenv = _load_dotenv
load_dotenv(ROOT / ".env", override=False)

KB_DIR = ROOT / "法规知识库"
DOCX_DIR = KB_DIR / "docx"
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
EVAL_CORPUS_PATH = DATA_DIR / "eval_corpus.json"
EVAL_MULTIHOP_PATH = DATA_DIR / "eval_multihop.json"

ALL_DIRS = (DOCX_DIR, TEXT_DIR, PARSED_DIR, CHUNK_DIR, INDEX_DIR)


def ensure_dirs() -> None:
    """确保管道各阶段输出目录存在。"""
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


def reflect_llm_config() -> LLMConfig:
    """审核节点的模型；**未配置时逐字段回退到 `llm_config()`**。

    为什么值得单独一个模型：审核是整个循环里唯一一个「判断题 + 严格 JSON 输出」的节点，
    也是实测下来唯一有明显判断缺口的地方（100 题里 162 次真判定有 64 次误报）；
    规划轮反而很听话（第 2 轮检索词平均 68% 的字符落在上轮 `missing` 里），不需要换。

    回退是**逐字段**的，不是「有一个没配就整体回退」：只想换模型名时写
    `AGENT_REFLECT_MODEL` 一项即可，不必把 key 和 base_url 再抄一遍。
    三项都不写 = 与单模型时逐位相同。

    ⚠️ 但**模型名不跨家**：`glm-4.7-flash` 只能配智谱的 base_url。只写了模型名而
    base_url 还是百炼的话，端点会回 400 —— 这个失败是响的（`ToolCallingLLM.chat`
    重试后抛 `RuntimeError`），不会静默劣化。
    """
    base = llm_config()
    return LLMConfig(
        base_url=_env("AGENT_REFLECT_BASE_URL") or base.base_url,
        api_key=_env("AGENT_REFLECT_API_KEY") or base.api_key,
        model=_env("AGENT_REFLECT_MODEL") or base.model,
        temperature=base.temperature,
    )


def embed_config() -> EmbedConfig:
    """向量模型配置；未单独配置时回退复用 LLM 端点。"""
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
    """Agent 循环的参数（阶段二）。

    与 RetrieveConfig 的分工：这里只管**循环**（跑几轮、给模型看多少字），
    检索本身的参数（top_k / candidates / rrf_k / booster）仍由 RetrieveConfig 说了算 ——
    只有一份真源，Agent 不许悄悄换一套检索参数，否则和基线的对照就不是同一个检索了。
    """

    max_steps: int = 2
    max_evidence: int = 0
    snippet_chars: int = 120
    article_chars: int = 400
    temperature: float = 0.0
    retries: int = 2


def agent_config() -> AgentConfig:
    return AgentConfig(
        max_steps=_env_int("AGENT_MAX_STEPS", 2),
        max_evidence=_env_int("AGENT_MAX_EVIDENCE", 0),
        snippet_chars=_env_int("AGENT_SNIPPET_CHARS", 120),
        article_chars=_env_int("AGENT_ARTICLE_CHARS", 400),
        temperature=_env_float("AGENT_TEMPERATURE", 0.0),
        retries=_env_int("AGENT_RETRIES", 2),
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
    """Milvus 连接与索引参数。

    中文化依赖内置 jieba 分词器；把 MILVUS_ANALYZER 置空则回退 standard 分析器
    （中文会退化成单字，不推荐）。
    """
    tokenizer = _env("MILVUS_ANALYZER", "jieba")
    return MilvusConfig(
        uri=_env("MILVUS_URI", "http://localhost:19530"),
        token=_env("MILVUS_TOKEN", "root:Milvus"),
        collection=_env("MILVUS_COLLECTION", "traffic_law"),
        analyzer_params={"tokenizer": tokenizer} if tokenizer else {},
        bm25_k1=_env_float("MILVUS_BM25_K1", 1.2),
        bm25_b=_env_float("MILVUS_BM25_B", 0.75),
    )
