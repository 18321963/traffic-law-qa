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

EVAL_RETRIEVAL_PATH = DATA_DIR / "eval_retrieval.json"
EVAL_REFERENCE_PATH = DATA_DIR / "eval_reference.json"
EVAL_NOGOLD_PATH = DATA_DIR / "eval_nogold.json"
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


def region_llm_config() -> LLMConfig:
    """入口地区裁决的模型；**未配置时逐字段回退到 `llm_config()`**。

    为什么值得单独一个模型：地区裁决是全循环唯一一处「判断题 + 严格 JSON 输出」，
    要的是判得准且便宜；规划轮要的是会调工具、也会自己判断什么时候停，两者不是一回事。

    回退是**逐字段**的，不是「有一个没配就整体回退」：只想换模型名时写
    `AGENT_REGION_MODEL` 一项即可，不必把 key 和 base_url 再抄一遍。
    三项都不写 = 与单模型时逐位相同。

    ⚠️ 但**模型名不跨家**：`glm-4.7-flash` 只能配智谱的 base_url。只写了模型名而
    base_url 还是百炼的话，端点会回 400 —— 这个失败是响的（`ToolCallingLLM.chat`
    重试后抛 `RuntimeError`），不会静默劣化。

    这三个键 2026-09 之前叫 `AGENT_REFLECT_*`（当时的服务对象是审核节点，那个节点已并入
    规划轮）。**旧名字不再被读取** —— 留着旧键的 `.env` 会静默回退到 `LLM_*`。
    """
    base = llm_config()
    return LLMConfig(
        base_url=_env("AGENT_REGION_BASE_URL") or base.base_url,
        api_key=_env("AGENT_REGION_API_KEY") or base.api_key,
        model=_env("AGENT_REGION_MODEL") or base.model,
        temperature=base.temperature,
    )


def review_llm_config() -> LLMConfig:
    """末端复核的模型；**未配置时逐字段回退到 `region_llm_config()`**（再往下一层是 `llm_config()`）。

    为什么默认就复用入口那只：复核与入口是同一类活（一段短 JSON 判断，要判得准且便宜），
    而**它绝不能是主模型** —— 让生成答案的那只模型给自己的答案打分，等于自己判自己；
    档位上也说得通：入口与复核都是「一次调用、一段 JSON」，主模型是「多轮工具调用」。

    回退链 `review → region → llm` 与 `region → llm` 同形，三项仍是逐字段回退：
    只想换复核模型时只写 `AGENT_REVIEW_MODEL` 一项。⚠️ 模型名同样不跨家（理由见上）。
    """
    base = region_llm_config()
    return LLMConfig(
        base_url=_env("AGENT_REVIEW_BASE_URL") or base.base_url,
        api_key=_env("AGENT_REVIEW_API_KEY") or base.api_key,
        model=_env("AGENT_REVIEW_MODEL") or base.model,
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
class LangfuseConfig:
    """Langfuse 云端追踪的凭据（可选；配了就开，agent / 评测 / 脚本都读它）。

    两个 key 缺一即 `ready=False`，此时一个客户端都不建、一个字节都不外发 ——
    与没有这个功能时逐位相同。**host 不是秘密，两个 key 是**：任何打印只许出现 host。
    """

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
