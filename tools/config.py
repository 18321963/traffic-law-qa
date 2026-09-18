"""集中式配置：目录布局、模型端点、检索参数。

所有可变项都从环境变量读取（写在项目根目录的 .env 里），代码中不出现任何密钥。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

try:  # python-dotenv 是软依赖，缺失时退化为纯环境变量
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover

    def _load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """python-dotenv 缺失时的空实现，直接读进程环境变量。"""
        del args, kwargs
        return False


load_dotenv = _load_dotenv
load_dotenv(ROOT / ".env", override=False)

# ---------------------------------------------------------------- 目录布局
KB_DIR = ROOT / "法规知识库"
DOCX_DIR = KB_DIR / "docx"          # 唯一真源：原始 docx，永不改写
TEXT_DIR = KB_DIR / "text"          # 人读层：法条 Markdown
PARSED_DIR = KB_DIR / "parsed"      # 结构层：法→章→节→条
CHUNK_DIR = KB_DIR / "chunks"       # 检索层：父子块 jsonl
INDEX_DIR = KB_DIR / "index"        # 索引层：index_meta.json 快照（数据在 Milvus）

MANIFEST_PATH = PARSED_DIR / "manifest.json"
CHUNKS_PATH = CHUNK_DIR / "chunks.jsonl"
PARENTS_PATH = CHUNK_DIR / "parents.jsonl"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"
# 向量数据存在 Milvus 里（docker-compose.yml 起服务），index/ 只留一份元信息快照

ALL_DIRS = (DOCX_DIR, TEXT_DIR, PARSED_DIR, CHUNK_DIR, INDEX_DIR)


def ensure_dirs() -> None:
    """确保管道各阶段输出目录存在。"""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 配置读取
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
    )


@dataclass(frozen=True)
class RetrieveConfig:
    top_k: int = 6            # 返回给 LLM 的法条（父块）数
    candidates: int = 20      # 单通道候选数
    rrf_k: int = 60           # Milvus RRFRanker 的平滑常数
    law_hint_boost: float = 1.5   # 查询命中法名片段时，该法规条文的分数加成


def retrieve_config() -> RetrieveConfig:
    return RetrieveConfig(
        top_k=_env_int("RAG_TOP_K", 6),
        candidates=_env_int("RAG_CANDIDATES", 20),
        rrf_k=_env_int("RAG_RRF_K", 60),
        law_hint_boost=_env_float("RAG_LAW_HINT_BOOST", 1.5),
    )


# ---------------------------------------------------------------- Milvus
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
