"""Milvus 集合：稠密向量 + BM25 稀疏向量（服务端 Function 自动生成）+ 混合检索。

    MilvusStore.recreate  : dim → 建集合（schema + BM25 函数 + 双索引）
    MilvusStore.insert    : list[(Chunk, dense_vector)] → 写入行数
    MilvusStore.hybrid_search : (稠密向量, 查询原文) → list[(chunk_id, score)]

为什么换 Milvus（替掉自研 BM25 + Chroma）：
1. **BM25 稀疏向量由服务端生成**：文本字段声明 `enable_analyzer=True` + 一个
   `FunctionType.BM25` 函数，插入时只给原文，Milvus 自己算稀疏向量；
   查询时也只给原文。我们那套「自己分词 + 维护 df/postings + 落盘 bm25.json」整段删掉。
2. **融合也在服务端**：`hybrid_search` + `RRFRanker(k)` 就是我们要的 RRF，
   不用再手工把两个通道的排名拼起来。
3. **过滤是原生能力**：`law_id in [...]` 直接下推，不用像 Chroma 那样绕 `where` 语法。

中文化依赖 Milvus 内置的 jieba 分词器（`analyzer_params={"tokenizer": "jieba"}`），
所以项目里也不再需要 jieba 这个 pip 依赖。
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, cast

from . import config
from .contracts import Chunk

# 集合字段名（改这里要同步 row_of / _output_fields）
F_CHUNK_ID = "chunk_id"
F_TEXT = "text"
F_DENSE = "dense"
F_SPARSE = "sparse"
F_PARENT = "parent_id"
F_LAW_ID = "law_id"

TEXT_MAX_LENGTH = 4096          # 字符数上限，最长条文 + 前缀约 650 字
SHORT_MAX_LENGTH = 256          # 法名 / 章节名 / 引用串
VARCHAR_MAX_LENGTH = 128        # 各种 id

# 检索时需要的标量字段
OUTPUT_FIELDS = (
    F_PARENT,
    F_LAW_ID,
    "law_name",
    "version",
    "citation",
    "article_no",
    "article_index",
    "part_index",
)


class MilvusError(RuntimeError):
    """Milvus 连接/操作失败，消息里直接给出修法。"""


def row_of(chunk: Chunk, vector: list[float] | None) -> dict[str, Any]:
    """把一个子块变成 Milvus 的一行。

    注意：不需要提供稀疏向量 —— BM25 函数会按 text 字段自动生成。
    """
    row: dict[str, Any] = {
        F_CHUNK_ID: chunk.chunk_id,
        F_TEXT: chunk.embed_text,
        F_PARENT: chunk.parent_id,
        F_LAW_ID: chunk.law_id,
        "law_name": chunk.law_name,
        "version": chunk.version,
        "citation": chunk.citation,
        "chapter": chunk.chapter or "",
        "section": chunk.section or "",
        "article_no": chunk.article_no,
        "article_index": chunk.article_index,
        "part_index": chunk.part_index,
        "part_total": chunk.part_total,
    }
    if vector is not None:
        row[F_DENSE] = vector
    return row


class MilvusStore:
    """Milvus 集合的读写门面（建集合 / 插入 / 混合检索）。"""

    input_desc = "list[(Chunk, vector)] | (稠密向量, 查询文本)"
    output_desc = "写入行数 | list[(chunk_id, score)]"

    def __init__(
        self,
        *,
        uri: str | None = None,
        token: str | None = None,
        collection: str | None = None,
        analyzer_params: dict | None = None,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        timeout: float = 30.0,
        verbose: bool = True,
    ) -> None:
        settings = config.milvus_config()
        self.uri = uri or settings.uri
        self.token = token or settings.token
        self.collection = collection or settings.collection
        self.analyzer_params = analyzer_params or settings.analyzer_params
        self.bm25_k1 = settings.bm25_k1 if bm25_k1 is None else bm25_k1
        self.bm25_b = settings.bm25_b if bm25_b is None else bm25_b
        self.timeout = timeout
        self.verbose = verbose
        self._client = None
        self._dim: int | None = None
        self._has_dense: bool | None = None

    # ============================================================ 连接
    @property
    def client(self):
        """懒加载 Milvus 客户端。"""
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:  # pragma: no cover
                raise MilvusError("未安装 pymilvus，请先执行 pip install -r requirements.txt") from exc

            try:
                self._client = MilvusClient(uri=self.uri, token=self.token, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001
                raise MilvusError(
                    f"连接 Milvus 失败（{self.uri}）：{exc}\n"
                    "  请先启动服务：docker compose up -d，再用 docker compose ps 确认 healthy"
                ) from exc
        return self._client

    def ping(self) -> str:
        """探活：返回版本号，连不上时抛 MilvusError。"""
        try:
            version = self.client.get_server_version()
        except MilvusError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MilvusError(
                f"Milvus 探活失败（{self.uri}）：{exc}\n  请确认 docker compose ps 中 milvus-standalone 为 healthy"
            ) from exc
        return str(version)

    def has_collection(self) -> bool:
        try:
            return bool(self.client.has_collection(self.collection))
        except Exception:  # noqa: BLE001 - 连不上时按"没有"处理
            return False

    def count(self) -> int:
        if not self.has_collection():
            return 0
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    def drop(self) -> None:
        if self.has_collection():
            self.client.drop_collection(self.collection)

    # ============================================================ 建集合
    def recreate(self, *, dim: int | None = None) -> dict:
        """重建集合。dim 为 None 时不建稠密向量字段（纯 BM25 模式）。"""
        from pymilvus import DataType, Function, FunctionType

        client = self.client
        self.drop()
        self._dim = dim
        self._has_dense = dim is not None

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(F_CHUNK_ID, DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH, is_primary=True)
        schema.add_field(
            F_TEXT,
            DataType.VARCHAR,
            max_length=TEXT_MAX_LENGTH,
            enable_analyzer=True,
            analyzer_params=self.analyzer_params,
        )
        schema.add_field(F_PARENT, DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH)
        schema.add_field(F_LAW_ID, DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH)
        schema.add_field("law_name", DataType.VARCHAR, max_length=SHORT_MAX_LENGTH)
        schema.add_field("version", DataType.VARCHAR, max_length=64)
        schema.add_field("citation", DataType.VARCHAR, max_length=SHORT_MAX_LENGTH)
        schema.add_field("chapter", DataType.VARCHAR, max_length=SHORT_MAX_LENGTH)
        schema.add_field("section", DataType.VARCHAR, max_length=SHORT_MAX_LENGTH)
        schema.add_field("article_no", DataType.VARCHAR, max_length=64)
        schema.add_field("article_index", DataType.INT64)
        schema.add_field("part_index", DataType.INT64)
        schema.add_field("part_total", DataType.INT64)
        if dim is not None:
            schema.add_field(F_DENSE, DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(F_SPARSE, DataType.SPARSE_FLOAT_VECTOR)

        # 关键：BM25 函数把 text 字段转成稀疏向量存进 sparse 字段
        schema.add_function(
            Function(
                name="text_bm25_emb",
                input_field_names=[F_TEXT],
                output_field_names=[F_SPARSE],
                function_type=FunctionType.BM25,
            )
        )

        index_params = client.prepare_index_params()
        if dim is not None:
            index_params.add_index(field_name=F_DENSE, index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(
            field_name=F_SPARSE,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": self.bm25_k1, "bm25_b": self.bm25_b},
        )

        client.create_collection(collection_name=self.collection, schema=schema, index_params=index_params)
        try:
            client.load_collection(self.collection)
        except Exception as exc:  # noqa: BLE001 - 自动 load 的版本会忽略
            if self.verbose:
                print(f"[milvus] load_collection 跳过：{exc}")
        return self.describe()

    def describe(self) -> dict:
        """集合结构摘要，写入 index/index_meta.json 便于排查。

        describe_collection 返回的字段里混着 protobuf 容器对象，必须先摘干净才能落 JSON。
        """
        if not self.has_collection():
            return {}
        # pymilvus 的存根把 describe_collection 归到了异步重载，这里显式声明同步返回 dict
        info = cast(dict[str, Any], self.client.describe_collection(self.collection))

        fields = []
        for field in info.get("fields", []) or []:
            params = field.get("params") or {}
            fields.append(
                {
                    "name": field.get("name"),
                    "type": str(field.get("type")),
                    "is_primary": bool(field.get("is_primary", False)),
                    "max_length": params.get("max_length"),
                    "dim": params.get("dim"),
                    "enable_analyzer": params.get("enable_analyzer", field.get("enable_analyzer")),
                    "analyzer_params": _jsonable(params.get("analyzer_params")),
                }
            )

        functions = [
            {
                "name": function.get("name"),
                "type": str(function.get("type")),
                "input": _jsonable(function.get("input_field_names")),
                "output": _jsonable(function.get("output_field_names")),
            }
            for function in info.get("functions", []) or []
        ]

        return {
            "collection": self.collection,
            "uri": self.uri,
            "fields": _jsonable(fields),
            "functions": functions,
            "row_count": self.count(),
        }

    def has_dense_field(self) -> bool:
        if self._has_dense is not None:
            return self._has_dense
        if not self.has_collection():
            return False
        info = cast(dict[str, Any], self.client.describe_collection(self.collection))
        self._has_dense = any(field.get("name") == F_DENSE for field in info.get("fields", []))
        return self._has_dense

    # ============================================================ 写入
    def insert(self, rows: list[dict], *, batch_size: int = 200) -> int:
        if not rows:
            return 0
        written = 0
        for start in range(0, len(rows), batch_size):
            window = rows[start : start + batch_size]
            self.client.insert(collection_name=self.collection, data=window)
            written += len(window)
        self.client.flush(self.collection)
        return written

    # ============================================================ 检索
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
        """稠密 + BM25 稀疏双路召回，服务端 RRF 融合。

        没有配置 embeddin 时自动只跑稀疏通道（dense_vector=None）。
        """
        from pymilvus import AnnSearchRequest, RRFRanker

        requests = []
        if dense_vector is not None and self.has_dense_field():
            requests.append(
                AnnSearchRequest(
                    data=[dense_vector],
                    anns_field=F_DENSE,
                    param={"metric_type": "COSINE"},
                    limit=candidates,
                    expr=filter_expr,
                )
            )
        requests.append(
            AnnSearchRequest(
                data=[query_text],          # 传原文，BM25 函数负责转稀疏向量
                anns_field=F_SPARSE,
                param={"metric_type": "BM25"},
                limit=candidates,
                expr=filter_expr,
            )
        )

        if len(requests) == 1:
            return self._single(requests[0], limit)

        response = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=requests,
            ranker=RRFRanker(rrf_k),
            limit=limit,
            output_fields=[F_LAW_ID],
        )
        return _parse_hits(response, limit)

    @staticmethod
    def _search_kwargs(filter_expr: str | None) -> dict[str, Any]:
        """pymilvus 的 filter 形参声明为 str（不接受 None），所以按需传入。"""
        return {"filter": filter_expr} if filter_expr else {}

    def dense_search(self, vector: list[float], *, limit: int, filter_expr: str | None = None) -> list[tuple[str, float]]:
        """仅稠密通道（用于诊断/对照）。"""
        response = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field=F_DENSE,
            search_params={"metric_type": "COSINE"},
            limit=limit,
            output_fields=[F_LAW_ID],
            **self._search_kwargs(filter_expr),
        )
        return _parse_hits(response, limit)

    def sparse_search(self, query_text: str, *, limit: int, filter_expr: str | None = None) -> list[tuple[str, float]]:
        """仅 BM25 通道（用于诊断/对照）。"""
        response = self.client.search(
            collection_name=self.collection,
            data=[query_text],
            anns_field=F_SPARSE,
            search_params={"metric_type": "BM25"},
            limit=limit,
            output_fields=[F_LAW_ID],
            **self._search_kwargs(filter_expr),
        )
        return _parse_hits(response, limit)

    def _single(self, request, limit: int) -> list[tuple[str, float]]:
        response = self.client.search(
            collection_name=self.collection,
            data=list(request.data),
            anns_field=request.anns_field,
            search_params=request.param,
            limit=limit,
            output_fields=[F_LAW_ID],
            **self._search_kwargs(request.expr),
        )
        return _parse_hits(response, limit)

    # ============================================================ 工具
    @staticmethod
    def law_filter_expr(law_ids: Iterable[str]) -> str | None:
        """把 law_id 列表变成 Milvus 过滤表达式。"""
        values = [law_id for law_id in law_ids if law_id]
        if not values:
            return None
        joined = ", ".join(json.dumps(value, ensure_ascii=False) for value in values)
        return f"{F_LAW_ID} in [{joined}]"

    def stats_summary(self, *, elapsed_ms: float, dense_rows: int, embedding_model: str | None,
                      embedding_dim: int | None, built_at: str) -> dict:
        rows = self.count()
        return {
            "collection": self.collection,
            "uri": self.uri,
            "rows": rows,
            "dense_rows": dense_rows,
            "sparse_rows": rows if rows else 0,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "vector_enabled": bool(dense_rows),
            "built_at": built_at,
            "elapsed_ms": elapsed_ms,
        }


def _jsonable(value: Any) -> Any:
    """把 protobuf 容器等不可序列化对象转成普通 Python 值。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _parse_hits(response: Any, limit: int) -> list[tuple[str, float]]:
    """统一解析 search / hybrid_search 的返回结构。

    坑：MilvusClient 返回的是 `SearchResult`（list 的子类），里面一层是 `HybridHits`，
    元素是 dict；主键在 dict 里的键名是**我们定义的字段名**（chunk_id），不是 "id"。
    照抄文档里 `hit["id"]` 的写法会静默拿到空结果。
    """
    if not response:
        return []
    first = response[0] if isinstance(response, (list, tuple)) else response

    hits: list[tuple[str, float]] = []
    for hit in first or []:
        if isinstance(hit, dict):
            chunk_id = hit.get(F_CHUNK_ID) or hit.get("id") or hit.get("pk")
            score = hit.get("distance", hit.get("score", 0.0))
        else:  # 老式 Hit 对象
            chunk_id = getattr(hit, "id", None) or getattr(hit, "pk", None)
            score = getattr(hit, "distance", getattr(hit, "score", 0.0))
        if chunk_id is None:
            continue
        hits.append((str(chunk_id), float(score)))
    return hits[:limit] if limit else hits


def wait_until_ready(store: MilvusStore, *, timeout: float = 120.0, interval: float = 3.0) -> str:
    """轮询等待 Milvus 就绪（compose 首启需要 30~90 秒）。"""
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return store.ping()
        except MilvusError as exc:
            last_error = exc
            time.sleep(interval)
    raise MilvusError(f"等待 Milvus 就绪超时（{timeout:.0f}s）：{last_error}")
