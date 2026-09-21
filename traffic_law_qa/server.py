"""单文件 HTTP 服务：把 `qa()` 包成 FastAPI，含 SSE 流式。

    uvicorn traffic_law_qa.server:app --port 8000
    # 或安装后：tlq-serve

    curl -X POST localhost:8000/qa -H 'Content-Type: application/json' \\
         -d '{"question":"醉驾怎么处罚"}'

    curl -N -X POST localhost:8000/qa/stream -H 'Content-Type: application/json' \\
         -d '{"question":"醉驾怎么处罚"}'

    curl localhost:8000/health

**这个服务真正想说明的不是「能起服务」，而是「装配只做一次」。**
`qa()` 每次调用都会重新跑一遍索引就绪检查、重读 chunks.jsonl、重连 Milvus ——
命令行里无所谓（进程活一次就退），但放进 HTTP 服务里就是每个请求都付这个钱。
所以这里把检索器与生成器在 lifespan 里装配一次缓存在 `app.state`，
请求只做「检索 + 生成」。两者的耗时差距在启动日志里会打出来。

`/health` 有两个通道字段，别只看一个：`channels` 报**此刻实际**在走的通道
（随预热与每次检索更新），`dense_built` 报集合**建库时**带没带稠密向量。
`纯 BM25` 配上 `dense_built: true` 就是「向量端点挂了，但服务照常在答」。

容器化见仓库根的 `Dockerfile` 与 `docker-compose.yml`（`app` 服务）。
容器里唯一的差别是启动参数：默认绑 `127.0.0.1` 只对本地裸跑安全，
容器内必须 `--host 0.0.0.0`，否则宿主机映射过来的端口连不上。

刻意**不做**的事（本服务的边界）：鉴权、限流、多副本、
请求级的超时与重试、把 server 拆成包。端到端只有三个端点，够跑通、够被 curl 验。
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .api import QaError, ReadyState, ensure_ready
from .contracts import Question
from .qa.rag import LegalRAG

__all__ = ["app", "main"]


@dataclass
class Runtime:
    """进程级装配好的部件。lifespan 里建一次，所有请求复用。"""

    ready: ReadyState
    rag: LegalRAG
    boot_ms: float
    dense_live: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时装配一次；失败不退出，把原因留在 /health 里。

    起不来的常见原因是 Milvus 没起或没配 LLM key —— 那种情况下进程直接崩掉
    会让人只看到一串堆栈。留住失败原因、让 /health 说出来，更好排查。
    """
    try:
        app.state.rt = _boot()
        app.state.boot_error = None
    except QaError as exc:
        app.state.rt = None
        app.state.boot_error = str(exc)
        print(f"[serve] 启动失败：{exc}")
    yield
    app.state.rt = None


app = FastAPI(
    title="交通法规问答 Agent",
    description="分层 RAG 管道 + 强制引用式生成，单文件 HTTP 封装",
    version="0.1.0",
    lifespan=lifespan,
)


def _boot() -> Runtime:
    """就绪检查 + 装配 RAG 工具。这一步可能触发建库，慢是正常的。"""
    started = time.perf_counter()
    ready = ensure_ready()
    rag = LegalRAG.load()
    warm_ms = rag.warm()
    boot_ms = (time.perf_counter() - started) * 1000
    print(f"[serve] 装配完成（{boot_ms / 1000:.2f}s）：{ready.describe()}")
    warm_note = f"{warm_ms:.0f}ms" if warm_ms is not None else "未执行（向量端点未配置或不可达，检索只走 BM25）"
    print(f"[serve] 稠密通道预热 {warm_note}")
    dense_live = ready.dense and warm_ms is not None
    return Runtime(ready=ready, rag=rag, boot_ms=boot_ms, dense_live=dense_live)


class QaRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    mode: Literal["ask", "search"] = Field("ask", description="ask=检索+生成，search=只检索")
    top_k: int | None = Field(None, ge=1, le=20, description="覆盖默认召回条数（默认 6）")


def _runtime() -> Runtime:
    rt = getattr(app.state, "rt", None)
    if rt is None:
        raise HTTPException(status_code=503, detail=app.state.boot_error or "服务未就绪")
    return rt


@app.get("/health")
def health() -> dict:
    """存活 + 库内规模 + 装配耗时 + 通道实况。未就绪时返回 503 与原因。

    `channels` 报**此刻实际**在走的通道，`dense_built` 报集合**建库时**带没带稠密向量。
    两个一起看才诊断得出来：`纯 BM25` 配 `dense_built: true` 是向量端点坏了；
    `dense_built: false` 是索引本来就只建了 BM25。

    通道降级**不改状态码**：端点挂了服务照常在答（BM25 是设计好的降级路径），
    判成 503 会让容器无谓地 flapping 成 unhealthy —— 那是另一种撒谎。
    """
    rt = getattr(app.state, "rt", None)
    if rt is None:
        raise HTTPException(status_code=503, detail=app.state.boot_error or "服务未就绪")
    return {
        "status": "ok",
        "boot_ms": round(rt.boot_ms, 1),
        "milvus": rt.ready.milvus,
        "laws": rt.ready.laws,
        "articles": rt.ready.articles,
        "rows": rt.ready.rows,
        "channels": "稠密+BM25" if rt.dense_live else "纯 BM25",
        "dense_built": rt.ready.dense,
        "index": rt.ready.action,
        "llm_ready": rt.rag.llm_ready,
    }


@app.post("/qa")
def ask(req: QaRequest) -> dict:
    """一次问答。mode=search 时只返回检索到的依据，不花 LLM 的钱。"""
    rt = _runtime()
    started = time.perf_counter()
    retrieval = rt.rag.search(req.question, top_k=req.top_k)
    rt.dense_live = retrieval.used_vector
    if req.mode == "search":
        payload = retrieval.to_dict()
    else:
        answer = rt.rag.answer(Question(text=req.question, top_k=req.top_k), retrieval)
        payload = answer.to_dict()
    payload["request_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


@app.post("/qa/stream")
def ask_stream(req: QaRequest) -> StreamingResponse:
    """SSE：先推检索到的依据，再逐块推生成的 token。

    事件类型：
      `evidence`  一次，检索结果（依据清单与条号）
      `delta`     多次，答案的文本片段
      `done`      一次，完整答案 + 耗时 + token 用量
      `error`     出错时一次，说明原因（可能是流中途才发生）
    """
    rt = _runtime()
    return StreamingResponse(
        _sse_events(rt, req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_events(rt: Runtime, req: QaRequest):
    started = time.perf_counter()
    try:
        retrieval = rt.rag.search(req.question, top_k=req.top_k)
    except Exception as exc:  # noqa: BLE001 - 检索层失败要给用户一句话而不是断连
        yield _sse("error", {"message": f"检索失败：{exc}"})
        return
    rt.dense_live = retrieval.used_vector

    yield _sse("evidence", retrieval.to_dict())

    question = Question(text=req.question, top_k=req.top_k)
    parts: list[str] = []
    usage: dict = {}
    try:
        for kind, payload in rt.rag.stream(question, retrieval):
            if kind == "delta":
                parts.append(payload)
                yield _sse("delta", {"text": payload})
            elif kind == "usage":
                usage = payload
    except Exception as exc:  # noqa: BLE001 - 已吐出的 token 收不回，只能补一个 error 事件
        yield _sse("error", {"message": str(exc), "partial": "".join(parts)})
        return

    yield _sse(
        "done",
        {
            "question": req.question,
            "answer": "".join(parts),
            "model": rt.rag.model_name,
            "usage": usage,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "citations": [hit.citation for hit in retrieval.articles],
        },
    )


def _sse(event: str, payload: dict) -> str:
    """SSE 帧。data 一律走 JSON —— 答案里有换行，裸拼会把一行拆成多行。"""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def main(argv: list[str] | None = None) -> int:
    """tlq-serve [--host H] [--port P]"""
    import sys

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="tlq-serve",
        description="把 qa() 包成 HTTP 服务。装配在启动时做一次，请求只做检索 + 生成。",
        epilog="等价写法：python -m traffic_law_qa.server；"
        "或让 uvicorn 直接指定 app：uvicorn traffic_law_qa.server:app --port 8000",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址（默认 127.0.0.1；容器内必须显式给 0.0.0.0，否则映射出来的端口连不上）",
    )
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")

    try:
        options = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    uvicorn.run(app, host=options.host, port=options.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
