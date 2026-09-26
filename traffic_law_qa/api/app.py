from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import config
from ..app.dispatch import MODE_AGENT, run
from ..contracts.answer import Answer, Question
from ..contracts.errors import QaError
from ..contracts.reports import channel_label
from ..infra.sqlite import init_database
from ..tools.schemas import TOP_K_MAX, TOP_K_MIN
from .routes import router, runtime_of
from .runtime import Runtime, agent_runner, boot_runtime

__all__ = ["app"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    try:
        app.state.rt = boot_runtime()
        app.state.boot_error = None
    except QaError as exc:
        app.state.rt = None
        app.state.boot_error = str(exc)
        print(f"[serve] 启动失败：{exc}")
    yield
    app.state.rt = None


app = FastAPI(
    title="交通法规问答 Agent",
    description="分层 RAG 管道 + 强制引用式生成，HTTP 封装",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.exception_handler(QaError)
def _qa_error(request: Request, exc: QaError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
def _runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"服务内部错误：{exc}"})


class QaRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    mode: Literal["ask", "search", "agent"] = Field(
        "ask",
        description="ask=检索+生成（默认）；search=只检索不花钱；agent=Agent 循环（多轮工具调用，慢）",
    )
    top_k: int | None = Field(
        None, ge=TOP_K_MIN, le=TOP_K_MAX, description="覆盖默认召回条数（默认 6）"
    )
    doc_ids: list[str] = Field(
        default_factory=list,
        description="mode=agent 时的会话材料 doc_id（POST /documents mode=session 的返回）",
    )


@app.get("/health")
async def health(request: Request) -> dict:
    rt = runtime_of(request)
    rerank = config.rerank_config()
    return {
        "status": "ok",
        "version": app.version,
        "boot_ms": round(rt.boot_ms, 1),
        "milvus": rt.ready.milvus,
        "laws": rt.ready.laws,
        "articles": rt.ready.articles,
        "rows": rt.ready.rows,
        "channels": channel_label(rt.dense_live),
        "rerank": Path(rerank.model).name if rerank.ready else False,
        "dense_built": rt.ready.dense,
        "index": rt.ready.action,
        "llm_ready": rt.rag.llm_ready,
        "agent_ready": rt.agent_extra and rt.rag.llm_ready,
    }


@app.post("/qa")
def ask(request: Request, req: QaRequest) -> dict:
    rt = runtime_of(request)
    started = time.perf_counter()
    if req.mode == MODE_AGENT:
        answer = agent_runner(rt).ask(
            Question(text=req.question, top_k=req.top_k), material_ids=req.doc_ids
        )
        payload = answer.to_dict()
        retrieval = answer.retrieval
    else:
        result = run(rt.rag, req.question, mode=req.mode, top_k=req.top_k)
        payload = result.to_dict()
        retrieval = result.retrieval if isinstance(result, Answer) else result
    if retrieval is not None:
        rt.dense_live = retrieval.used_vector
    payload["mode"] = req.mode
    payload["request_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


@app.post("/qa/stream")
def ask_stream(request: Request, req: QaRequest) -> StreamingResponse:
    if req.mode == MODE_AGENT:
        raise HTTPException(
            status_code=400,
            detail="流式端点不支持 mode=agent（Agent 是多轮工具循环，没有 token 流可推）：要它请 POST /qa",
        )
    rt = runtime_of(request)
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
    except Exception as exc:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
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
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
