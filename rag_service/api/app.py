from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_contracts import config
from rag_contracts.domain.answer import Answer, Question, drain
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.reports import WEIGHTS_OK, channel_label
from rag_contracts.domain.retrieval import (
    TOP_K_MAX,
    TOP_K_MIN,
    MaterialPassage,
    RetrievalResult,
    WebFinding,
)
from rag_contracts.ports import TRUNCATED_FINISH_REASON

from ..adapters.sqlite import init_database
from ..query.dispatch import run
from .routes import router, runtime_of
from .runtime import Runtime, boot_runtime

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
    title="交通法规知识库服务",
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


class QuestionBody(BaseModel):
    text: str = Field(..., min_length=1, description="用户问题")
    history: list[tuple[str, str]] = Field(
        default_factory=list, description="多轮历史 (role, content)，原样进生成提示词"
    )


class AnswerRequest(BaseModel):
    question: QuestionBody = Field(..., description="问题与历史；没有 top_k，生成用不到它")
    retrieval: dict = Field(
        ...,
        description="已经检索好的 RetrievalResult.to_dict()（POST /qa?mode=search 的返回）",
    )
    timeliness: list[dict] = Field(default_factory=list, description="[时效N] 联网片段")
    materials: list[dict] = Field(default_factory=list, description="[材料N] 会话材料片段")


class QaRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    mode: Literal["ask", "search"] = Field(
        "ask", description="ask=检索+生成（默认）；search=只检索不花钱"
    )
    top_k: int | None = Field(
        None, ge=TOP_K_MIN, le=TOP_K_MAX, description="覆盖默认召回条数（默认 6）"
    )
    pool: int | None = Field(
        None,
        ge=TOP_K_MIN,
        le=config.pool_max(),
        description=(
            "候选池深度，同时按它放行超过 top_k 上限的返回条数（默认取 RAG_CANDIDATES=20，"
            "上限 RAG_POOL_MAX）。与 eval 的 --pool 同口径；给了它就盖过 top_k"
        ),
    )
    debug: bool = Field(
        False,
        description="检索诊断：带上两路通道各自的名次与原始分（该通道没跑时为 null）",
    )
    law_filter: list[str] = Field(
        default_factory=list,
        description="只在给定 law_id 内检索（GET /laws 的 law_id）",
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
        "weights": rt.ready.weights,
        "degraded": rt.ready.weights != WEIGHTS_OK,
        "llm_ready": rt.rag.llm_ready,
    }


@app.post("/qa")
def ask(request: Request, req: QaRequest) -> dict:
    rt = runtime_of(request)
    started = time.perf_counter()
    result = run(
        rt.rag,
        req.question,
        mode=req.mode,
        top_k=req.pool or req.top_k,
        channel_debug=req.debug,
        law_filter=tuple(req.law_filter),
        candidates=req.pool,
    )
    payload = result.to_dict()
    retrieval = result.retrieval if isinstance(result, Answer) else result
    if retrieval is not None:
        rt.dense_live = retrieval.used_vector
    payload["mode"] = req.mode
    payload["request_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


@app.post("/answer")
def generate(request: Request, req: AnswerRequest) -> dict:
    rt = runtime_of(request)
    started = time.perf_counter()
    question = Question(
        text=req.question.text,
        history=tuple((str(role), str(content)) for role, content in req.question.history),
    )
    retrieval = RetrievalResult.from_dict(req.retrieval)
    answer = rt.rag.answer(
        question,
        retrieval,
        timeliness=tuple(WebFinding.from_dict(item) for item in req.timeliness),
        materials=tuple(MaterialPassage.from_dict(item) for item in req.materials),
    )
    rt.dense_live = retrieval.used_vector
    payload = answer.to_dict()
    payload["request_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


@app.post("/answer/stream")
def generate_stream(request: Request, req: AnswerRequest) -> StreamingResponse:
    return _streaming(_answer_events(runtime_of(request), req))


@app.post("/qa/stream")
def ask_stream(request: Request, req: QaRequest) -> StreamingResponse:
    return _streaming(_sse_events(runtime_of(request), req))


def _streaming(events) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_events(rt: Runtime, req: QaRequest):
    started = time.perf_counter()
    try:
        retrieval = rt.rag.search(
            req.question,
            top_k=req.pool or req.top_k,
            channel_debug=req.debug,
            law_filter=tuple(req.law_filter),
            candidates=req.pool,
        )
    except Exception as exc:  # noqa: BLE001
        yield _sse("error", {"message": f"检索失败：{exc}"})
        return
    rt.dense_live = retrieval.used_vector

    yield _sse("evidence", retrieval.to_dict())

    question = Question(text=req.question, top_k=req.top_k)
    parts: list[str] = []
    usage: dict = {}
    truncated = False
    try:
        for kind, payload in rt.rag.stream(question, retrieval):
            if kind == "delta":
                parts.append(payload)
                yield _sse("delta", {"text": payload})
            elif kind == "usage":
                usage = payload
            elif kind == "finish_reason":
                truncated = payload == TRUNCATED_FINISH_REASON
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
            "truncated": truncated,
        },
    )


def _answer_events(rt: Runtime, req: AnswerRequest):
    started = time.perf_counter()
    question = Question(
        text=req.question.text,
        history=tuple((str(role), str(content)) for role, content in req.question.history),
    )
    retrieval = RetrievalResult.from_dict(req.retrieval)
    timeliness = tuple(WebFinding.from_dict(item) for item in req.timeliness)
    materials = tuple(MaterialPassage.from_dict(item) for item in req.materials)
    rt.dense_live = retrieval.used_vector

    out: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            frames = rt.rag.stream(question, retrieval, timeliness=timeliness, materials=materials)
            answer = drain(frames, on_delta=lambda text: out.put(("delta", text)))
            payload = answer.to_dict()
            payload["request_ms"] = round((time.perf_counter() - started) * 1000, 2)
            out.put(("done", payload))
        except Exception as exc:  # noqa: BLE001
            out.put(("error", {"message": str(exc)}))
        finally:
            out.put(None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = out.get()
        if item is None:
            return
        kind, payload = item
        if kind == "delta":
            yield _sse("delta", {"text": payload})
        else:
            yield _sse(kind, payload)


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
