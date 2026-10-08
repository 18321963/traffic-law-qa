from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_contracts.domain.answer import Question
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.retrieval import TOP_K_MAX, TOP_K_MIN

from .. import container

__all__ = ["app"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runner = None
    app.state.boot_error = None
    try:
        app.state.runner = container.boot_agent_runner(container.build_client())
    except QaError as exc:
        app.state.boot_error = str(exc)
        print(f"[agent] 启动失败：{exc}")
    yield
    app.state.runner = None


app = FastAPI(
    title="交通法规问答 Agent 服务",
    description="Agent 循环（规划 → 工具调用 → 复核 → 生成）；问答与检索都走 rag 服务的 HTTP 面",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(QaError)
def _qa_error(request: Request, exc: QaError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
def _runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"服务内部错误：{exc}"})


def runner_of(request: Request):
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise HTTPException(
            status_code=503, detail=getattr(request.app.state, "boot_error", None) or "服务未就绪"
        )
    return runner


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int | None = Field(
        None, ge=TOP_K_MIN, le=TOP_K_MAX, description="覆盖默认召回条数 / 证据条数（默认 6）"
    )
    doc_ids: list[str] = Field(
        default_factory=list,
        description="本次会话材料的 doc_id（rag 服务 POST /documents mode=session 的返回）",
    )


@app.get("/health")
async def health(request: Request) -> dict:
    runner = runner_of(request)
    try:
        rag = runner.rag.health()
    except QaError as exc:
        rag = {"status": "unreachable", "detail": str(exc)}
    status = "ok" if rag.get("status") == "ok" else "degraded"
    return {"status": status, "version": app.version, "rag": rag}


@app.post("/qa")
def ask(request: Request, req: AgentRequest) -> dict:
    runner = runner_of(request)
    started = time.perf_counter()
    answer = runner.ask(Question(text=req.question, top_k=req.top_k), material_ids=req.doc_ids)
    payload = answer.to_dict()
    payload["request_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


@app.post("/qa/stream")
def ask_stream(request: Request, req: AgentRequest) -> StreamingResponse:
    runner = runner_of(request)
    return _streaming(_events(runner, req))


def _streaming(events) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _events(runner, req: AgentRequest):
    started = time.perf_counter()
    question = Question(text=req.question, top_k=req.top_k)
    try:
        for kind, payload in runner.stream(question, material_ids=req.doc_ids):
            if kind == "step":
                yield _sse("step", payload)
                continue
            if kind != "answer":
                continue
            payload["request_ms"] = round((time.perf_counter() - started) * 1000, 2)
            yield _sse("done", payload)
    except Exception as exc:  # noqa: BLE001
        yield _sse("error", {"message": f"Agent 未完成：{exc}"})


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
