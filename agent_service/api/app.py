from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_contracts import config
from rag_contracts.domain.answer import Question
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.retrieval import TOP_K_MAX, TOP_K_MIN

from .. import container
from ..agents.errors import ResumeConflict, SessionUnsupported
from . import auth

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

_limiter = auth.MinuteWindowLimiter()


@app.middleware("http")
async def _guard(request: Request, call_next):
    if request.url.path not in auth.EXEMPT_PATHS:
        cfg = config.agent_config()
        if cfg.api_keys:
            presented = request.headers.get(auth.API_KEY_HEADER, "")
            if not auth.keys_match(presented, cfg.api_keys):
                return JSONResponse(status_code=401, content={"detail": auth.UNAUTHORIZED_DETAIL})
            if cfg.rate_limit_rpm > 0:
                retry_after = _limiter.hit(presented, cfg.rate_limit_rpm, time.time())
                if retry_after:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": auth.rate_limit_detail(cfg.rate_limit_rpm)},
                        headers={"Retry-After": str(retry_after)},
                    )
    return await call_next(request)


@app.exception_handler(QaError)
def _qa_error(request: Request, exc: QaError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(SessionUnsupported)
def _session_unsupported(request: Request, exc: SessionUnsupported) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ResumeConflict)
def _resume_conflict(request: Request, exc: ResumeConflict) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


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
    session_id: str | None = Field(
        None, min_length=1, description="会话 id：带上才有跨轮记忆；响应会把同一个值回传"
    )


class ResumeValue(BaseModel):
    region: str = Field(..., min_length=1, description="澄清回复的地区名；national=只按全国法作答")


class ResumeRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="那个被中断响应的 session_id")
    value: ResumeValue = Field(
        ..., description='澄清回复，形如 {"region": "national"} 或 {"region": "深圳"}'
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
    status, payload = runner.ask_payload(
        Question(text=req.question, top_k=req.top_k),
        material_ids=req.doc_ids,
        session_id=req.session_id,
    )
    return {"status": status, **payload, "request_ms": round((time.perf_counter() - started) * 1000, 2)}


@app.post("/qa/resume")
def resume(request: Request, req: ResumeRequest) -> dict:
    runner = runner_of(request)
    started = time.perf_counter()
    status, payload = runner.resume(req.session_id, {"region": req.value.region})
    return {"status": status, **payload, "request_ms": round((time.perf_counter() - started) * 1000, 2)}


@app.post("/qa/stream")
def ask_stream(request: Request, req: AgentRequest) -> StreamingResponse:
    runner = runner_of(request)
    return _streaming(
        _events(
            runner.stream(
                Question(text=req.question, top_k=req.top_k),
                material_ids=req.doc_ids,
                session_id=req.session_id,
            )
        )
    )


@app.post("/qa/resume/stream")
def resume_stream(request: Request, req: ResumeRequest) -> StreamingResponse:
    runner = runner_of(request)
    return _streaming(_events(runner.resume_stream(req.session_id, {"region": req.value.region})))


def _streaming(events) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _events(events):
    started = time.perf_counter()
    try:
        for kind, payload in events:
            if kind == "step":
                yield _sse("step", payload)
                continue
            if kind == "delta":
                yield _sse("delta", payload)
                continue
            if kind == "interrupt":
                yield _sse("interrupt", payload)
                continue
            if kind != "answer":
                continue
            payload["request_ms"] = round((time.perf_counter() - started) * 1000, 2)
            yield _sse("done", payload)
    except Exception as exc:  # noqa: BLE001
        yield _sse("error", {"message": f"Agent 未完成：{exc}"})


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
