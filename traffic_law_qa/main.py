from __future__ import annotations

import argparse
import importlib.util
import json
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .api import QaError, ReadyState, ensure_ready
from .api.routes import router, runtime_of
from .contracts import Question
from .database.init_db import init_database
from .qa.rag import LegalRAG

if TYPE_CHECKING:
    from .agents.graph import AgentRunner

__all__ = ["app", "main"]


@dataclass
class Runtime:

    ready: ReadyState
    rag: LegalRAG
    boot_ms: float
    dense_live: bool = False
    agent_extra: bool = False
    agent: "AgentRunner | None" = None


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


def boot_runtime() -> Runtime:
    started = time.perf_counter()
    ready = ensure_ready()
    rag = LegalRAG.load()
    warm_ms = rag.warm()
    boot_ms = (time.perf_counter() - started) * 1000
    print(f"[serve] 装配完成（{boot_ms / 1000:.2f}s）：{ready.describe()}")
    warm_note = f"{warm_ms:.0f}ms" if warm_ms is not None else "未执行（向量端点未配置或不可达，检索只走 BM25）"
    print(f"[serve] 稠密通道预热 {warm_note}")
    dense_live = ready.dense and warm_ms is not None
    return Runtime(
        ready=ready, rag=rag, boot_ms=boot_ms, dense_live=dense_live, agent_extra=_agent_extra()
    )


def _agent_extra() -> bool:
    try:
        return importlib.util.find_spec("langgraph") is not None
    except (ImportError, ValueError):
        return False


_AGENT_LOCK = threading.Lock()


def _agent_runner(rt: Runtime) -> "AgentRunner":
    if rt.agent is not None:
        return rt.agent
    with _AGENT_LOCK:
        if rt.agent is not None:
            return rt.agent
        if not rt.agent_extra:
            raise QaError('未安装 langgraph：mode="agent" 需要 pip install -e ".[agent]"')
        if not rt.rag.llm_ready:
            raise QaError("未配置 LLM_API_KEY：agent 这条路要走模型（只检索请用 mode=search）")
        from .agents.graph import AgentRunner

        runner = AgentRunner.attach(rt.rag)
        runner.graph()
        rt.agent = runner
        return runner


class QaRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    mode: Literal["ask", "search", "agent"] = Field(
        "ask",
        description="ask=检索+生成（默认）；search=只检索不花钱；agent=Agent 循环（多轮工具调用，慢）",
    )
    top_k: int | None = Field(None, ge=1, le=20, description="覆盖默认召回条数（默认 6）")
    doc_ids: list[str] = Field(
        default_factory=list,
        description="mode=agent 时的会话材料 doc_id（POST /documents mode=session 的返回）",
    )


@app.get("/health")
async def health(request: Request) -> dict:
    rt = runtime_of(request)
    return {
        "status": "ok",
        "version": app.version,
        "boot_ms": round(rt.boot_ms, 1),
        "milvus": rt.ready.milvus,
        "laws": rt.ready.laws,
        "articles": rt.ready.articles,
        "rows": rt.ready.rows,
        "channels": "稠密+BM25" if rt.dense_live else "纯 BM25",
        "dense_built": rt.ready.dense,
        "index": rt.ready.action,
        "llm_ready": rt.rag.llm_ready,
        "agent_ready": rt.agent_extra and rt.rag.llm_ready,
    }


@app.post("/qa")
def ask(request: Request, req: QaRequest) -> dict:
    rt = runtime_of(request)
    started = time.perf_counter()
    if req.mode == "agent":
        answer = _agent_runner(rt).ask(
            Question(text=req.question, top_k=req.top_k), material_ids=req.doc_ids
        )
        if answer.retrieval is not None:
            rt.dense_live = answer.retrieval.used_vector
        payload = answer.to_dict()
    else:
        retrieval = rt.rag.search(req.question, top_k=req.top_k)
        rt.dense_live = retrieval.used_vector
        if req.mode == "search":
            payload = retrieval.to_dict()
        else:
            answer = rt.rag.answer(Question(text=req.question, top_k=req.top_k), retrieval)
            payload = answer.to_dict()
    payload["mode"] = req.mode
    payload["request_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


@app.post("/qa/stream")
def ask_stream(request: Request, req: QaRequest) -> StreamingResponse:
    if req.mode == "agent":
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


def main(argv: list[str] | None = None) -> int:
    import sys

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="tlq-serve",
        description="把 qa() 包成 HTTP 服务。装配在启动时做一次，请求只做检索 + 生成。",
        epilog="等价写法：python -m traffic_law_qa.main；"
        "或让 uvicorn 直接指定 app：uvicorn traffic_law_qa.main:app --port 8000。"
        '命令行问答那条路是 python -m traffic_law_qa "问题"；端点一览在 /docs',
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
