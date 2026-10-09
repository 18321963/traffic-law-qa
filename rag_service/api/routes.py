from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_contracts import config
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.reports import channel_label
from rag_contracts.domain.retrieval import (
    MATERIAL_TOP_K_DEFAULT,
    MATERIAL_TOP_K_MAX,
    MATERIAL_TOP_K_MIN,
    RetrievalResult,
)

from ..adapters.sqlite import (
    MODE_LABELS,
    MODE_PERMANENT,
    MODE_SESSION,
    STATUS_READY,
    STATUS_REJECTED,
    DocumentRow,
    delete_document,
    get_document,
    list_documents,
    new_doc_id,
    save_document,
    utc_now,
)
from ..indexing.ingest import dry_run, promote, read_material, store_material
from ..query.articles import build_article_index, lookup_text
from .gate import QUEUE_TIMEOUT_DETAIL, RagQueueTimeout, gate
from .runtime import boot_runtime

__all__ = ["router", "runtime_of"]

router = APIRouter()

MODE_NOTE = {
    MODE_SESSION: "仅本次会话可检索，未入知识库；引用标 [材料N]",
    MODE_PERMANENT: "待重建索引后才可检索；重建后进依据链，引用标 [依据N]",
}

REJECTED_NOTE_PREFIX = "拒收"
PERMANENT_DELETE_NOTE = (
    "永久入库的法规不能从这里删：删掉 {name} 后还要重建索引，"
    "手工做 —— 移除 {dir} 下的该文件，再跑 python -m rag_service.cli.build build"
)
UNKNOWN_MODE_NOTE = "mode 只能是 {modes}"
MISSING_DOC_NOTE = "没有这个 doc_id：{doc_id}"


def runtime_of(request: Request) -> Any:
    rt = getattr(request.app.state, "rt", None)
    if rt is None:
        raise HTTPException(
            status_code=503, detail=getattr(request.app.state, "boot_error", None) or "服务未就绪"
        )
    return rt


def _rejected(doc_id: str, mode: str, display_name: str, note: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "accepted": False,
            "doc_id": doc_id,
            "mode": mode,
            "display_name": display_name,
            "note": note,
        },
    )


@router.post("/documents")
async def upload(
    request: Request,
    file: UploadFile = File(..., description="docx / md / txt / pdf"),
    mode: str = Form(MODE_SESSION, description="session=仅本次会话；permanent=入知识库"),
) -> Any:
    runtime_of(request)
    if mode not in MODE_LABELS:
        raise HTTPException(status_code=400, detail=UNKNOWN_MODE_NOTE.format(modes="、".join(MODE_LABELS)))

    display_name = Path(file.filename or "未命名").name
    data = await file.read()
    suffix = Path(display_name).suffix.lower()
    doc_id = new_doc_id()

    if mode == MODE_SESSION:
        paragraphs, note = read_material(data, display_name)
        if paragraphs is None:
            save_document(
                _row(doc_id, display_name, mode, suffix, data, status=STATUS_REJECTED, note=note)
            )
            return _rejected(doc_id, mode, display_name, note)

        store_material(doc_id, data, display_name, paragraphs)
        save_document(
            _row(
                doc_id,
                display_name,
                mode,
                suffix,
                data,
                status=STATUS_READY,
                note=MODE_NOTE[mode],
                paragraphs=len(paragraphs),
            )
        )
        return {
            "accepted": True,
            "doc_id": doc_id,
            "mode": mode,
            "display_name": display_name,
            "paragraphs": len(paragraphs),
            "note": MODE_NOTE[mode],
        }

    plan, message = dry_run(data, display_name)
    if plan is None:
        save_document(
            _row(doc_id, display_name, mode, suffix, data, status=STATUS_REJECTED, note=message)
        )
        return _rejected(doc_id, mode, display_name, message)

    promote(data, plan)
    save_document(
        _row(
            doc_id,
            display_name,
            mode,
            suffix,
            data,
            status=STATUS_READY,
            note=MODE_NOTE[mode],
            law_id=plan.law_id,
            law_name=plan.law_name,
            version=plan.version,
            articles=plan.articles,
            paragraphs=plan.paragraphs,
            file=plan.filename,
        )
    )
    return {
        "accepted": True,
        "doc_id": doc_id,
        "mode": mode,
        "display_name": display_name,
        "file": plan.filename,
        "law_id": plan.law_id,
        "law_name": plan.law_name,
        "version": plan.version,
        "citation": plan.citation,
        "articles": plan.articles,
        "note": MODE_NOTE[mode] + "；POST /reindex 立刻生效，不点则下次问答时自动重建",
    }


def _row(
    doc_id: str,
    display_name: str,
    mode: str,
    suffix: str,
    data: bytes,
    *,
    status: str,
    note: str,
    file: str = "",
    law_id: str | None = None,
    law_name: str | None = None,
    version: str | None = None,
    articles: int = 0,
    paragraphs: int = 0,
) -> DocumentRow:
    return DocumentRow(
        doc_id=doc_id,
        file=file or f"source{suffix}",
        display_name=display_name,
        mode=mode,
        suffix=suffix,
        sha1=hashlib.sha1(data).hexdigest(),
        bytes=len(data),
        law_id=law_id,
        law_name=law_name,
        version=version,
        articles=articles,
        paragraphs=paragraphs,
        status=status,
        note=note,
        created_at=utc_now(),
    )


@router.get("/documents")
def documents(request: Request, mode: str | None = None) -> dict:
    runtime_of(request)
    rows = list_documents(mode=mode)
    return {
        "count": len(rows),
        "documents": [row.to_dict() for row in rows],
        "modes": MODE_LABELS,
    }


@router.delete("/documents/{doc_id}")
def remove(request: Request, doc_id: str) -> dict:
    runtime_of(request)
    row = get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=MISSING_DOC_NOTE.format(doc_id=doc_id))
    if row.mode == MODE_PERMANENT:
        raise HTTPException(
            status_code=400,
            detail=PERMANENT_DELETE_NOTE.format(name=row.file, dir=config.SOURCE_DIR),
        )
    shutil.rmtree(config.upload_dir(doc_id), ignore_errors=True)
    delete_document(doc_id)
    return {"removed": doc_id, "display_name": row.display_name}


@router.post("/reindex")
def reindex(request: Request) -> dict:
    started = time.perf_counter()
    try:
        rt = boot_runtime(rebuild=True)
    except QaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    request.app.state.rt = rt
    request.app.state.boot_error = None
    return {
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "action": rt.ready.action,
        "reason": rt.ready.reason,
        "laws": rt.ready.laws,
        "articles": rt.ready.articles,
        "rows": rt.ready.rows,
        "channels": channel_label(rt.dense_live),
    }


@router.get("/laws")
def laws(request: Request) -> dict:
    rt = runtime_of(request)
    items = [law.to_dict() for law in rt.rag.laws()]
    return {"count": len(items), "laws": items}


class ArticleLookup(BaseModel):
    text: str | None = Field(
        None, description="一句话，条号写在里面（与 article_no 二选一，给了 article_no 就按它走）"
    )
    article_no: str | None = Field(
        None, description="条号，中英文数字都行，例如「第九十条」或「90」"
    )
    law_name: str | None = Field(
        None, description="法规名，可选；多部法规都有这个条号时必须给"
    )


@router.post("/articles/lookup")
def article_lookup(request: Request, req: ArticleLookup) -> dict:
    rt = runtime_of(request)
    if req.article_no:
        result, note = rt.rag.get_article(req.article_no, req.law_name)
    elif req.text:
        parents = rt.rag.parents
        result, note = lookup_text(req.text, parents=parents, index=build_article_index(parents))
    else:
        raise HTTPException(status_code=400, detail="text 与 article_no 至少给一个")

    if result is None:
        result = RetrievalResult(
            query=req.text or req.article_no or "",
            articles=(),
            used_vector=False,
            used_bm25=False,
            elapsed_ms=0.0,
        )
    payload = result.to_dict()
    payload["found"] = not result.is_empty
    payload["note"] = note
    return payload


class MaterialSearch(BaseModel):
    query: str = Field(..., min_length=1, description="检索词，用材料里真出现过的说法")
    doc_ids: list[str] = Field(
        default_factory=list, description="会话材料 doc_id（POST /documents mode=session 的返回）"
    )
    top_k: int = Field(
        MATERIAL_TOP_K_DEFAULT,
        ge=MATERIAL_TOP_K_MIN,
        le=MATERIAL_TOP_K_MAX,
        description=f"返回段数，默认 {MATERIAL_TOP_K_DEFAULT}",
    )


@router.post("/materials/search")
def materials_search(request: Request, req: MaterialSearch) -> dict:
    rt = runtime_of(request)
    with gate() as ok:
        if not ok:
            raise RagQueueTimeout(QUEUE_TIMEOUT_DETAIL)
        hits, text = rt.rag.search_materials(req.query, req.doc_ids, top_k=req.top_k)
    return {
        "query": req.query,
        "count": len(hits),
        "passages": [hit.to_dict() for hit in hits],
        "text": text,
    }
