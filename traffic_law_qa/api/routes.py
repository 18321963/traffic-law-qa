from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .. import config
from ..contracts import QaError
from ..database.init_db import (
    delete_document,
    get_document,
    list_documents,
    new_doc_id,
    save_document,
    utc_now,
)
from ..database.models import (
    MODE_LABELS,
    MODE_PERMANENT,
    MODE_SESSION,
    STATUS_READY,
    STATUS_REJECTED,
    DocumentRow,
)
from ..services.ingest import dry_run, promote, read_paragraphs, store_material

__all__ = ["router", "runtime_of"]

router = APIRouter()

MODE_NOTE = {
    MODE_SESSION: "仅本次会话可检索，未入知识库；引用标 [材料N]",
    MODE_PERMANENT: "待重建索引后才可检索；重建后进依据链，引用标 [依据N]",
}

REJECTED_NOTE_PREFIX = "拒收"
PERMANENT_DELETE_NOTE = (
    "永久入库的法规不能从这里删：删掉 {name} 后还要重建索引，"
    "手工做 —— 移除 {dir} 下的该文件，再跑 python -m traffic_law_qa.pipeline build"
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
    file: UploadFile = File(..., description="docx / md / txt（PDF 暂不支持）"),
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
        try:
            paragraphs = read_paragraphs(data, display_name)
        except Exception as exc:  # noqa: BLE001
            save_document(
                _row(doc_id, display_name, mode, suffix, data, status=STATUS_REJECTED, note=str(exc))
            )
            return _rejected(doc_id, mode, display_name, f"读不出内容：{exc}")
        if not paragraphs:
            note = "文件里没有可解析的段落（是空的，或读取后一个字都没有）"
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
    runtime_of(request)
    from ..main import boot_runtime

    started = time.perf_counter()
    try:
        rt = boot_runtime()
    except QaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    request.app.state.rt = rt
    return {
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "action": rt.ready.action,
        "reason": rt.ready.reason,
        "laws": rt.ready.laws,
        "articles": rt.ready.articles,
        "rows": rt.ready.rows,
        "channels": "稠密+BM25" if rt.dense_live else "纯 BM25",
    }
