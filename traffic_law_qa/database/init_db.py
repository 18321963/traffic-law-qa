from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from .models import DDL, DocumentRow

__all__ = [
    "connect",
    "init_database",
    "save_document",
    "get_document",
    "list_documents",
    "delete_document",
    "mark_document",
    "new_doc_id",
]


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(DDL)


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]


def save_document(row: DocumentRow, db_path: Path | None = None) -> DocumentRow:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, file, display_name, mode, suffix, sha1,"
            " bytes, law_id, law_name, version, articles, paragraphs, status, note, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row.doc_id,
                row.file,
                row.display_name,
                row.mode,
                row.suffix,
                row.sha1,
                row.bytes,
                row.law_id,
                row.law_name,
                row.version,
                row.articles,
                row.paragraphs,
                row.status,
                row.note,
                row.created_at,
            ),
        )
    return row


def get_document(doc_id: str, db_path: Path | None = None) -> DocumentRow | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return DocumentRow.from_row(row) if row else None


def list_documents(
    *, mode: str | None = None, status: str | None = None, db_path: Path | None = None
) -> list[DocumentRow]:
    sql = "SELECT * FROM documents"
    clauses: list[str] = []
    params: list[str] = []
    if mode:
        clauses.append("mode = ?")
        params.append(mode)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC, doc_id DESC"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [DocumentRow.from_row(row) for row in rows]


def mark_document(doc_id: str, *, status: str, note: str, db_path: Path | None = None) -> bool:
    with connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE documents SET status = ?, note = ? WHERE doc_id = ?", (status, note, doc_id)
        )
    return cursor.rowcount > 0


def delete_document(doc_id: str, db_path: Path | None = None) -> bool:
    with connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    return cursor.rowcount > 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
