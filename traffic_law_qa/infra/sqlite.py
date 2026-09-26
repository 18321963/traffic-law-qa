from __future__ import annotations

import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import config

__all__ = [
    "DDL",
    "MODE_LABELS",
    "MODE_PERMANENT",
    "MODE_SESSION",
    "STATUS_READY",
    "STATUS_REJECTED",
    "STATUS_SUPERSEDED",
    "DocumentRow",
    "connect",
    "init_database",
    "save_document",
    "get_document",
    "list_documents",
    "delete_document",
    "mark_document",
    "new_doc_id",
    "utc_now",
]


MODE_SESSION = "session"
MODE_PERMANENT = "permanent"

STATUS_READY = "ready"
STATUS_REJECTED = "rejected"
STATUS_SUPERSEDED = "superseded"

MODE_LABELS = {MODE_SESSION: "会话材料", MODE_PERMANENT: "永久入库"}

DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    file         TEXT NOT NULL,
    display_name TEXT NOT NULL,
    mode         TEXT NOT NULL,
    suffix       TEXT NOT NULL,
    sha1         TEXT NOT NULL,
    bytes        INTEGER NOT NULL DEFAULT 0,
    law_id       TEXT,
    law_name     TEXT,
    version      TEXT,
    articles     INTEGER NOT NULL DEFAULT 0,
    paragraphs   INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL,
    note         TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_mode ON documents(mode, created_at);
"""


@dataclass(frozen=True)
class DocumentRow:

    doc_id: str
    file: str
    display_name: str
    mode: str
    suffix: str
    sha1: str
    bytes: int
    law_id: str | None
    law_name: str | None
    version: str | None
    articles: int
    paragraphs: int
    status: str
    note: str
    created_at: str

    @property
    def mode_label(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_READY

    def to_dict(self) -> dict:
        data = asdict(self)
        data["mode_label"] = self.mode_label
        data["accepted"] = self.accepted
        return data

    @classmethod
    def from_row(cls, row) -> "DocumentRow":
        return cls(**{key: row[key] for key in row.keys()})


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
