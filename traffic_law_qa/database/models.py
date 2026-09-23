from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = [
    "DDL",
    "MODE_PERMANENT",
    "MODE_SESSION",
    "STATUS_READY",
    "STATUS_REJECTED",
    "STATUS_SUPERSEDED",
    "DocumentRow",
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
