from __future__ import annotations

from rag_contracts.domain.errors import QaError

__all__ = ["ResumeConflict", "SessionUnsupported"]


class SessionUnsupported(QaError):
    pass


class ResumeConflict(QaError):
    pass
