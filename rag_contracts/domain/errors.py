__all__ = ["QaError", "QaTimeout"]


class QaError(RuntimeError):
    pass


class QaTimeout(QaError):
    pass
