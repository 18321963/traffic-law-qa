from __future__ import annotations

import hashlib
from pathlib import Path

from rag_contracts import config

__all__ = ["LocalModel", "weights_fingerprint"]


def weights_fingerprint(model: str) -> str:
    root = Path(model)
    if not root.is_dir():
        return ""
    entries = []
    for item in root.rglob("*"):
        if item.is_file():
            info = item.stat()
            entries.append(
                f"{item.relative_to(root).as_posix()}|{info.st_size}|{info.st_mtime_ns}"
            )
    if not entries:
        return ""
    return hashlib.sha1("\n".join(sorted(entries)).encode("utf-8")).hexdigest()[:16]


class LocalModel:

    _CACHE: dict[tuple, tuple] = {}

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._device = None

    @property
    def available(self) -> bool:
        return not self.unavailable_reason

    @property
    def unavailable_reason(self) -> str:
        raise NotImplementedError

    @property
    def model_label(self) -> str:
        return Path(self.cfg.model).name

    @property
    def fingerprint(self) -> str:
        return weights_fingerprint(self.cfg.model)

    def _load(self, half: bool):
        raise NotImplementedError

    def _ensure_model(self):
        if self._model is None:
            if not self.available:
                raise RuntimeError(self.unavailable_reason)
            import torch

            device = config.resolve_device(self.cfg.device, torch.cuda.is_available())
            half = config.wants_half(self.cfg.dtype, device)
            key = (type(self).__name__, str(self.cfg.model), device, half)
            if key not in LocalModel._CACHE:
                model, tokenizer = self._load(half)
                LocalModel._CACHE[key] = (model.to(device).eval(), tokenizer, device)
            self._model, self._tokenizer, self._device = LocalModel._CACHE[key]
        return self._model, self._tokenizer, self._device
