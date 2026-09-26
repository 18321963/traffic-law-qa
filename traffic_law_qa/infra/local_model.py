from __future__ import annotations

from pathlib import Path

from .. import config

__all__ = ["LocalModel"]


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
