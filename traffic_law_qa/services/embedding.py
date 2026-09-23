from __future__ import annotations

import time
from typing import Any

from .. import config

EMBED_FAILED_NOTE_PREFIX = "向量化失败"


class EmbeddingClient:

    input_desc = "list[str]"
    output_desc = "list[list[float]]"

    def __init__(self, cfg: config.EmbedConfig | None = None, *, retries: int = 3) -> None:
        self.cfg = cfg or config.embed_config()
        self.retries = retries
        self._client = None

    @property
    def available(self) -> bool:
        return self.cfg.ready

    @property
    def unavailable_reason(self) -> str:
        if not self.cfg.api_key:
            return "未配置 EMBED_API_KEY / LLM_API_KEY，只建 BM25 稀疏索引（纯关键词检索）"
        return ""

    @property
    def model_label(self) -> str:
        return f"{self.cfg.model}@{self.cfg.base_url}"

    def _ensure_client(self):
        if self._client is None:
            if not self.available:
                raise RuntimeError(self.unavailable_reason)
            from openai import OpenAI

            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        vectors: list[list[float]] = []
        batch = max(1, self.cfg.batch)
        for start in range(0, len(texts), batch):
            vectors.extend(self._embed_window(client, texts[start : start + batch]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        prefix = self.cfg.query_prefix
        return self.embed([prefix + text if prefix else text])[0]

    def _embed_window(self, client, window: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {}
        if self.cfg.dim:
            kwargs["dimensions"] = self.cfg.dim
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = client.embeddings.create(model=self.cfg.model, input=window, **kwargs)
                return [item.embedding for item in response.data]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"向量化失败（{self.cfg.model}）：{last_error}") from last_error
