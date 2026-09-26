from __future__ import annotations

from .. import config
from ..ports import Embedder
from .local_model import LocalModel

EMBED_FAILED_NOTE_PREFIX = "向量化失败"

__all__ = ["EMBED_FAILED_NOTE_PREFIX", "LocalEmbedder"]


class LocalEmbedder(LocalModel, Embedder):

    input_desc = "list[str]"
    output_desc = "list[list[float]]"

    def __init__(self, cfg: config.EmbedConfig | None = None) -> None:
        super().__init__(cfg or config.embed_config())

    @property
    def unavailable_reason(self) -> str:
        if not self.cfg.model:
            return "未配置 EMBED_MODEL，只建 BM25 稀疏索引（纯关键词检索）"
        if not self.cfg.ready:
            return f"本地没有权重目录 {self.cfg.model}，只建 BM25 稀疏索引（纯关键词检索）"
        return ""

    def _load(self, half: bool):
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(str(self.cfg.model))
        if half:
            model = model.half()
        return model, AutoTokenizer.from_pretrained(str(self.cfg.model))

    def embed(self, texts: list[str]) -> list[list[float]]:
        model, tokenizer, device = self._ensure_model()
        import torch

        vectors: list[list[float]] = []
        batch = max(1, self.cfg.batch)
        for start in range(0, len(texts), batch):
            encoded = tokenizer(
                list(texts[start : start + batch]),
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state[:, 0]
            vectors.extend(torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
