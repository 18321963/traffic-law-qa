from __future__ import annotations

from .. import config
from ..ports import Reranker
from .local_model import LocalModel

__all__ = ["LocalReranker"]


class LocalReranker(LocalModel, Reranker):

    input_desc = "query + list[str]"
    output_desc = "list[float]"

    def __init__(self, cfg: config.RerankConfig | None = None) -> None:
        super().__init__(cfg or config.rerank_config())

    @property
    def unavailable_reason(self) -> str:
        if not self.cfg.enabled:
            return "RAG_RERANK=0，按融合顺序返回"
        if not self.cfg.ready:
            return f"本地没有重排权重 {self.cfg.model}，按融合顺序返回"
        return ""

    @property
    def top_n(self) -> int:
        return self.cfg.top_n

    def _load(self, half: bool):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(str(self.cfg.model))
        if half:
            model = model.half()
        return model, AutoTokenizer.from_pretrained(str(self.cfg.model))

    def score(self, query: str, texts: list[str]) -> list[float]:
        model, tokenizer, device = self._ensure_model()
        import torch

        scores: list[float] = []
        batch = max(1, self.cfg.batch)
        for start in range(0, len(texts), batch):
            window = list(texts[start : start + batch])
            encoded = tokenizer(
                [query] * len(window),
                window,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                logits = model(**encoded).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores
