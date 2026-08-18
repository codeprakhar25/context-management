"""OpenAI embeddings + token/cost tracking."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

# https://openai.com/api/pricing — approximate; log as estimate
PRICE_PER_1M = {
    "text-embedding-3-small": 0.02,
}


class Embedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: OpenAI | None = None,
        cache_path: Path | None = None,
    ):
        self.model = model
        # Always OpenAI direct for embeds — never OpenRouter / OPENAI_BASE_URL hijack
        self.client = client or OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )
        self.cache_path = cache_path
        self.cache: dict[str, list[float]] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text())
        self.prompt_tokens = 0
        self.calls = 0

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out: list[list[float]] = [None] * len(texts)  # type: ignore
        missing_idx: list[int] = []
        for i, t in enumerate(texts):
            if t in self.cache:
                out[i] = self.cache[t]
            else:
                missing_idx.append(i)

        for start in range(0, len(missing_idx), batch_size):
            chunk_ids = missing_idx[start : start + batch_size]
            chunk = [texts[i] for i in chunk_ids]
            if not chunk:
                continue
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            self.calls += 1
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.prompt_tokens += int(usage.total_tokens)
            for j, item in enumerate(resp.data):
                vec = list(item.embedding)
                idx = chunk_ids[j]
                out[idx] = vec
                self.cache[texts[idx]] = vec
        self._save_cache()
        return np.asarray(out, dtype=np.float32)

    def cost_usd(self) -> float:
        rate = PRICE_PER_1M.get(self.model, 0.02)
        return self.prompt_tokens / 1_000_000.0 * rate

    def stats(self) -> dict[str, Any]:
        return {
            "embed_model": self.model,
            "embed_calls": self.calls,
            "embed_tokens": self.prompt_tokens,
            "embed_cost_usd_est": round(self.cost_usd(), 6),
        }
