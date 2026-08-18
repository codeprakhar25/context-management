"""Frozen reader: query + pack → short answer via OpenAI chat."""
from __future__ import annotations

import os
import time
import threading
from typing import Any

from openai import OpenAI

# approximate list prices USD / 1M tokens
READER_PRICE = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
}


SYSTEM = (
    "You answer using ONLY the provided memory facts. "
    "If facts are insufficient, say UNKNOWN. "
    "Reply with a short answer only — no preamble."
)


def format_pack(hits: list[Any]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] ({h.id}) {h.text}")
    return "\n".join(lines) if lines else "(no memories)"


class Reader:
    def __init__(self, model: str = "gpt-4o-mini", client: OpenAI | None = None, temperature: float = 0.0):
        self.model = model
        # OpenAI direct only — never OpenRouter
        self.client = client or OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )
        self.temperature = temperature
        self.in_tokens = 0
        self.out_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()
        self.latencies_ms: list[float] = []

    def answer(self, question: str, hits: list[Any]) -> str:
        user = f"Memories:\n{format_pack(hits)}\n\nQuestion: {question}\nShort answer:"
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        with self._lock:
            self.latencies_ms.append((time.perf_counter() - t0) * 1000)
            self.calls += 1
            if usage:
                self.in_tokens += int(usage.prompt_tokens or 0)
                self.out_tokens += int(usage.completion_tokens or 0)
        return (resp.choices[0].message.content or "").strip()

    def cost_usd(self) -> float:
        p = READER_PRICE.get(self.model, {"in": 0.15, "out": 0.60})
        return self.in_tokens / 1e6 * p["in"] + self.out_tokens / 1e6 * p["out"]

    def stats(self) -> dict[str, Any]:
        lats = self.latencies_ms
        p50 = float(np_percentile(lats, 50)) if lats else 0.0
        p95 = float(np_percentile(lats, 95)) if lats else 0.0
        return {
            "reader_model": self.model,
            "reader_calls": self.calls,
            "reader_prompt_tokens": self.in_tokens,
            "reader_completion_tokens": self.out_tokens,
            "reader_cost_usd_est": round(self.cost_usd(), 6),
            "reader_latency_ms_p50": round(p50, 2),
            "reader_latency_ms_p95": round(p95, 2),
        }


def np_percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    import numpy as np

    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))
