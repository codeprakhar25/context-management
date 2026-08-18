"""LLM-as-judge for LoCoMo-style short answers (lead metric)."""
from __future__ import annotations

import os
import re
import threading
from typing import Any

from openai import OpenAI


JUDGE_SYSTEM = """You grade whether a predicted answer is correct given the gold answer.
Be lenient on date/format/paraphrase (e.g. "7 May 2023" vs "May 7, 2023" = correct).
Reply with a single token: YES or NO."""


class AnswerJudge:
    def __init__(self, model: str = "gpt-4o-mini", client: OpenAI | None = None):
        self.model = model
        self.client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.calls = 0
        self._lock = threading.Lock()
        self.in_tokens = 0
        self.out_tokens = 0

    def grade(self, question: str, gold: str, pred: str) -> float:
        if not (pred or "").strip() or pred.strip().upper() == "UNKNOWN":
            return 0.0
        user = (
            f"Question: {question}\n"
            f"Gold answer: {gold}\n"
            f"Predicted answer: {pred}\n"
            "Correct? YES or NO:"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        with self._lock:
            self.calls += 1
            if usage:
                self.in_tokens += int(usage.prompt_tokens or 0)
                self.out_tokens += int(usage.completion_tokens or 0)
        text = (resp.choices[0].message.content or "").strip().upper()
        if re.search(r"\bYES\b", text):
            return 1.0
        return 0.0

    def stats(self) -> dict[str, Any]:
        cost = self.in_tokens / 1e6 * 0.15 + self.out_tokens / 1e6 * 0.60
        return {
            "judge_model": self.model,
            "judge_calls": self.calls,
            "judge_prompt_tokens": self.in_tokens,
            "judge_completion_tokens": self.out_tokens,
            "judge_cost_usd_est": round(cost, 6),
        }
