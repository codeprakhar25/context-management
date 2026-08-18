"""BM25 Okapi ranker — no embeddings."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from harness.index import Hit

_TOKEN = re.compile(r"[a-z0-9_./+-]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self, facts: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.facts = facts
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(f["text"]) for f in facts]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(len(self.docs), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        n = len(self.docs)
        self.idf = {
            t: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for t, freq in df.items()
        }
        self.tf = [Counter(d) for d in self.docs]

    def _scores(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = [0.0] * len(self.facts)
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i]
            s = 0.0
            for t in q:
                if t not in tf or t not in self.idf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf[t] * (f * (self.k1 + 1)) / denom
            scores[i] = s
        return scores

    def _topk(self, scores: list[float], k: int, mask: list[bool] | None) -> list[Hit]:
        idxs = list(range(len(scores)))
        if mask is not None:
            idxs = [i for i in idxs if mask[i]]
        idxs.sort(key=lambda i: -scores[i])
        hits: list[Hit] = []
        for i in idxs[:k]:
            if scores[i] <= 0 and mask is not None:
                # allow zero only if we still need fill? skip non-positive
                pass
            f = self.facts[i]
            hits.append(
                Hit(
                    id=f["id"],
                    text=f["text"],
                    score=float(scores[i]),
                    project=f.get("project"),
                    path=list(f.get("path") or []),
                )
            )
        return hits

    def retrieve_flat(self, query: str, k: int) -> list[Hit]:
        return self._topk(self._scores(query), k, mask=None)

    def retrieve_hier(self, query: str, k: int, project: str | None, k_global: int) -> list[Hit]:
        scores = self._scores(query)
        is_global = [f.get("project") is None for f in self.facts]
        if project is None:
            return self._topk(scores, k, mask=is_global)

        k_g = min(k_global, k)
        k_p = k - k_g
        is_proj = [f.get("project") == project for f in self.facts]
        proj = self._topk(scores, k_p, mask=is_proj) if k_p > 0 else []
        glob = self._topk(scores, k_g, mask=is_global) if k_g > 0 else []
        by_id: dict[str, Hit] = {}
        for h in proj + glob:
            prev = by_id.get(h.id)
            if prev is None or h.score > prev.score:
                by_id[h.id] = h
        return sorted(by_id.values(), key=lambda h: -h.score)[:k]
