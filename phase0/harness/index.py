"""Flat and hierarchical exact top-k cosine retrieve."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


@dataclass
class Hit:
    id: str
    text: str
    score: float
    project: str | None
    path: list[str]


class MemoryIndex:
    def __init__(self, facts: list[dict[str, Any]], vectors: np.ndarray):
        assert len(facts) == len(vectors)
        self.facts = facts
        self.vectors = _l2_normalize(vectors.astype(np.float32))
        self.id_to_i = {f["id"]: i for i, f in enumerate(facts)}

    def _hits_from_scores(self, scores: np.ndarray, k: int, mask: np.ndarray | None) -> list[Hit]:
        s = scores.copy()
        if mask is not None:
            s = np.where(mask, s, -1e9)
        k = min(k, int(np.sum(mask)) if mask is not None else len(s))
        if k <= 0:
            return []
        # argpartition then sort top-k
        idx = np.argpartition(-s, kth=k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        hits: list[Hit] = []
        for i in idx:
            if s[i] < -1e8:
                continue
            f = self.facts[int(i)]
            hits.append(
                Hit(
                    id=f["id"],
                    text=f["text"],
                    score=float(s[i]),
                    project=f.get("project"),
                    path=list(f.get("path") or []),
                )
            )
        return hits

    def retrieve_flat(self, qvec: np.ndarray, k: int) -> list[Hit]:
        q = _l2_normalize(qvec.reshape(1, -1))[0]
        scores = self.vectors @ q
        return self._hits_from_scores(scores, k, mask=None)

    def retrieve_hier(
        self,
        qvec: np.ndarray,
        k: int,
        project: str | None,
        k_global: int,
    ) -> list[Hit]:
        """Path-scoped: top-(k-k_global) in project + top-k_global in global.

        If project is None (pure global query): all k from global only.
        """
        q = _l2_normalize(qvec.reshape(1, -1))[0]
        scores = self.vectors @ q

        is_global = np.array([f.get("project") is None for f in self.facts], dtype=bool)

        if project is None:
            return self._hits_from_scores(scores, k, mask=is_global)

        k_g = min(k_global, k)
        k_p = k - k_g
        is_proj = np.array([f.get("project") == project for f in self.facts], dtype=bool)

        proj_hits = self._hits_from_scores(scores, k_p, mask=is_proj) if k_p > 0 else []
        glob_hits = self._hits_from_scores(scores, k_g, mask=is_global) if k_g > 0 else []

        # merge unique by id, keep higher score
        by_id: dict[str, Hit] = {}
        for h in proj_hits + glob_hits:
            prev = by_id.get(h.id)
            if prev is None or h.score > prev.score:
                by_id[h.id] = h
        merged = sorted(by_id.values(), key=lambda h: -h.score)
        return merged[:k]

    def retrieve_subtree(
        self,
        qvec: np.ndarray,
        k: int,
        active_path: list[str],
    ) -> list[Hit]:
        """ANN only over facts whose path is active_path or under it."""
        q = _l2_normalize(qvec.reshape(1, -1))[0]
        scores = self.vectors @ q
        active = list(active_path or [])

        def in_subtree(f: dict) -> bool:
            p = list(f.get("path") or [])
            if not active:
                return True
            return len(p) >= len(active) and p[: len(active)] == active

        mask = np.array([in_subtree(f) for f in self.facts], dtype=bool)
        return self._hits_from_scores(scores, k, mask=mask)
