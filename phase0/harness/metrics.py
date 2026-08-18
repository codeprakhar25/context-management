"""Retrieval + QA metrics."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9.\s=+-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred: str, gold: str) -> float:
    p = _norm(pred).split()
    g = _norm(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common: dict[str, int] = defaultdict(int)
    for t in p:
        common[t] += 1
    match = 0
    for t in g:
        if common[t] > 0:
            match += 1
            common[t] -= 1
    if match == 0:
        return 0.0
    prec = match / len(p)
    rec = match / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


def retrieval_scores(
    hit_ids: list[str],
    gold_ids: list[str],
    hit_projects: list[str | None],
    query_project: str | None,
    query_type: str,
) -> dict[str, float]:
    gold = set(gold_ids)
    hit_set = set(hit_ids)
    recall = len(gold & hit_set) / len(gold) if gold else 0.0

    # in-scope: for local/hard/adversarial with project, count hits in that project or global
    if query_type == "global" or query_project is None:
        # expect global facts
        in_scope = sum(1 for p in hit_projects if p is None)
    else:
        in_scope = sum(1 for p in hit_projects if p == query_project or p is None)
    precision_scope = in_scope / len(hit_ids) if hit_ids else 0.0

    return {
        "recall_at_k": recall,
        "precision_in_scope": precision_scope,
        "n_hits": float(len(hit_ids)),
        "n_gold_hit": float(len(gold & hit_set)),
    }


def answer_in_hits(gold_answer: str, hit_texts: list[str]) -> float:
    """Proxy recall: normalized gold substring appears in any retrieved text."""
    g = _norm(gold_answer)
    if not g or g in ("yes", "no", "unknown"):
        # too weak / binary — skip as 0 rather than inflate
        return 0.0
    for t in hit_texts:
        if g in _norm(t):
            return 1.0
    return 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "recall_at_k",
        "precision_in_scope",
        "em",
        "f1",
        "retrieve_ms",
        "answer_in_hits",
        "judge",
    ]
    out: dict[str, Any] = {"n": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        out[f"mean_{k}"] = sum(vals) / len(vals) if vals else 0.0

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r.get("type", "?")].append(r)
    out["by_type"] = {}
    for t, rs in by_type.items():
        jvals = [float(r["judge"]) for r in rs if r.get("judge") is not None]
        out["by_type"][t] = {
            "n": len(rs),
            "mean_recall_at_k": sum(r.get("recall_at_k", 0.0) for r in rs) / len(rs),
            "mean_precision_in_scope": sum(r.get("precision_in_scope", 0.0) for r in rs)
            / len(rs),
            "mean_em": sum(r["em"] for r in rs) / len(rs),
            "mean_f1": sum(r["f1"] for r in rs) / len(rs),
            "mean_answer_in_hits": sum(r.get("answer_in_hits", 0.0) for r in rs) / len(rs),
            "mean_judge": sum(jvals) / len(jvals) if jvals else None,
        }
    return out
