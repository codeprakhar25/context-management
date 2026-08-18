"""Write-policy metrics: op match (strict + lenient) + invalidation / drop rates.

Strict vs lenient
-----------------
`ops_match_strict` = exact multiset of (op, target). Kept for reference.

`ops_match_lenient` additionally treats as equivalent:
  1. NOOP target id — NOOP mutates nothing, so its target is not identifying.
     A manager that answers {"event":"NOOP","id":null} is as correct as one
     that names the covered memory.
  2. Padding NOOPs — a NOOP emitted alongside real ops is a no-op by
     definition; it is dropped before comparison.
  3. UPDATE-as-merge — `DELETE t + ADD` and `UPDATE t` reach the same end
     state *only when the ADD text preserves what the DELETE removed*.
     Memory-R1 defines UPDATE as merge-preserving ("keep the same ID and
     preserve old_memory"), so a manager that spells a merge as DELETE+ADD is
     not wrong — but a manager that deletes "Blake adopted Luna" and adds only
     "Blake adopted Nala" has destroyed a true fact, whatever it called the op.

     So the fold is asymmetric:
       gold DELETE+ADD vs pred UPDATE   -> always folded (UPDATE preserves)
       gold UPDATE vs pred DELETE+ADD   -> folded only if the ADD text covers
                                           every content token of the deleted
                                           text (`_pred_fold_ok`)

Lenient is the headline; strict is reported next to it so the gap is visible.
`false_invalidation` remains the independent check on end state — read it
alongside op accuracy, never instead of it.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

OpKey = tuple[str, str | None]


def normalize_gold_ops(gold_ops: list[dict]) -> list[OpKey]:
    return [(g["op"].upper(), g.get("target_id")) for g in gold_ops]


def normalize_pred_ops(logs: list[dict]) -> list[OpKey]:
    return [(l["op"].upper(), l.get("fact_id")) for l in logs]


def _strict_key(ot: OpKey) -> OpKey:
    op, tid = ot
    return ("ADD", None) if op == "ADD" else (op, tid)


_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "to", "in", "of", "and",
    "or", "later", "another", "has", "have", "had", "by", "for", "on", "at",
    "with", "that", "this", "it", "as", "from", "also", "now", "then",
}


def _content(text: str | None) -> set[str]:
    import re

    if not text:
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def _pred_fold_ok(logs: list[dict]) -> bool:
    """True if every DELETE's content survives in some ADD text.

    Guards the DELETE+ADD -> UPDATE fold against lossy rewrites.
    Returns True when texts are unavailable (nothing to check).
    """
    deleted = [l.get("before_text") for l in logs if l["op"].upper() == "DELETE"]
    added = [l.get("after_text") for l in logs if l["op"].upper() == "ADD"]
    if not deleted or not added:
        return True
    add_tokens: set[str] = set()
    for a in added:
        add_tokens |= _content(a)
    for d in deleted:
        d_tokens = _content(d)
        if d_tokens and not d_tokens <= add_tokens:
            return False
    return True


def _canon_lenient(pairs: list[OpKey], *, fold_delete_add: bool = True) -> Counter:
    """Drop ADD/NOOP target ids, drop padding NOOPs, optionally fold DELETE+ADD."""
    norm: list[OpKey] = []
    for op, tid in pairs:
        if op in ("ADD", "NOOP"):
            norm.append((op, None))
        else:
            norm.append((op, tid))

    if any(op != "NOOP" for op, _ in norm):
        norm = [x for x in norm if x[0] != "NOOP"]

    c = Counter(norm)
    if not fold_delete_add:
        return +c
    n_add = c.pop(("ADD", None), 0)
    for key in [k for k in list(c) if k[0] == "DELETE"]:
        while c[key] > 0 and n_add > 0:
            c[key] -= 1
            n_add -= 1
            c[("UPDATE", key[1])] += 1
    if n_add:
        c[("ADD", None)] = n_add
    return +c  # drop zero counts


def ops_match_strict(gold_ops: list[dict], logs: list[dict]) -> bool:
    gold = Counter(_strict_key(g) for g in normalize_gold_ops(gold_ops))
    pred = Counter(_strict_key(p) for p in normalize_pred_ops(logs))
    return gold == pred


def ops_match_lenient(gold_ops: list[dict], logs: list[dict]) -> bool:
    g_raw, p_raw = normalize_gold_ops(gold_ops), normalize_pred_ops(logs)
    # (a) same ops, ignoring NOOP ids and padding NOOPs
    if _canon_lenient(g_raw, fold_delete_add=False) == _canon_lenient(
        p_raw, fold_delete_add=False
    ):
        return True
    # (b) same ops after folding DELETE+ADD -> UPDATE, only if pred is lossless
    if not _pred_fold_ok(logs):
        return False
    return _canon_lenient(g_raw) == _canon_lenient(p_raw)


# back-compat alias — now points at the lenient scorer
def ops_match(gold_ops: list[dict], logs: list[dict]) -> bool:
    return ops_match_lenient(gold_ops, logs)


def invalidation_flags(
    case: dict[str, Any],
    logs: list[dict],
    store_valid_ids: set[str],
) -> dict[str, float]:
    """true/false invalidation + silent-drop for this case.

    true_invalidation   control case where the gold DELETE target is gone.
    false_invalidation  a seed that should have stayed valid was deleted.
    fact_dropped        gold said the incoming fact is new information, but the
                        manager emitted only NOOP -> the fact entered nowhere.
                        Same end state as deletion, no audit trail.
    """
    typ = case["type"]
    gold_ops = case["gold_ops"]
    seed_ids = {s["id"] for s in case["seeds"]}
    ops = [l["op"].upper() for l in logs]
    deleted = {l["fact_id"] for l in logs if l["op"].upper() == "DELETE" and l.get("fact_id")}

    gold_delete_ids = {
        g["target_id"]
        for g in gold_ops
        if g["op"].upper() == "DELETE" and g.get("target_id")
    }
    must_live = seed_ids - gold_delete_ids

    true_inv = 0.0
    false_inv = 0.0
    if typ == "control" and gold_delete_ids:
        # UPDATE-as-merge onto the gold target also retires the stale claim
        merged = {
            l["fact_id"]
            for l in logs
            if l["op"].upper() == "UPDATE" and l.get("fact_id")
        }
        true_inv = (
            1.0
            if all(
                (tid not in store_valid_ids) or (tid in merged)
                for tid in gold_delete_ids
            )
            else 0.0
        )
        if deleted & must_live:
            false_inv = 1.0
    else:
        if deleted & must_live:
            false_inv = 1.0
        if typ == "condition":
            other = [
                s["id"]
                for s in case["seeds"]
                if s.get("project") != case["incoming"]["project"]
            ]
            if any(oid not in store_valid_ids for oid in other):
                false_inv = 1.0

    gold_has_add = any(g["op"].upper() == "ADD" for g in gold_ops)
    fact_dropped = (
        1.0 if gold_has_add and ops and all(o == "NOOP" for o in ops) else 0.0
    )

    return {
        "true_invalidation": true_inv,
        "false_invalidation": false_inv,
        "fact_dropped": fact_dropped,
        "n_deleted": float(len(deleted)),
    }


def fi_metrics(
    required_ids: set[str],
    final_valid_ids: set[str],
    ops_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """False invalidation against LoCoMo's own gold evidence — no new labels.

    A memory is `required` when some question's gold_ids names it, i.e. LoCoMo
    itself asserts the answer depends on it. If such a memory is not in the
    final valid store, the manager destroyed something the benchmark says is
    needed. That is sound by construction: no annotation, no judgement call.

    Two disappearance channels, kept apart on purpose:

      deleted   an explicit DELETE named this id. The claim is gone.
      absorbed  the id never entered the store because the manager answered
                UPDATE/NOOP instead of ADD. The *content* may survive inside
                the merged target, so this is an upper bound on merge loss,
                NOT destruction. Reported separately and never folded into
                fi_sound. The QA arm is what settles whether content survived.

    fi_sound is therefore an undercount of true false invalidation. That is the
    intent — a large undercount is harder to argue with than a generous count.
    """
    deleted_ids = {r["fact_id"] for r in ops_rows if r["op"] == "DELETE" and r.get("fact_id")}
    added_ids = {r["fact_id"] for r in ops_rows if r["op"] == "ADD" and r.get("fact_id")}
    incoming_ids = {r["incoming_id"] for r in ops_rows if r.get("incoming_id")}

    # A memory the manager was never offered cannot have been invalidated by it.
    # Matters whenever the ingest was subset (--limit-obs); a no-op on full runs,
    # but it keeps `n_unexplained_missing` meaningful as an integrity check —
    # nonzero there is a bug, not a finding.
    n_required_raw = len(required_ids)
    required_ids = required_ids & incoming_ids

    missing = required_ids - final_valid_ids
    deleted_required = missing & deleted_ids
    absorbed_required = {
        i for i in missing - deleted_required if i in incoming_ids and i not in added_ids
    }
    unexplained = missing - deleted_required - absorbed_required

    n_req = len(required_ids)
    n_del_ops = sum(1 for r in ops_rows if r["op"] == "DELETE")
    return {
        "n_required": n_req,
        "n_required_gold_total": n_required_raw,
        "n_required_never_ingested": n_required_raw - n_req,
        "n_required_present": len(required_ids & final_valid_ids),
        "gold_retention": round(len(required_ids & final_valid_ids) / n_req, 4) if n_req else None,
        "fi_sound": round(len(deleted_required) / n_req, 4) if n_req else None,
        "n_deleted_required": len(deleted_required),
        "n_absorbed_required": len(absorbed_required),
        "frac_absorbed_required": round(len(absorbed_required) / n_req, 4) if n_req else None,
        "n_unexplained_missing": len(unexplained),
        "fi_total_delete_ops": n_del_ops,
        "n_deleted_ids": len(deleted_ids),
        "frac_deletes_hitting_required": (
            round(len(deleted_required) / len(deleted_ids), 4) if deleted_ids else None
        ),
    }


def aggregate_write(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    n = len(rows)
    controls = [r for r in rows if r.get("type") == "control"]

    def _mean(rs: list[dict], key: str) -> float:
        return sum(r.get(key, 0.0) for r in rs) / len(rs)

    out: dict[str, Any] = {
        "n": n,
        "op_accuracy": sum(1 for r in rows if r.get("op_correct")) / n,
        "op_accuracy_strict": sum(1 for r in rows if r.get("op_correct_strict")) / n,
        "mean_true_invalidation": (_mean(controls, "true_invalidation") if controls else None),
        "mean_false_invalidation": _mean(rows, "false_invalidation"),
        "mean_fact_dropped": _mean(rows, "fact_dropped"),
        "n_control": len(controls),
    }
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["type"], []).append(r)
    out["by_type"] = {
        t: {
            "n": len(rs),
            "op_accuracy": sum(1 for r in rs if r.get("op_correct")) / len(rs),
            "op_accuracy_strict": sum(1 for r in rs if r.get("op_correct_strict")) / len(rs),
            "false_invalidation": _mean(rs, "false_invalidation"),
            "fact_dropped": _mean(rs, "fact_dropped"),
            "true_invalidation": (_mean(rs, "true_invalidation") if t == "control" else None),
        }
        for t, rs in by.items()
    }
    return out
