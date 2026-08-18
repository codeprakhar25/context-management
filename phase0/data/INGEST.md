# Ingest + storage (architecture family A)

## Architecture — LOCKED

**Family A:** flat **fact/claim bank** + ops `{ADD, UPDATE, DELETE, NOOP}` + human `path` / `project`.

Not Graphiti (temporal KG). Not Graphify (code AST graph).

```text
incoming claim → Manager (policy) → HierStore.apply_ops → ops_log
                                      ↑ dumb executor
```

| Layer | Role |
|-------|------|
| **HierStore** | Deterministic apply; soft-delete; embeds; audit |
| **Manager** | Chooses ops (`AlwaysADD`, `RuleV0`, later `LLMv0` / RL) |

Store **never** auto-merges on create.

---

## Op semantics

| Op | Effect |
|----|--------|
| ADD | INSERT new id |
| UPDATE | Same id: rewrite text, `revision++`, drop stale embed |
| DELETE | Soft `valid=0` default; `hard=True` for GC/tests |
| NOOP | No mutation; still logged |

UPDATE ≠ DELETE+ADD (Buddy/Scout failure mode).

---

## Conflict / write policies

| Policy | Who | Behavior |
|--------|-----|----------|
| `AlwaysADD` (= legacy `append_only` ingest) | Manager | Always ADD |
| `RuleV0` | Manager | Scoped cosine: NOOP / UPDATE / ADD |
| `LLMv0` | Manager | Scoped retrieve → gpt → ops; **neutral frozen prompt**; retry×1 then ADD; log `invalid_output_rate` |

**Prompt:** op definitions only — no preference rules / few-shots. Normative consolidate policy = train later.

**Control supersede gold:** `DELETE` old + `ADD` new. Complement gold: `UPDATE`. Condition-scoped: `ADD` (both scopes keep facts).

```bash
# conflict_v1 (~200, dev/test)
python3 scripts/build_conflict_v1.py
python3 scripts/tune_rulev0_dev.py          # thresholds on DEV only
python3 scripts/probe_conflict_v1.py --split test --managers AlwaysADD,RuleV0
python3 scripts/probe_conflict_v1.py --split test --managers LLMv0 --real-embed
```

Legacy meta key `CONFLICT_POLICY=append_only` remains for LoCoMo DB built with AlwaysADD. Prefer manager name in new runs.

---

## HierStore API

Path examples: `data/locomo/hierstore.sqlite`, probe temp DBs.

- `create` / `get` / `read_all` / `read_by_project`
- `update` / `delete(hard=False)`
- `apply_ops([Op(...), ...], manager=...)`
- `read_ops_log`
- embedding get/put/load matrix

Schema extras: `revision`, `parent_id`, `episode_id`, table `ops_log`.

---

## Conflict pack probe

```bash
python3 scripts/build_conflict_v0.py
python3 scripts/probe_conflict_v0.py
```

Data: `data/conflict_v0/` (complement, contradict, dupe, new_topic, wrong_project).  
Metrics: **op_accuracy**, bank size, `answer_in_hits` (QA proxy). Out: `runs/conflict_v0_probe/`.

---

## LoCoMo ingest (AlwaysADD baseline)

```bash
python3 scripts/ingest_locomo.py --replace
```

Observations → CREATE under `path=["project", sample_id]`. Streaming manager ingest = later.

---

## Flat / Hier retrieve (unchanged)

Same fact rows. Flat = ANN all; Hier = ANN in `project` (+ optional global).

```bash
python3 -m harness.run \
  --store data/locomo/hierstore.sqlite \
  --queries data/locomo/queries.jsonl \
  --ranker embed --methods flat,hier --k 5 --k-global 0
```

Primary metrics: **EM, F1, answer_in_hits, cost**. For write probes: **op_accuracy** first.  
`precision_in_scope` diagnostic only.

---

## Out of scope (for now)

Graphiti wrap, Graphify, MOVE op, RL training, full LoCoMo re-run under RuleV0 until conflict probes reviewed.
