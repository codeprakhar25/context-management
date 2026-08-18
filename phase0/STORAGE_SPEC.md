# Storage layer — locked design (grill 2026-08-02)

Focus: **storage first**. Retrieve optimize later. Trained placer later.
Format (sqlite vs md) = packaging; **sqlite source of truth**; md export optional.

---

## Unit & address

| Item | Lock |
|------|------|
| Leaf payload | **A → B**: plain text claim first; later + small structured header (source, mtime, confidence, …) |
| Address | **Hard tree (C)**: fact under exactly one parent path |
| MOVE | First-class; re-parent; leaf text unchanged |
| Leaf ops | **ADD / UPDATE / DELETE / NOOP** (MOVE separate) |
| DELETE | **Soft default, hard optional**; user confirm = product/UX later, not probe blocker |
| Same fact, many homes | **No** (single parent). Unsure → inbox, not copy |

## Tree growth & caps

| Item | Lock |
|------|------|
| Who places | LLM under schema (**D+B**): fixed tops; model picks children / MOVE / mkdir |
| mkdir | LLM may create folders under allowed parents |
| Depth | Hard cap **5** folder segments; sweet spot **~4**. Store rejects `>5`. Prompt states cap. Metric: cap-violation rate → 0 |
| Roots (v0) | **Match user-dir top folders** when snapshot exists; synthetic uses same names |
| Open top-level roots | **Later**, after path fidelity OK |
| Unsure path | **`inbox/` + low-confidence flag**; sweeper MOVE later |

## Prove storage works (order)

1. **A — CRUD integrity**: scripted ops; ids/paths/MOVE/mkdir/soft-hard delete match gold  
2. **B — Path fidelity**: gold path per fact; score place/MOVE/mkdir (no QA)  
3. **C — Retention stress**: gold_retention / FI_sound under a writer (tree-aware)  
4. QA / judge — last (confounds reader)

Gold paths: **synthetic tree first → user-dir snapshot (C)**. User = curator; freeze snapshot so desktop drift ≠ eval drift.

## Who writes in the gate

| Stage | Driver | Measures |
|-------|--------|----------|
| 1 | **Scripted oracle** | Executor correct (not “CRUD exists” — **hard-tree + MOVE + mkdir** contract) |
| 2 | **LLM placer** | Path/MOVE/mkdir policy vs gold |
| Later | SFT/train | Project-sensible trees inside rails (depth, roots, inbox) |

Scripted ≠ product. Product driver = LLM (then trained).

## Read path (defined now, optimize later)

| Mode | Rule |
|------|------|
| Lead | **Subtree**: active node + descendants + embed ANN inside that set |
| Ablation | **Flat**: global ANN; tree write-only |
| Active path (probes) | **Caller supplies** |
| Active path (later) | Optional **LLM router** question → path — only after B vs flat measured with gold paths |

## Write-time related facts

**Global ANN candidates, prefer same subtree.** Cross-path UPDATE/MOVE allowed; measure/penalize false cross-scope merges.

## Parked (grill later)

- Embed model / what fields get embedded  
- Dupe policy (same text, two adds)  
- Confidence field schema details  
- Hard-delete confirmation UX  
- md export as view (not primary executor)

---

## Build order

1. ~~Hard-tree executor~~ **done** (`harness/store.py`: dirs table, MKDIR/MOVE, depth≤5, fixed roots, `read_subtree`, soft/hard delete)  
2. ~~Scripted oracle~~ **done** — `data/storage_oracle/cases.jsonl` + `scripts/run_storage_oracle.py` (10/10 pass)  
3. LLM placer path fidelity — `scripts/probe_llm_placer.py` + `data/storage_oracle/place_tasks.jsonl`  
4. ~~User-dir snapshot~~ **v0 frozen** — `data/user_dir_snap/` (`sources.json` → `snapshot_user_dir.py`); viz `runs/tree_viz_user_dir.txt`; SFT seed `place_tasks_from_snap.jsonl`  
5. Placer + SFT  
   - Fair smoke user-dir+dirs: exact **0.43**  
   - OpenAI FT blocked → Modal Qwen-1.5B collapsed (personal-tree SFT)  
   - **Multi-tree synth smoke (B+D):** `build_multitree_synth.py` + `eval_multitree_smoke.py`  
     - 10 trees / 7 train / 3 holdout; twin flat→subtree intrusion **1.0→0.0**  
     - gpt-4o holdout+dirs: exact **0.52**, soft/branch **~0.99**  
     - Next: mid-scale synth → 7–8B LoRA (not more personal-snap SFT)  

**Gate 1–2 (executor):** `python3 scripts/run_storage_oracle.py` → `all_pass: true`.  
**Gate 3 (placer):** `python3 scripts/probe_llm_placer.py --model gpt-4o --out runs/llm_placer_gpt4o`  
Lead score: **path_soft** (exact or prefix); **branch_ok** = same root. `path_exact` diagnostic.  
Rescore without API: `--rescore-only --out runs/llm_placer_gpt4o`  

**Subtree vs flat (B):** `build_confusable_tree.py` → `probe_subtree_vs_flat.py`  
Viz: `viz_tree.py` (place_tasks gold / `--store` / LoCoMo summary)  
Closed-vocab placer (A): parked — open naming expected.
