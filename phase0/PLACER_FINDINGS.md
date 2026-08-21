# Placer — record of runs and results

**Date:** 2026-08-03

Placer = the component that decides which folder of a hierarchical store a new
note goes in (`ADD` to path `["work","slm-lab","research"]`).

Hypothesis tested: a small model fine-tuned on a specific tree learns that tree's
filing conventions better than a frontier model can be prompted into.

Result: a retrieve-then-pick cascade (embedding shortlist → gpt-4o) beats the
fine-tuned 8B on both evaluation regimes, with no training.

```
                          folders SEEN      folders UNSEEN
  cascade                      0.643             0.419
  LoRA 8B (fine-tuned)         0.537             0.363
  kNN alone                    0.524             0.000
  gpt-4o flat                  0.286             0.354
```

**This result is corpus-A-specific and does not generalize** — read section
2's corpus B / corpus A′ subsections and section 3's conclusions before
citing anything above as a general finding. The honest paper-level framing,
settled after adversarial review (2026-08-19): *observed target-folder
support is a major evaluation axis for note routing; method rankings vary
across support regimes, and cross-corpus reversals reveal real sensitivity
to corpus construction that this project has not fully explained.*

---

## 1. Timeline

### 1a. Synthetic SFT (before 2026-08-03)

50 generated trees (`lab/life/dump`, `forecast/models/results`, …), 1200 train /
300 holdout, LoRA on llama-3.1-8b via Fireworks. Variants: `mid`, `mid_hard`
(paraphrase partition), `domain_shift`.

Reported: LoRA 0.747 vs gpt-4o 0.553 on hard holdout; 0.315 vs 0.433 on a real
tree.

### 1b. Decomposing the synth holdout

Checked gold-label class balance. 60% of synth gold paths end in the literal
segment `results`; all 50 trees share one 11-directory shape.

| gold class | n | LoRA hard | gpt-4o |
|---|---|---|---|
| ends in `results` | 180 | 0.989 | 0.356 |
| depth-1 | 10 | 0.900 | 0.600 |
| other | 110 | 0.336 | 0.873 |
| overall | 300 | 0.747 | 0.553 |

gpt-4o's misses on the `results` class are mostly prefix hits (soft 0.994) — it
predicts the parent folder.

The overall LoRA win comes from the 60% majority class. On the 110 items needing
a judgement, gpt-4o is 2.6× better. 0.336 (synth, `other`) ≈ 0.315 (real tree).

Script: `scripts/decompose_placer_holdout.py`

### 1c. Untrained base-8B arm

No untrained baseline existed, so SFT's effect was unmeasured. Fireworks dropped
`llama-v3p1-8b-instruct` from serverless and the LoRA deployments were stopped,
so this ran through OpenRouter (same base weights, temp 0, byte-identical prompt;
different serving stack).

| arm | synth `other` (n=110) | v1 real tree (n=127) |
|---|---|---|
| base-8B, no SFT | 0.273 | 0.142 |
| LoRA hard | 0.336 (p=0.25, ns) | 0.315 (p=1.0e-5) |
| gpt-4o | 0.873 | 0.433 |

Conditioned on the predicted folder existing:

| arm | valid preds | exact within valid |
|---|---|---|
| base-8B | 64/127 | 0.281 |
| LoRA hard | 118/127 | 0.339 |
| gpt-4o | 124/127 | 0.444 |

The real-tree gain is schema compliance — base-8B invents non-existent folders
49.6% of the time. Within-valid accuracy is not significantly different.

### 1d. Audit of the training procedure

| check | result |
|---|---|
| train prompt == eval prompt | byte-identical (system and user) |
| train/holdout tree overlap | 0 of 40 vs 10 trees |
| holdout note text seen in train | 0/300 |
| user-dir data present in train | 0/126 texts, 0/53 paths |
| job state | `COMPLETED`, status OK |

Two defects found:

1. Overfit; deployed checkpoint was the worse one. `mid_hard` train loss → 0.0000
   by step 37 of 76; eval loss 0.0184 at step 38 → 0.1421 at step 76.
   `earlyStop` was false. (`mid`: eval 0.0001 → 0.0000 — holdout was memorizable.
   `domain_shift`: 0.0596 → 0.0569, flat.)
2. `confidence` output is dead by construction. All 1200 assistant targets are
   `"confidence": 0.9`. The system prompt says *"If unsure, use ["inbox"] and low
   confidence"* — no example demonstrates it. Still unfixed.

The synth LoRAs were never trained on user directories.
`scripts/build_sft_placer.py` defaults to `--tasks data/user_dir_snap/...`, but
the jobs used `multitree_synth_*`. "user-dir" was always a transfer eval.

### 1e. Rebuilding the eval

v1 was n=127 (`work`=103, `personal`=1, `inbox`=23). A 20% val split gives 26
items.

Widened `sources.json` to the rest of the machine → 1318 raw tasks / 365 folders.
Two problems, both from pointing at whole repo roots:

- 12% of gold paths silently truncated by `max_depth: 5` (natural depths to 11).
- 22% of tasks shared exact text with another task (copied/vendored files) —
  identical content in two folders.

Both filtered → 870 clean tasks, 170 folders. `personal` root dropped:
`~/personal` is 23 tax/bank PDFs and 1 markdown file.

```
v2: 870 clean tasks · 170 folders · 2 roots (work 788, inbox 82)
    majority-class baseline 0.053 · text ~492 chars/note
    existing_dirs in every prompt: 365 folders ≈ 6.3k tokens
```

Scripts: `scripts/snapshot_user_dir.py`, `scripts/build_user_dir_split.py`

### 1f. Two splits

| split | construction | tests |
|---|---|---|
| item | stratified inside each folder; 576 train / 294 val; every val folder seen in training | learning this tree's conventions |
| folder | whole folders held out; 517 train / 353 val; zero val folders seen in training | placing into a folder never seen used |

In both, `existing_dirs` lists all 365 folders, so held-out folders stay
selectable.

### 1g. Training runs

```
epochs           1        (was 2)
loraRank         16       lr 1e-4      base llama-v3p1-8b-instruct
maxContextLength 16384    (was 0/auto)
earlyStop        removed — Fireworks rejects it as deprecated
```

Rows are ~9,010 tokens because 365 folders are injected into every prompt (synth
rows were ~600). Truncation checked four ways: server read-back = 16384; dataset
`estimatedTokenCount` 5,189,800 / 576 = 9,010 tok/row; trainer processed
5,178,927 tokens = 99.8% of the dataset; `effective_batch_fill_ratio` 1.0 (0.087
on the synth run).

| run | train loss end | eval loss |
|---|---|---|
| item split | 0.079 | 0.075 |
| folder split | 0.044 | 0.140 |
| (old `mid_hard`) | 0.0000 | 0.018 → 0.142 |

Job IDs: `placer-userdir-v2-llama31-8b-20260803-1248`,
`placer-userdir-v2fold-llama31-8b-20260803-1401`

---

## 2. Results

All arms use one prompt, parser and scorer (`scripts/eval_fireworks_placer.py`,
`--provider {openai,openrouter,fireworks}`). `exact` = full path match.
`soft` = exact or prefix-related.

### Item split — val folders seen in training (n=294, paired)

| arm | exact | soft | invalid dir | too deep | too shallow | tokens/call |
|---|---|---|---|---|---|---|
| cascade note@20 + descriptions | 0.660 | 0.725 | — | — | — | 1,513 |
| cascade note@20 | 0.643 | 0.714 | — | — | — | 948 |
| kNN k=8 | 0.554 | 0.626 | — | — | — | 0 |
| LoRA v2 | 0.537 | 0.582 | 0.7% | 1 | 12 | 9,010 |
| kNN k=5 | 0.524 | 0.605 | — | — | — | 0 |
| gpt-4o flat + descriptions | 0.306 | 0.418 | — | — | — | 14,963 |
| gpt-4o flat | 0.286 | 0.463 | 1.4% | 40 | 12 | 9,010 |
| base-8B | 0.051 | 0.105 | 24.8% | 6 | 10 | 9,010 |

```
flat+desc    vs flat        : +29 / -23    p = 0.489    ns
cascade      vs LoRA        : +53 / -22    p = 0.00045
cascade      vs kNN k=5     : +77 / -42    p = 0.0017
cascade      vs gpt-4o flat : +114 / -9    p = 2.7e-24
cascade+desc vs cascade     : +19 / -14    p = 0.487    ns
LoRA         vs kNN k=5     : +64 / -60    p = 0.788    ns
LoRA         vs gpt-4o flat : +90 / -16    p = 1.1e-13
```

### Folder-disjoint split — val folders never seen (n=353, paired)

| arm | exact | soft | candidate recall | invalid dir | tokens/call |
|---|---|---|---|---|---|
| cascade path@50 | 0.419 | 0.496 | 0.844 | — | 1,668 |
| cascade path@20 | 0.385 | 0.470 | 0.677 | — | 949 |
| LoRA fold | 0.363 | 0.470 | — | 2.0% | 9,010 |
| gpt-4o flat | 0.354 | 0.484 | — | 2.8% | 9,010 |
| base-8B | 0.059 | 0.139 | — | 22.1% | 9,010 |
| kNN (any k) | 0.000 | 0.15 | 0.000 | — | 0 |

```
cascade@50 vs gpt-4o flat : +45 / -22    p = 0.0067
cascade@50 vs LoRA fold   : +53 / -33    p = 0.040
cascade@50 vs cascade@20  : +17 / -5     p = 0.017
LoRA fold  vs gpt-4o flat : +37 / -34    p = 0.813    ns
LoRA fold  vs base-8B     : +114 / -7    p = 5.1e-26
```

### Retrieval x grounding (item split, n=294, paired)

Both factors varied independently over the same prompt, parser and scorer.

| | no descriptions | + descriptions | grounding effect |
|---|---|---|---|
| flat (all 365 folders) | 0.286 | 0.306 | +2.0 pt, p = 0.489 ns |
| retrieved shortlist (20) | 0.643 | 0.660 | +1.7 pt, p = 0.487 ns |
| retrieval effect | +35.7 pt, p = 2.7e-24 | +35.4 pt, p = 1.7e-23 | |

The two effects are separable. Retrieval is worth ~35 pt in both rows; folder
descriptions are within noise in both columns.

PaperRouter-Agent (arXiv 2607.11564) reports Recall@1 0.39 -> 0.61 for a
Planner/Retriever/Inspector/Reflector agent against a single-shot baseline that
lists the whole hierarchy. That comparison varies retrieval and content grounding
together. In this decomposition the retrieval half accounts for the gain.

Scope of that statement: the descriptions here are gpt-4o-mini one-liners built
from at most 4 training notes truncated to 300 chars. PaperRouter's Inspector
reads member documents at inference time. This tests compressed folder content,
not full member text.

### Candidate recall (embeddings only, `scripts/candidate_recall.py`)

| source | item @20 | folder @20 | folder @50 |
|---|---|---|---|
| `note` — folders of nearest training notes | 0.983 | 0.000 | 0.000 |
| `path` — folders whose path string embeds nearest | 0.721 | 0.677 | 0.844 |
| `union` (interleaved) | 0.959 | 0.572 | 0.745 |

`note` cannot reach a folder holding no training note. `path` reaches any folder.
The union is below the better single source on both splits.

Shortlist of 20 ≈ 236 tokens vs 6,251 for the full folder list.

### Accuracy given the gold folder was in the shortlist

```
item,   note@20              recall 0.983  ->  0.654
item,   note@20 + descriptions recall 0.983 -> 0.671
folder, path@20              recall 0.677  ->  0.569
folder, path@50              recall 0.844  ->  0.497
```

### Supporting breakdowns

Accuracy by number of training examples in the gold folder (item split):

| train examples | n | LoRA | kNN k=5 | gpt-4o |
|---|---|---|---|---|
| 1–2 | 18 | 0.222 | 0.444 | 0.278 |
| 3–9 | 105 | 0.371 | 0.381 | 0.200 |
| 10+ | 171 | 0.673 | 0.620 | 0.339 |

Concentration: 83% of item-split val is one repo (`work/openclaw`, n=245, LoRA
0.612). `work/ml-resarch` (n=25): LoRA 0.320, gpt-4o 0.240. `inbox/misc-cc`
(n=24): LoRA 0.000, gpt-4o 0.167.

kNN k-sweep (item split): 1→0.374, 3→0.429, 5→0.524, 8→0.554, 12→0.548,
20→0.486, 40→0.446.

Leak checks on the item-split LoRA result: 0/294 exact text leak train→val,
0 shared source files, median near-duplicate similarity to a same-folder train
note 0.30, zero pairs >0.8.

Folder descriptions were generated by gpt-4o-mini from training files only
(`scripts/gen_folder_descriptions.py`, 170 folders, $0.009). They are clean on
the item split, where 61/61 val folders hold training files. On the
folder-disjoint split 0/72 val folders hold any training file, so a
content-derived description would encode the answer; descriptions were not run
there.

### Corpus B — 27 public vaults, rebuilt at max_depth=8

4,199 clean tasks, 1,469 item-val, 1,879 folder-val. One store per vault.
Depth is a per-corpus property: A is capped at 5, B at 8. Raising B 5→8
recovered 424 tasks (quanru/lifeos 84→225, pm-brain 53→162,
Cybersecurity-Notes 58→157). 366 notes dropped as duplicate text
(AgriciDaniel/flow −126, obsidian-slr −93, pm-brain −88). 9 vaults hit the
250-note cap. Folder-disjoint split is clean: 1,879/1,879 val items have
zero training notes in the gold folder.

Item-split occupancy (train notes in gold folder) is the coverage curve the
claim needs, and B actually has mass at the sparse end:

| train examples | corpus A | corpus B |
|---|---|---|
| 1–2 | 18 | 258 |
| 3–9 | 105 | 653 |
| 10+ | 171 | 558 |

Candidate recall (`runs/vaultB_recall/pooled.json`, embeddings only, mean
38.9 folders/vault):

| source | item @20 | folder @20 | folder @50 |
|---|---|---|---|
| `note` | 0.993 | 0.000 | 0.000 |
| `path` | 0.935 | 0.929 | 0.996 |
| `union` | 0.993 | 0.883 | 0.986 |

Folder path@20 is 0.929 vs 0.677 on A because the trees are small, not
because the embedder got better. path@50 ≈ the full folder list (only 5
vaults have >50 folders) — do not run a path@50 cascade arm.

kNN (`runs/vaultB_knn/pooled.json`):

| k | item exact | folder exact |
|---|---|---|
| 1 | 0.678 | 0.000 |
| 3 | 0.711 | 0.000 |
| 5 | **0.716** | 0.000 |
| 8 | 0.706 | 0.000 |
| 12 | 0.686 | 0.000 |

Phase 2 gpt-4o grid, 27/27, `parse_fail=0` (`runs/vaultB_{cascade,flat}/pooled.json`):

| arm | item (n=1,469) | folder (n=1,879) |
|---|---|---|
| kNN k=5 | **0.716** | 0.000 |
| cascade | 0.592 (recall 0.993) | **0.583** (recall 0.929) |
| gpt-4o flat | 0.516 | 0.543 |

Item occupancy (kNN column is k=1 — holdout file writes only the first k):

| train notes in gold | n | kNN k=1 | cascade | flat |
|---|---|---|---|---|
| 1–2 | 258 | 0.508 | **0.628** | 0.547 |
| 3–9 | 653 | 0.655 | **0.729** | 0.662 |
| 10+ | 558 | **0.783** | 0.414 | 0.332 |

Cascade beats kNN on 15/27 vaults; the item-pool loss is three dense
vaults (TheRoadOfSO 0.020 vs 0.788, anthonyamar 0.312 vs 0.896,
ManadayM 0.379 vs 0.947). TheRoadOfSO is opaque hex folder names —
81/99 cascade preds are the parent. `branch_ok` is vacuous (single
root `vault`). Folder exact is 0.000 at every kNN k, as required.

gpt-4o spend: cascade $10.30 + flat $15.87. Mean tok/call 1,092
(cascade) vs 1,725 / 1,791 (flat).

Second picker family, Llama 3.3 70B via OpenRouter, same cascade recipe
(`runs/vaultB_cascade_llama70b/pooled.json`, 27/27 both splits, parse_fail
0/1,469 item, 3/1,879 folder):

| arm | item (n=1,469) | folder (n=1,879) |
|---|---|---|
| kNN k=5 | **0.716** | 0.000 |
| gpt-4o cascade | 0.592 | **0.583** |
| Llama 70B cascade | 0.523 | 0.499 |
| gpt-4o flat | 0.516 | 0.543 |

Llama trails gpt-4o on the same cascade by 6.9pt item / 8.4pt folder, but
the ranking against flat does not hold cleanly: Llama cascade beats gpt-4o
flat on item (0.523 vs 0.516) but not folder (0.499 vs 0.543) — no
Llama-flat arm was run, so "cascade beats flat" for this picker is
unverified on the folder split, only inferred from the item split. Spend
~$0.50 (OpenRouter, no per-call retry cost since 0 item parse_fail).

LoRA on corpus B, both splits, same recipe as corpus A (llama-v3p1-8b, 1
epoch, lr 1e-4, loraRank 16, maxContextLength 16384, no earlyStop), trained
per-vault-split-pooled (`scripts/build_sft_placer_vaults.py`: each row keeps
its own vault's `existing_dirs`/`MAX_DEPTH`, only rendering is pooled) —
`runs/vaultB_lora_{item,fold}/pooled.json`, 27/27 vaults, parse_fail 0/0:

| arm | item (n=1,469) | folder (n=1,879) |
|---|---|---|
| **LoRA** | **0.7556** | 0.4747 |
| kNN k=5 | 0.716 | 0.000 |
| gpt-4o cascade | 0.592 | **0.583** |
| Llama 70B cascade | 0.523 | 0.499 |
| gpt-4o flat | 0.516 | 0.543 |

**This reverses corpus A's headline finding.** On A the cascade beat the
fine-tuned LoRA on both splits (0.643 vs 0.537 item, 0.419 vs 0.363 folder).
On B, LoRA wins item split outright and is mid-pack on folder-disjoint.
"Cascade beats fine-tuning, no training needed" is not corpus-general — it
held on A's single large mixed-domain tree (365 folders) and inverts on B's
27 small single-person trees (mean 39 folders).

**Correction (2026-08-19, caught in adversarial review before drafting):**
the table below originally read "LoRA beats every arm at every bucket," but
it compared LoRA against kNN **k=1** while the headline kNN number elsewhere
in this doc is **k=5** — `knn_placer_baseline.py` only writes per-item
`holdout_results.jsonl` for the first k requested, so the occupancy
breakdown silently used whichever k happened to be listed first. Reran kNN
with k=5 as the sole k (`runs/vaultB_knn_k5only/`, item split, $0) to get a
consistent per-item comparison:

| train notes in gold | n | LoRA | kNN k=5 | cascade |
|---|---|---|---|---|
| 1–2 | 258 | **0.667** | 0.457 | 0.628 |
| 3–9 | 653 | **0.776** | 0.718 | 0.729 |
| 10+ | 558 | 0.772 | **0.833** | 0.414 |

The claim was false, and the corrected picture is more interesting than the
one it replaces: **kNN beats LoRA by 6pt in the dense bucket**, not the
near-tie the k=1 table showed. LoRA wins sparse and mid, kNN wins dense,
cascade never wins any bucket on this split — three methods, three
non-overlapping regimes, which is a cleaner statement of the coverage-curve
thesis than "LoRA wins everywhere" was. Eval loss at 1 epoch was 0.0247
(item) / 0.0691 (folder) — lower than corpus A's 0.075/0.140 on the same
recipe, consistent with a narrower per-vault label space, not leakage (val
notes held out, zero text overlap — see also the cross-vault leakage check
below, run after this correction).

**Tree-size "mechanism" — tested and rejected (2026-08-19).** The original
version of this section reported a monotone item-pooled tercile pattern
(+.245/+.178/+.069 small→large folders, all "significant") as dose-response
evidence that smaller trees favor LoRA. An adversarial pre-draft review
(Codex, `/codex consult`) caught the actual problem: **items within a vault
aren't independent draws**, so pooling 1,469 items into three bins and
running McNemar on each bin is pseudo-replication — the real number of
independent units is 27 vaults (or 43 pooling in corpus A′), not 1,469
items. The original *vault-level* correlation (r=0.13, reported and then
set aside in favor of the item-pooled version) was the correct analysis all
along, just underpowered-looking without a real significance test. Ran one:
a permutation test on Spearman correlation between log(folder count) and
per-vault (LoRA − cascade) margin, the proper unit, both corpora:

```
corpus B    n=27  spearman(log-folders, margin) = +0.029   p=0.887
corpus A′   n=16  spearman(log-folders, margin) = -0.115   p=0.669
pooled      n=43  spearman(log-folders, margin) = +0.034   p=0.826
```

Null in both corpora, opposite-signed between them, nowhere close to
significant. **The tree-size dose-response claim does not survive at the
correct unit of analysis and is retracted.** The item-pooled tercile
pattern was real in the sense that it reproduced across two corpora, but it
was measuring something correlated with vault identity, not folder count
per se — plausibly candidate-set size (LoRA sees the full folder list,
cascade sees a top-20 shortlist; in small trees those are nearly the same
list, in large trees they aren't, which is confounded with tree size by
construction — see open items) or something about which vaults got sampled
into which tercile. What survives: LoRA beats cascade on corpus B and A′
overall (see per-corpus tables), a real and repeated result; *why* is open.

Paired McNemar, LoRA vs gpt-4o cascade, matched by item id (exact test,
`math.comb`):

```
item   n=1469  LoRA .7556 vs cascade .5916  +319/-78   p=1.1e-35
folder n=1879  LoRA .4747 vs cascade .5833  +147/-351  p=2.7e-20
```

Item win is broad, not a few vaults carrying the pool: LoRA wins 19/27
vaults outright, cascade 6, 2 tied. Folder loss is narrower at the vault
level (cascade 13, LoRA 10, 4 tied) despite the large pooled p-value —
worth a footnote that the aggregate gap is not uniformly distributed across
vaults on that split.

**Caveat on the p-values above:** these item-pooled McNemar tests have the
same pseudo-replication issue as the retracted tercile analysis — 1,469
items cluster inside 27 vaults, not 1,469 independent draws. The p-values
are almost certainly too small; the trustworthy summary of "is this real"
is the **vault-win-count** (19/27, 6/27, 2 tied), not the item-level p.
Report both, lead with the vault count in the paper.

Hardware note: item-split LoRA was served on H200 via Fireworks' vetted
`rft-llama-v3p1-8b-instruct` preset. Mid-session, H200 dedicated capacity
went `no available capacity` for the fold-split deploy (confirmed
industry-wide H200/B200 shortage, not Fireworks-specific — lead times
36–52 weeks as of 2026-08). Fold split was served on H100 instead, freeform
config (no preset, explicit `precision: BF16`, `maxContextLength: 16384`)
— same numerics, unvalidated preset-equivalence. Both splits' item and
folder numbers are otherwise from the identical training run; only the
inference deployment differs.

Spend: item train ~$2.63, fold train ~est. similar (job cost field showed
$2.63 for item; fold job cost not separately captured — check
`runs/fireworks_placer-vaultb-fold-*_create.json` sibling job-status call if
needed). Dedicated deploy time: item ~15 min H200, fold ~9 min H100 CREATING
+ eval + teardown, both confirmed torn down (`deleteTime` set, deployment
list empty after).

`union` is worse than `path` alone on the folder-disjoint split (0.883 vs
0.929 @20): note candidates are 0.000 there by construction, so mixing them
in only displaces path candidates out of the shortlist. The candidate source
has to be switched by regime, not merged.

kNN across all 27 vaults (`runs/vaultB_knn/pooled.json`, embeddings only, $0):

| k | item exact | folder exact | folder soft |
|---|---|---|---|
| 1 | 0.678 | 0.000 | 0.184 |
| 3 | 0.711 | 0.000 | 0.180 |
| 5 | **0.716** | 0.000 | 0.182 |
| 8 | 0.706 | 0.000 | 0.179 |
| 12 | 0.686 | 0.000 | 0.173 |

kNN peaks at k=5 on B, not k=8 as on A. Item-split kNN is 0.716 on B vs
0.524 on A — B's trees are 39 folders, A's are 170, so this is a difficulty
difference between corpora, not a method result. It matters for the grid:
the cascade's headroom above kNN on B is at most 0.993 − 0.716.

The coverage curve replicates on B, on 4× the items in the sparse buckets:

| train notes in gold folder | n | kNN k=5 exact |
|---|---|---|
| 1–2 | 258 | 0.508 |
| 3–9 | 653 | 0.655 |
| 10+ | 558 | 0.783 |
| 0 (folder-disjoint) | 1,879 | 0.000 |

`branch_ok` is degenerate on corpus B and must not be reported. Every gold
path starts `["vault", "<repo-slug>", …]` because each vault is built as one
logical root, so `same_root` is 1.000 by construction for all arms. The first
user-meaningful segment is index 2; scored there, kNN k=5 gets 0.875 (item)
and 0.583 (folder-disjoint), 97.8% of items being depth ≥ 3.

---

### Corpus A′ — 16 public software repos, item split only

Confirmatory second-genre check on the tree-size mechanism found in corpus B's
tercile analysis. Same construction as B (`scripts/build_vault_corpus.py`,
`select_vaults.py`-style spread-by-size selection from the 95 repos the vault
classifier flagged as software, pinned commits). 25 selected, 9 failed to
clone (`git fetch --depth 1` timed out at 180s — `fetch_vaults.py` originally
had no timeout handling and crashed the whole batch on the first hang;
fixed to catch `TimeoutExpired` per-repo and continue). 16 vaults built,
2,644 clean tasks, 797 item-val, dir range 10–164 (wider than B's 14–129).
Item split only, per scope — folder-disjoint/flat/second-picker weren't
needed for the mechanism check, though the folder split built for free
alongside it (`fold_train`/`fold_val` exist if ever wanted).

| arm | item exact (n=797) |
|---|---|
| LoRA | 0.716 |
| kNN k=12 | 0.732 |
| gpt-4o cascade | 0.552 |
| Llama 70B cascade | 0.459 (parse_fail 8/797, spread across 5 vaults, not systemic) |

Cascade loses to both retrieval-based methods here too — the corpus-A
"cascade beats fine-tuning" finding fails to replicate a second time, on a
different genre.

**Tercile dose-response — RETRACTED (2026-08-19).** This section originally
reported the item-pooled tercile margin below as the paper's strongest
evidence for a tree-size mechanism, replicated across both public corpora:

```
                small        mid          large
corpus B:      +0.245       +0.178       +0.069     (folders 14–27 / 27–45 / 45–129)
                p=2.6e-25    p=1.5e-15    p=0.0024
corpus A′:     +0.249       +0.162       +0.082     (folders 16–31 / 31–66 / 67–164)
                p=1.1e-12    p=9.4e-06    p=0.014
```

Same pseudo-replication problem as corpus B's version of this analysis
(items pooled across vaults, not independent) — see the correction in the
corpus B section above for the full explanation and the vault-level
permutation test that replaces it (n=27 B, n=16 A′, n=43 pooled — all null,
p=0.67–0.89). The numbers above are real and reproduce, but they are not
evidence for tree size specifically; kept here struck through rather than
deleted so the record shows what was claimed and why it didn't hold up.

Spend: kNN/recall $0, gpt-4o cascade ~$3 (797 items), Llama ~$0.30, LoRA
train $2.59 + ~8 min H200 dedicated deploy (no capacity stall this time).

---

### Lexical baselines — BM25 and majority-class (2026-08-19)

Added after adversarial pre-draft review asked why a supposedly
retrieval-heavy task had no lexical retrieval arm and no trivial floor.
`scripts/eval_lexical_baselines.py`, k=5, item split on all three corpora,
$0 spend (no API calls). Output: `runs/lexical_baselines/summary.json`.

| corpus | n | majority | BM25 k=5 |
|---|---|---|---|
| A  | 294 | 0.061 | 0.442 |
| B  | 1,469 | 0.236 | 0.646 |
| A' | 797 | 0.356 | 0.666 |

By occupancy bucket (train notes already in the gold folder):

| corpus | bucket | n | majority | BM25 |
|---|---|---|---|---|
| A  | 1-2 | 18 | 0.000 | 0.333 |
| A  | 3-9 | 105 | 0.000 | 0.343 |
| A  | 10+ | 171 | 0.105 | 0.515 |
| B  | 1-2 | 258 | 0.000 | 0.403 |
| B  | 3-9 | 653 | 0.064 | 0.636 |
| B  | 10+ | 558 | 0.545 | 0.771 |
| A' | 1-2 | 119 | 0.000 | 0.311 |
| A' | 3-9 | 304 | 0.092 | 0.579 |
| A' | 10+ | 374 | 0.684 | 0.850 |

Two things this changed in the paper's framing:

1. **BM25 alone beats the gpt-4o cascade on corpus B** (0.646 vs 0.592) and
   on A' (0.666 vs 0.552). A pure lexical index with zero API spend outruns
   an embedding shortlist plus an LLM pick per note. That strengthens the
   "retrieval-only baseline is mandatory" recommendation considerably —
   it's not just embedding kNN, even BM25 is enough.
2. **Majority-class is not a trivial floor on the dense bucket.** B 10+
   majority is 0.545 and A' 10+ is 0.684, i.e. above gpt-4o flat (0.332 on
   B 10+). Dense buckets are partly a label-skew artifact, which is more
   evidence that occupancy strata encode folder/vault identity and are not
   a clean causal axis. Corpus A's majority floor is 0.061 because its
   tree is much flatter in label mass.

Note A' has a bucket breakdown here (119/304/374) even though the LoRA /
cascade / kNN arms on A' were pooled-only — these two baselines are free to
stratify, the LLM arms were not re-run per bucket.

---

### Occupancy sensitivity — do 3 vaults carry the crossing? (2026-08-21)

Raised while rewriting the paper: cascade's dense-bucket collapse (0.414)
is concentrated in three vaults that PLACER_FINDINGS already flagged
(TheRoadOfSO 0.020 vs 0.788, anthonyamar 0.312 vs 0.896, ManadayM 0.379
vs 0.947). A pooled crossing driven by three pathological vaults is a weak
result, so the whole item-split table was rebuilt from per-item artifacts
with them removed. `scripts/occupancy_sensitivity.py`, $0 (reads
`runs/vaultB_{flat,knn_k5only,cascade,lora_item}/*/holdout_results.jsonl`
joined to `data/vaults_build/*/{train,val}.jsonl` for occupancy).
Output: `runs/occupancy_sensitivity/summary.json`.

All 27 vaults — reproduces the published table exactly, independent path:

| occ | n | majority | flat | BM25 | kNN | cascade | LoRA |
|---|---|---|---|---|---|---|---|
| 1-2 | 258 | 0.000 | 0.547 | 0.403 | 0.457 | 0.628 | 0.667 |
| 3-9 | 653 | 0.064 | 0.662 | 0.636 | 0.718 | 0.729 | 0.776 |
| 10+ | 558 | 0.545 | 0.332 | 0.771 | 0.833 | 0.414 | 0.772 |
| all | 1,469 | 0.235 | 0.516 | 0.646 | 0.716 | 0.592 | 0.756 |

Excluding the 3 outlier vaults (drops 40% of the dense bucket):

| occ | n | majority | flat | BM25 | kNN | cascade | LoRA |
|---|---|---|---|---|---|---|---|
| 1-2 | 240 | 0.000 | 0.542 | 0.408 | 0.454 | 0.629 | 0.671 |
| 3-9 | 604 | 0.070 | 0.662 | 0.627 | 0.710 | 0.732 | 0.781 |
| 10+ | 335 | 0.510 | 0.445 | 0.708 | 0.776 | 0.621 | 0.806 |
| all | 1,179 | 0.181 | 0.576 | 0.606 | 0.677 | 0.679 | 0.766 |

Three conclusions, and they do not all point the same way:

1. **The crossing SURVIVES.** Cascade beats kNN by +17.5pt at 1-2 (0.629
   vs 0.454) and loses by -15.5pt at 10+ (0.621 vs 0.776). Full-corpus
   swing 59pt -> 33pt trimmed, but the sign flip is intact. This is the
   paper's central claim and it is robust.
2. **"Retrieval-only beats the cascade in aggregate" does NOT survive.**
   Full: kNN 0.716 vs cascade 0.592 (+12.4pt). Trimmed: kNN 0.677 vs
   cascade 0.679 — a dead tie, and BM25 (0.606) now *loses* to cascade.
   That claim was carried entirely by the 3 vaults. Paper reworded: the
   reason to require a retrieval baseline is that omitting it can invert
   a conclusion, not that it reliably wins.
3. **The parent-naming pathology is confined to the outliers.** Share of
   cascade preds that are a proper prefix of gold, dense bucket: 14.3%
   all-vaults vs **2.4%** trimmed (sparse 6.2/6.7%, mid 4.1/2.6%). So it
   is a data artifact of vaults with uninformative folder names
   (TheRoadOfSO's hex dirs), not a general retrieve-then-pick failure.
   Good news for the method, bad news for corpus B's cleanliness.

Also unstable and now flagged in the paper: **who wins the dense bucket**.
Full corpus kNN 0.833 > LoRA 0.772; trimmed LoRA 0.806 > kNN 0.776. Not
settled by this data. What is settled is that cascade is not the winner.

Majority-class floor added to the table for the first time and it is not
trivial: **0.545 at B 10+** (above flat gpt-4o's 0.332), 0.510 trimmed.
Dense buckets are partly label skew, reinforcing that occupancy strata
encode folder/vault identity and are not a clean causal axis.

---

### Cross-vault leakage check (2026-08-19)

Raised in adversarial review: corpus B/A′ train sets are pooled across
vaults before an SFT job, but exact-duplicate filtering happens *within*
each vault's own snapshot (`build_user_dir_split.py`) — a validation note
in one vault could closely resemble a training note in a *different* vault
and never get caught. Checked directly using the embed cache already built
for the recall gates (`runs/_embed_cache/candidate_recall.json`, text-keyed,
zero new spend): 6,843 items pooled across both corpora (4,577 train / 2,266
val).

- Exact-text duplicates spanning >1 vault: **1**, and it's train↔train
  (two A′ repos both vendoring a Contributor Covenant `CODE_OF_CONDUCT.md`)
  — no train↔val exposure.
- Near-duplicates (cosine ≥ 0.95, embedding space) across different vaults,
  train↔val: **7 / 2,266 val items (0.31%)**, all one pair of corpus-B
  vaults (`drshahizan/obsidian`, `Youngermaster/Obsidian-Brain-Notes`) that
  both ship the Obsidian Excalidraw plugin — the "duplicate" content is the
  plugin's auto-generated placeholder file, not user-authored notes.

Bounded and non-systemic — doesn't explain the corpus-A→B/A′ reversal, but
real enough to disclose. Excalidraw placeholder files (and similar
plugin-generated boilerplate) are a real data-cleaning gap worth filtering
in a v2 build (`scripts/build_vault_corpus.py`'s `SKIP_DIRS` list catches
directories, not this kind of same-directory generated file).

---

### Gold-label annotation study (corpus A)

100-item stratified sample (35 folder-disjoint / 18 sparse / 25 mid / 22
dense, matching the coverage-curve buckets exactly), judged independently
by 2 annotators: is the note's actual folder-of-record a sensible home for
it, given only the text? Not judging any model prediction — checking
whether the gold label itself is trustworthy. Tool: `scripts/build_annotation_sample.py`
+ an HTML annotation instrument (autosave, JSON export/import).

| annotator | correct | ambiguous | unclear | wrong |
|---|---|---|---|---|
| prakhar | 80/98 | 12/98 | 6/98 | 0/98 |
| rocky | 82/100 | 10/100 | 8/100 | 0/100 |

n=98 vs n=100: the sample is 100 unique items (`build_annotation_sample.py`,
fixed version). Prakhar's pass predates the fix — 2 of his 100 judgments
landed on items that later turned out to be the same physical note sampled
twice, collapsing to 98 unique — see the dedup bug write-up further down.
Rocky's pass used the corrected 100-item file. 98 is the common-items count
for inter-rater agreement; each annotator's own total (98, 100) is correct
for that annotator.

Inter-rater agreement on the 98 common items: **exact-verdict 95/98 (0.969),
Cohen's κ = 0.901** ("almost perfect"). Binary usable-vs-not (correct+ambiguous
vs unclear+wrong) agreement 97/98. Zero correct-vs-wrong crossings between
raters — every disagreement is a boundary call (correct↔ambiguous or
ambiguous↔unclear). By occupancy bucket (prakhar, corrected): 0-notes 30/33
(91%), 1–2 10/18 (56%, weakest — sparse folders are hardest for humans too,
same shape as the model coverage curve), 3–9 19/25 (76%), 10+ 20/22 (91%).

**Two scope limits, flagged in adversarial review, kept explicit rather than
implied:**
1. **This validates corpus A's gold only.** Corpus B and A′ were never
   annotated (external-annotator exposure to other people's public-vault
   text was ruled out on licensing grounds — see HANDOFF_PAPER.md). A high
   κ on A says nothing directly about whether B's or A′'s folder-of-record
   labels are sensible; do not write sentences that let that inference slip
   through.
2. **The annotation question is "is this a sensible home," not "is this the
   only correct home."** Exact-match accuracy scores against a single gold
   path, which assumes uniqueness the annotation study explicitly does not
   claim — 12% of prakhar's judgments were "ambiguous" (plausible elsewhere
   too). A method that picks a *different* plausible folder is scored wrong
   by exact-match but might be judged fine by an annotator. This caps how
   much the accuracy numbers in this document can be over-interpreted as
   "correctness," corpus-wide, not just in the annotated sample.
   Additionally: annotators disagreed with the gold most in the *same*
   sparse bucket where every model also does worst (56% vs 91%/76%/91%) —
   some fraction of the coverage curve's shape may be gold-label ambiguity
   in sparse folders, not purely task difficulty. Not separable with the
   data in hand; flag as an open confound, not a resolved one.

Reading: **corpus A's gold is trustworthy.** ~80-92% straight "correct"
depending on how ambiguous cases are counted, essentially no cases either
annotator called outright wrong, and disagreement concentrates exactly
where the paper's own claim says it should (sparse folders). This backs
every corpus-A number in this document — they are not being computed
against noisy labels. Does not touch corpus B (kept to corpus A only —
external annotators + other people's public-vault text was ruled out on
licensing grounds, see HANDOFF_PAPER.md).

Data-quality gotcha caught during review: two of the 98 items were the
*same physical note*, sampled once from each split's independent val pool
(a note can legitimately land in both item-split val and fold-split val by
chance, since the splits are independent random partitions) — collapsed to
one judgment in the tool, silently understating n by 2. Fixed in
`build_annotation_sample.py` (dedupes by id, pins each already-judged id to
its original bucket so a re-run doesn't reassign it).

---

## 3. Conclusions

- The cascade beats the fine-tuned 8B on both splits: 0.643 vs 0.537 (p=0.00045)
  with folders seen, 0.419 vs 0.363 (p=0.040) with folders unseen. No training.
- kNN alone ties the LoRA on the item split (p=0.79) at $0.003, and is
  structurally 0.000 on unseen folders.
- The LoRA's item-split advantage over gpt-4o scales with per-folder training
  volume and inverts below ~3 examples per folder.
- What SFT delivered on both splits and on synth is output discipline: valid
  paths (invalid 24.8% → 2.0%) and depth calibration (3 too-deep errors vs
  gpt-4o's 26). Constrained decoding was tested separately on v1 and added
  nothing (0.315 → 0.307).
- Cascade prompt size is independent of tree size (948 tokens). Flat prompts are
  6.3k tokens at 365 folders.
- Folder descriptions are within noise with retrieval (+1.7pt, p=0.487) and
  without it (+2.0pt, p=0.489). Retrieval is worth ~35pt in both conditions.
  Accuracy-given-in-shortlist moved 0.654 → 0.671.
- All four bullets above are corpus-A-specific and do not generalize: on
  corpus B (27 small single-person trees, mean 39 folders) LoRA *beats* the
  cascade on item split (0.756 vs 0.592, vault-count 19/27) and wins the
  sparse and mid occupancy buckets (kNN wins the dense bucket — see the
  corrected occupancy table in the corpus B section). "Fine-tuning loses to
  retrieve-then-pick" depends on the corpus, confirmed on two independent
  public corpora (B, A′). **Why it depends on the corpus is open** — a
  tree-size dose-response mechanism was proposed and tested properly
  (vault-level permutation test, the correct unit of replication) and did
  not survive: p=0.67–0.89, both corpora, opposite-signed between them. See
  corpus B and corpus A′ sections for the full correction.

The v2 gold is folder-of-record — where the file happens to sit — not ground
truth. A note may plausibly belong in several folders (12% of the corpus-A
annotation sample was independently judged "ambiguous" on exactly this
question — see the annotation study section). Exact-match accuracy therefore
undercounts every method's real correctness by some unknown amount; nothing
in this document corrects for it.

### Known limitations, from adversarial pre-draft review (2026-08-19)

Caught by an independent Codex review before drafting began
(`/codex consult`); the fixable ones (statistical unit of analysis, factual
contradictions, leakage) are corrected above. These are the ones left as
honest caveats rather than new experiments, because fixing them properly
means real spend/time — flagged here so the draft states them rather than
omits them:

- **Corpus A is 78% one software project** (`work/openclaw`, 679/870
  tasks) — a construct-validity concern for calling it a general "personal
  directory" corpus. It's one person's real tree, not a designed sample.
- **Corpus A′ lost 9/25 selected repos to clone timeouts** and **corpus B
  capped 9/27 vaults at 250 notes, taken in file order, not randomly** — both
  are selection steps that could shift measured occupancy and tree size as
  a side effect of which specific files survived, not a random sampling
  artifact. No sensitivity analysis run against uncapped/randomly-capped
  alternatives.
- **Occupancy and tree size are confounded with candidate-set size.** The
  cascade sees a top-20 shortlist; LoRA and kNN effectively see the whole
  tree. In small trees those are nearly the same list; in large trees they
  aren't. Any small-tree advantage for LoRA/kNN over cascade could be a
  distractor-count effect, not a "narrower personal convention" effect —
  not disentangled here.
- **Pooled cross-tree LoRA training is a real confound for the B/A′
  reversal.** Each corpus trains *one* LoRA pooled across all its vaults/
  repos (existing_dirs stays per-vault at inference time, but model weights
  are shared). The B/A′ win over cascade could come from more total SFT
  data or cross-tree representation transfer, not from anything about
  individual tree size. The clean test — per-vault isolated LoRAs, or a
  leave-one-vault-out ablation — was not run; it's expensive (each vault
  needs its own train+deploy+eval cycle) and is the natural next spend if
  this claim needs to be load-bearing in the paper rather than a flagged
  open question.
- **Depth caps differ by corpus** (A: 5, B/A′: 8) and correlate with tree
  size, candidate count, and prompt length — not disentangled from the
  corpus-level results either.
- **No held-out confirmation split anywhere in this project.** Every
  reported number, across all three corpora, comes from a val set that
  also implicitly shaped which k / shortlist size / analysis got reported.
  Label every number in the draft as exploratory, not confirmatory.
- **The H100/H200 swap on corpus B's fold-split LoRA deploy is asserted
  equivalent, never verified** — no paired-prediction comparison was run
  between the two hardware configs.
- Retrieval×grounding null results (p≈0.49, both cells) are failure-to-
  reject, not evidence of equivalence — worded that way above, keep that
  distinction in the draft.

---

## 4. Open

- Per-note routing between `note` and `path` candidates. The fixed union is below
  the better single source on both splits. Note-side top similarity is a
  candidate routing signal. Scoring change, no new API calls.
- Shortlist size / reranking. 20→50 traded +17pt recall for −7pt picking accuracy.
- `confidence` field is dead (all training targets 0.9), so low-confidence notes
  cannot route to `inbox` for review.
- Cheaper picker: gpt-4o-mini or an 8B on the 948-token cascade prompt.

Untested and not planned: more synthetic data, larger LoRA rank, more epochs,
further fine-tuning of the placer.

---

## 5. Reproduce

```bash
cd phase0

# benchmark
python3 scripts/snapshot_user_dir.py --config data/user_dir_snap_v2/sources.json \
                                     --out data/user_dir_snap_v2
python3 scripts/build_user_dir_split.py --snap data/user_dir_snap_v2 --val-frac 0.4
python3 scripts/build_user_dir_split.py --snap data/user_dir_snap_v2 --val-frac 0.4 \
                                        --split-by folder --out-prefix fold_

# flat baselines
python3 scripts/eval_fireworks_placer.py --tasks data/user_dir_snap_v2/val.jsonl \
  --store data/user_dir_snap_v2/hierstore.sqlite --provider openai --model gpt-4o \
  --out runs/v2_val_gpt4o
python3 scripts/knn_placer_baseline.py --train data/user_dir_snap_v2/train.jsonl \
  --val data/user_dir_snap_v2/val.jsonl --out runs/v2_val_knn --k 8

# candidate recall gate (embeddings only, run before any cascade spend)
python3 scripts/candidate_recall.py --train data/user_dir_snap_v2/train.jsonl \
  --val data/user_dir_snap_v2/val.jsonl --store data/user_dir_snap_v2/hierstore.sqlite \
  --out runs/v2_candidate_recall.json

# cascade — note candidates for seen folders, path candidates for unseen
python3 scripts/cascade_placer.py --train data/user_dir_snap_v2/train.jsonl \
  --val data/user_dir_snap_v2/val.jsonl --store data/user_dir_snap_v2/hierstore.sqlite \
  --mode note --n 20 --model gpt-4o --out runs/v2_val_cascade_note20
python3 scripts/cascade_placer.py --train data/user_dir_snap_v2/fold_train.jsonl \
  --val data/user_dir_snap_v2/fold_val.jsonl --store data/user_dir_snap_v2/hierstore.sqlite \
  --mode path --n 50 --model gpt-4o --out runs/v2fold_val_cascade_path50

# folder descriptions (item split only — see leak note in section 2)
python3 scripts/gen_folder_descriptions.py --train data/user_dir_snap_v2/train.jsonl \
  --out data/user_dir_snap_v2/folder_descriptions.json
# grounding without retrieval — the flat cell of the 2x2. Keep workers low:
# 14,963 tok/call against a 450k tok/min cap.
python3 scripts/eval_fireworks_placer.py --tasks data/user_dir_snap_v2/val.jsonl \
  --store data/user_dir_snap_v2/hierstore.sqlite --provider openai --model gpt-4o \
  --descriptions data/user_dir_snap_v2/folder_descriptions.json --workers 2 \
  --out runs/v2_val_gpt4o_desc
python3 scripts/cascade_placer.py --train data/user_dir_snap_v2/train.jsonl \
  --val data/user_dir_snap_v2/val.jsonl --store data/user_dir_snap_v2/hierstore.sqlite \
  --mode note --n 20 --model gpt-4o \
  --descriptions data/user_dir_snap_v2/folder_descriptions.json \
  --out runs/v2_val_cascade_note20_desc

# synth holdout decomposition
python3 scripts/decompose_placer_holdout.py \
  "LoRA=runs/fireworks_placer_mid_hard" "gpt-4o=runs/gpt4o_mid_hard_holdout"
```

Fine-tuning: upload with `scripts/fireworks_upload_dataset.sh <id> <file.jsonl>`,
then create the job with `epochs:1, loraRank:16, lr:1e-4, maxContextLength:16384`
and no `earlyStop`. llama-3.1-8b is not serverless on Fireworks — eval needs a
dedicated H200 deployment; delete it afterwards
(`DELETE /deployments/<id>?ignoreChecks=true`) and confirm the list is empty.

**Artifacts.** `runs/v2_val_{gpt4o,base8b,lora,knn,cascade_note20,cascade_note20_desc}/`,
`runs/v2fold_val_{gpt4o,base8b,lora,knn,cascade_path20,cascade_path50}/`,
`runs/{v2,v2fold}_candidate_recall.json`, `data/user_dir_snap_v2/`.
`data/user_dir_snap/` (v1, n=127) is frozen and is the basis of all pre-v2
numbers; it is not comparable to v2 (shallower tree, 69 folders vs 365).

**Cost:** ~$52 — two SFT runs (5.2M + 4.6M tokens), two H200 sessions ~25 min
each, ~$40 of gpt-4o passes (including ~$15 for the flat+descriptions cell, of
which ~$4 was the rate-limited first attempt), ~$0.02 of embeddings and
descriptions.

---

## 6. Gotchas

- `earlyStop: true` is rejected by Fireworks ("deprecated and not supported by
  managed training") — the whole job create fails.
- Dataset create needs `datasetId` in the body, not as a query parameter.
- A `nohup bash -c '...' &` inside a tool call gets reaped mid-run; a chained
  `until ! pgrep -f "<script>"` waiter matches its own command line and waits
  forever. Poll for an output file instead.
- `call_placer` had no retry. At 14,963 tokens/call an 8-worker run exceeds the
  450k tokens/min gpt-4o cap and the first flat+descriptions pass lost 185/294
  items to 429s — a partial, timing-selected sample that still writes a
  plausible-looking `summary.json` (it read `path_exact` 0.122 against a true
  0.306). Retry with backoff is now inside `call_placer`, so every arm has it.
  Check `parse_fail` before reading any summary.
- The hardcoded `gpt4o_baseline` block in `eval_fireworks_placer.py`'s summary
  was a stale n=90 smoke result; removed.
