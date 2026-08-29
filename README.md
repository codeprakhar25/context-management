# The Occupancy Curve

**Why note-placement accuracy depends on how full the folder already is.**

An agent that keeps long-term notes has to decide, every time it writes, which
folder the note goes in. That decision has a convenient gold label, because real
note collections record where each note actually ended up, and recent work scores
it with a single accuracy number.

This repository is the evaluation study arguing that the single number is the
problem. Filing into a folder that already holds a dozen similar notes and filing
into a folder created this morning are different problems that reward different
machinery, and pooling them hides a method ranking that reorders — and, across
corpora, reverses.

## The result in one table

We define a note's **occupancy** as the number of training notes already filed in
its gold folder, and report accuracy separately at each level. Exact-match
accuracy on corpus B, 27 public note vaults, item-stratified split:

| occupancy | n | majority | flat gpt-4o | BM25 | kNN | cascade | LoRA 8B |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–2 | 258 | 0.000 | 0.547 | 0.403 | 0.457 | **0.628** | **0.667** |
| 3–9 | 653 | 0.064 | 0.662 | 0.636 | 0.718 | 0.729 | **0.776** |
| 10+ | 558 | 0.545 | 0.332 | 0.771 | **0.833** | 0.414 | 0.772 |
| **all** | **1,469** | 0.235 | 0.516 | 0.646 | 0.716 | 0.592 | **0.756** |

The `all` row is the number a conventional paper reports. The rows above it are
what that number averages over. Both retrieval baselines climb from the bottom of
the sparse bucket to the top of the dense one; the retrieve-then-pick cascade
falls from second to fourth, losing 21 points. A reader given only the pooled row
would conclude LoRA is the method and kNN a respectable second — the stratified
rows say the two are separated almost entirely by what happens in folders that are
nearly empty.

**Folders never used are a separate regime.** On a folder-disjoint split of the
same vaults (n=1,879), where whole folders are held out of training, the order
inverts: cascade 0.583, flat gpt-4o 0.543, Llama cascade 0.499, LoRA 0.475, kNN
0.000 by construction. Retrieval over training notes cannot name a folder that has
none. This is where retrieve-then-pick earns its keep, and it is the only place.

**The ranking does not travel between corpora.**

| corpus | LoRA | cascade | kNN | BM25 | vault-level wins (LoRA/casc/tie) |
|---|---:|---:|---:|---:|---|
| A — 1 private tree | 0.537 | **0.643** | 0.524 | 0.442 | — |
| B — 27 public vaults | **0.756** | 0.592 | 0.716 | 0.646 | 19 / 6 / 2 |
| A′ — 16 repositories | 0.716 | 0.552 | **0.732** | 0.666 | 13 / 3 / 0 |

The cascade wins outright on the private tree and loses on both public corpora,
broadly enough at the vault level that it is not a pooled-average artifact.

## A mechanism we tested and retracted

The natural explanation for that reversal is tree size: LoRA sees the whole folder
list, the cascade sees twenty candidates, so smaller trees should favour LoRA.
Pooling all 1,469 items into terciles by folder count produced a clean monotone
dose-response replicated across both public corpora, with p-values down to 10⁻²⁵.

It was wrong. Items within a vault are not independent draws, so pooling them and
testing per bin is pseudo-replication — the real number of independent units is 27
vaults, not 1,469 items. Redone at the correct unit, as a permutation test on the
Spearman correlation between log folder count and each vault's own
(LoRA − cascade) margin, the effect vanishes: B ρ=+0.029, p=0.887 (n=27);
A′ ρ=−0.115, p=0.669 (n=16); pooled ρ=+0.034, p=0.826 (n=43). Null in both
corpora, opposite-signed between them.

The reversal is real and reproduced. Its cause is open. It is reported that way.

## What this repository claims and does not

Occupancy is **confounded** with folder, vault, and corpus identity. Flat gpt-4o
sees no training members at all, yet still swings 0.547 → 0.662 → 0.332 across the
same buckets, so part of what the strata capture is difficulty rather than evidence
available to the model. Flat gpt-4o is the negative control that forces this
caution: occupancy is reported as an axis along which rankings change, not as a
cause.

The aggregate "retrieval beats the cascade" result **does not survive** a
sensitivity analysis. Three vaults with opaque folder names carry it; removing them
turns the aggregate into a tie (kNN 0.677 vs. cascade 0.679). The crossing itself
survives the same trim. Neighbourhood size *k* and shortlist size were selected on
the data reported on, so every number here is exploratory rather than confirmatory.

## Corpora

| | what | scope |
|---|---|---|
| A | one private working directory — 870 notes, 170 occupied folders, 365 candidate folders | full grid, all arms; 78% of tasks from one project |
| B | 27 public personal note vaults from GitHub | full grid, item-stratified + folder-disjoint |
| A′ | 16 public software repositories — 2,644 notes | item-stratified only |

`phase0/data/vaults_manifest.json` and `vaultsA_manifest.json` pin the exact commit
each vault and repository was built from. Raw clones, and any dataset that renders
third-party note text into training rows, are **not redistributed here** — the
manifests plus the build scripts are what let anyone reconstruct the corpora. Other
people's notes are not ours to republish. `vaults_probe.json` records the full
selection funnel (790 candidates screened → 132 usable) as aggregate counts only.

**Gold-label validation.** 100 items sampled from corpus A, judged independently by
two annotators against one question — is the recorded folder a sensible home for
this note, given only its text? Cohen's κ = 0.901 on the 98 shared items, no item
called correct by one annotator and wrong by the other. The annotation instrument
is not included, since it embeds private corpus-A note excerpts.

## Layout

```
phase0/
  harness/       note store and retrieval core — store.py, embed.py, bm25.py, index.py
  scripts/       one entry point per method × corpus combination
  data/          corpus manifests, synthetic trees, training splits
  PLACER_FINDINGS.md   the full run record
```

[`phase0/PLACER_FINDINGS.md`](phase0/PLACER_FINDINGS.md) is the primary document
here: a dated, 973-line record of every run behind the tables above, including the
corrections and the negative results. It is where the retracted tree-size mechanism
is worked through, where the dedup bug in the annotation sample is written up, and
where each number's provenance can be checked.

`harness/store.py` is a hierarchical note store over SQLite: a hard tree (each note
under exactly one path), `ADD`/`UPDATE`/`DELETE`/`NOOP` ops with soft delete, an
append-only `ops_log`, and cached embeddings. Everything else in the repository
reads or writes through it.

## Reproduce

```bash
cd phase0
pip install -r requirements.txt
# API keys (OpenAI, Fireworks, OpenRouter) go in phase0/.env — not included
```

```bash
# candidate-recall gate — embeddings only, run before spending on any picker
python3 scripts/candidate_recall.py --train <train.jsonl> --val <val.jsonl> \
  --store <hierstore.sqlite> --out <out.json>

# the four methods
python3 scripts/eval_fireworks_placer.py ...   # LoRA
python3 scripts/cascade_placer.py ...          # retrieve-then-pick
python3 scripts/knn_placer_baseline.py ...     # embedding kNN
python3 scripts/eval_lexical_baselines.py ...  # BM25

# rebuild corpus B / A′ from a manifest
python3 scripts/fetch_vaults.py --manifest data/vaults_manifest.json --dest data/vaults_raw
python3 scripts/build_vault_corpus.py --manifest data/vaults_manifest.json \
  --raw data/vaults_raw --out data/vaults_build

# occupancy tables and the sensitivity analysis — no API calls, reads stored artifacts
python3 scripts/occupancy_sensitivity.py ...
```

Embeddings throughout are OpenAI `text-embedding-3-small`. The occupancy tables and
the sensitivity analysis rebuild from stored per-item run artifacts, so they can be
checked without spending anything.

### Corpus format

Corpora are JSONL. `facts.jsonl`, one object per line:

| field | type | notes |
|---|---|---|
| `id` | string | stable, e.g. `f_a01` |
| `text` | string | the note or atomic claim |
| `path` | string[] | gold folder, e.g. `["work","slm-lab","infra"]` |
| `kind` | string | `claim` |
| `project` | string \| null | project scope, `null` if global |
| `t` | string | ISO-8601 |
| `tags` | string[] | optional, markers for analysis |

Retrieval evaluations additionally read `queries.jsonl` with `id`, `text`, `type`,
`project`, `gold_ids`, `gold_answer`, and optional `distractor_ids`.

## License

Code is MIT — see [LICENSE](LICENSE). The corpus manifests are pointers into public
GitHub repositories, not redistributed content; the underlying note and source text
belongs to its original authors. Aggregate statistics only, no verbatim quoting, and
takedown requests are honoured against the manifests.
