# The Occupancy Curve

Evaluation study for LLM note placement (given a new note and an existing
folder tree, predict which folder it belongs in). The core finding: method
rankings depend on how many notes already sit in the target folder (its
*support*), an axis prior work in this task does not stratify results by.
A second, more striking result — a fine-tuned model beating a
retrieval-then-pick cascade on two independent public corpora, in a regime
where the reverse holds on our private corpus — is reported as a real,
reproduced, but unexplained reversal; a proposed tree-size mechanism for it
was tested directly and did not survive.

Paper draft: [`phase0/paper/main.pdf`](phase0/paper/main.pdf)
(LaTeX source: `phase0/paper/main.tex`). Full results record, including
every run, correction, and negative result along the way:
[`phase0/PLACER_FINDINGS.md`](phase0/PLACER_FINDINGS.md).

## Corpora

| | what | scope |
|---|---|---|
| A | one private working directory, 870 tasks / 170 folders | full grid, all arms |
| B | 27 public personal note vaults (GitHub) | full grid, item + folder-disjoint |
| A′ | 16 public software repositories (GitHub) | item-stratified only |

Corpus B and A′ manifests (`phase0/data/vaults_manifest.json`,
`phase0/data/vaultsA_manifest.json`) pin the exact commit each vault/repo
was built from. Raw clones and any dataset that renders third-party note
text into training rows are not redistributed here — the manifests and
build scripts are what let anyone reconstruct the corpora themselves;
other people's note text is not ours to redistribute.

## Reproduce

```bash
cd phase0
pip install -r requirements.txt
# API keys (OpenAI, Fireworks, OpenRouter) in phase0/.env — not included
```

```bash
# candidate-recall gate (embeddings only, run before any picker spend)
python3 scripts/candidate_recall.py --train <train.jsonl> --val <val.jsonl> \
  --store <hierstore.sqlite> --out <out.json>

# flat / cascade / kNN / LoRA baselines
python3 scripts/eval_fireworks_placer.py ...
python3 scripts/cascade_placer.py ...
python3 scripts/knn_placer_baseline.py ...

# corpus B / A′ construction (from a manifest)
python3 scripts/fetch_vaults.py --manifest data/vaults_manifest.json --dest data/vaults_raw
python3 scripts/build_vault_corpus.py --manifest data/vaults_manifest.json --raw data/vaults_raw --out data/vaults_build
```

`harness/` holds the retrieval/scoring core (`store.py`, `embed.py`,
`metrics.py`); `scripts/` holds every experiment entry point, one per
method/corpus combination.

## Gold-label validation

100 items sampled from corpus A, judged independently by two annotators
against one question — is the recorded folder a sensible home for this
note, given only the text? Cohen's κ = 0.901. See §5 of the paper and the
annotation instrument the study used (a self-contained HTML tool with
local autosave and JSON export, not included in this repo since it embeds
private corpus-A note excerpts).

## License / data policy

Code in this repository is available for reuse. Corpus B/A′ manifests are
pointers (commit SHAs) into public GitHub repositories, not redistributed
content — the underlying note/source text belongs to its original authors.
Aggregate statistics only; no verbatim quoting; honor takedown requests
against the manifests.
