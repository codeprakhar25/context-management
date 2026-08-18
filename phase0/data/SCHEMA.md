# Phase-0 corpus schema

Frozen for toy + later full synthetic. Embed: OpenAI API `text-embedding-3-small` (direct, not Azure).

## facts.jsonl

One JSON object per line.

| field | type | required | notes |
|-------|------|----------|-------|
| `id` | string | yes | stable id e.g. `f_a01` |
| `text` | string | yes | atomic claim / fact |
| `path` | string[] | yes | e.g. `["project","alpha"]` or `["global"]` |
| `kind` | string | yes | `claim` for v0 |
| `project` | string\|null | yes | `alpha`/`beta`/`gamma` or `null` if global |
| `t` | string | yes | ISO-8601 |
| `tags` | string[] | no | jargon markers for analysis |
| `episode_id` | string\|null | no | reserved for later |

## queries.jsonl

| field | type | required | notes |
|-------|------|----------|-------|
| `id` | string | yes | e.g. `q_01` |
| `text` | string | yes | natural language question |
| `type` | string | yes | `local` \| `global` \| `mixed` \| `hard_local` \| `adversarial` |
| `project` | string\|null | yes | active project scope for Hier; null for pure global |
| `gold_ids` | string[] | yes | fact ids that must be retrieved |
| `gold_answer` | string | yes | short string for EM/F1 |
| `distractor_ids` | string[] | no | known wrong-project near-misses (analysis) |
| `notes` | string | no | why this query is hard |

## run config (later)

Must log: `corpus_id`, `EMBED_MODEL=text-embedding-3-small`, `embedder=OpenAI`, `READER_MODEL`, `k`, `k_proj`, `k_global`, temp=0.
