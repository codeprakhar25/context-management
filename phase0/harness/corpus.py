"""Load canonical corpus + hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def corpus_hash(facts_path: Path, queries_path: Path) -> str:
    h = hashlib.sha256()
    for p in (facts_path, queries_path):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def load_corpus(data_dir: Path) -> tuple[list[dict], list[dict], str]:
    facts_path = data_dir / "facts.jsonl"
    queries_path = data_dir / "queries.jsonl"
    facts = _read_jsonl(facts_path)
    queries = _read_jsonl(queries_path)
    cid = corpus_hash(facts_path, queries_path)
    return facts, queries, cid
