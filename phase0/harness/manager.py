"""Pluggable memory managers (family A). Choose ops; HierStore applies them.

AlwaysADD  — baseline (always insert)
RuleV0     — scoped embed thresholds (NOOP / UPDATE / ADD)
LLMv0      — scoped retrieve + gpt chooses {ADD, UPDATE, DELETE, NOOP}
"""
from __future__ import annotations

import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from openai import OpenAI

from harness.store import HierStore, Op


@dataclass
class ManagerInput:
    """One candidate claim to write into the store."""

    text: str
    project: str | None
    path: list[str] | None = None
    kind: str = "claim"
    t: str | None = None
    episode_id: str | None = None
    fact_id: str | None = None  # optional pre-assigned id for ADD


@dataclass
class ManagerDecision:
    ops: list[Op]
    retrieve_ids: list[str]
    reason: str
    raw_llm: str | None = None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9.\s=+-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_NEG = re.compile(
    r"\b(not|never|no longer|doesn't|does not|isn't|is not|wasn't|was not|"
    r"moved (away )?from|left|quit|stopped|false|incorrect)\b",
    re.I,
)

# FROZEN neutral prompt — identical across manager arms. No preference rules / few-shots.
# Hash when changing: sha256 of this string logged in runs.
LLM_MANAGER_SYSTEM = """You are a memory manager for a fact bank.
Given a NEW fact and RELATED existing memories, choose zero or more operations.

Operation definitions (semantics only — no preferred policy):
- ADD: insert a new memory with the given text.
- UPDATE: replace the text of an existing RELATED memory (same id).
- DELETE: soft-remove an existing RELATED memory (id required).
- NOOP: make no change to the bank.

Constraints:
- Only UPDATE/DELETE ids that appear in RELATED.
- You may emit multiple ops (e.g. DELETE then ADD).
- Project-scoped NEW facts may target global RELATED memories when appropriate.
- Reply with JSON ONLY: a list of objects.
  Each: {"event":"ADD"|"UPDATE"|"DELETE"|"NOOP","id":"<existing id or null>","text":"<string or null>"}
  For ADD: id may be null; text required.
  For UPDATE: id required; text required.
  For DELETE: id required; text null.
  For NOOP: id optional; text null.
"""

# Ablation prompt: identical to LLM_MANAGER_SYSTEM except for the block below.
# Tests whether the condition-scoped failure is a vocabulary gap or a prompting
# gap. Deliberately abstract — names no case type, no domain, gives no example,
# and uses none of the vocabulary the eval set is generated from, so a gain here
# cannot be an answer key (the mistake in the v0 prompt).
_COND_BLOCK = """
Scope:
- A claim may hold only under some condition — a particular context, subject,
  system, place, or period. Two claims that appear to conflict are not in
  conflict when they hold under different conditions; both remain true.
- Treat a NEW fact as replacing an existing memory only when the two share the
  same condition. Otherwise they coexist.
"""

LLM_MANAGER_SYSTEM_COND = LLM_MANAGER_SYSTEM.replace(
    "Constraints:", _COND_BLOCK.strip() + "\n\nConstraints:"
)

PROMPTS: dict[str, str] = {
    "neutral": LLM_MANAGER_SYSTEM,
    "cond": LLM_MANAGER_SYSTEM_COND,
}


def prompt_sha(text: str) -> str:
    try:
        import hashlib as _hashlib

        return _hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    except Exception:  # pragma: no cover
        return "unavailable"


LLM_MANAGER_SYSTEM_SHA256 = prompt_sha(LLM_MANAGER_SYSTEM)


def scoped_candidates(store: HierStore, project: str | None) -> list[dict]:
    local = store.read_by_project(project, valid_only=True)
    if project is None:
        return local
    glob = store.read_by_project(None, valid_only=True)
    seen = {f["id"] for f in local}
    return local + [g for g in glob if g["id"] not in seen]


def rank_by_embed(
    query_vec: np.ndarray,
    facts: Sequence[dict],
    *,
    store: HierStore,
    model: str,
    embed_fn,
    cache: dict[str, np.ndarray],
    top_m: int,
) -> list[tuple[float, dict]]:
    scored: list[tuple[float, dict]] = []
    missing_texts: list[str] = []
    missing_ids: list[str] = []
    for f in facts:
        if f["id"] in cache:
            continue
        stored = store.get_embedding(f["id"], model)
        if stored is not None:
            cache[f["id"]] = np.asarray(stored, dtype=np.float32)
        else:
            missing_texts.append(f["text"])
            missing_ids.append(f["id"])
    if missing_texts and embed_fn is not None:
        mat = np.asarray(embed_fn(missing_texts), dtype=np.float32)
        for i, fid in enumerate(missing_ids):
            cache[fid] = mat[i]
            store.put_embedding(fid, model, mat[i].tolist())
    for f in facts:
        v = cache.get(f["id"])
        if v is None:
            continue
        scored.append((_cosine(query_vec, v), f))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_m]


def _default_path(project: str | None, path: list[str] | None) -> list[str]:
    if path is not None:
        return path
    return ["global"] if project is None else ["project", project]


class MemoryManager(ABC):
    name: str = "base"

    @abstractmethod
    def decide(self, inp: ManagerInput, store: HierStore) -> ManagerDecision:
        ...

    def apply(self, inp: ManagerInput, store: HierStore) -> list[dict[str, Any]]:
        dec = self.decide(inp, store)
        for op in dec.ops:
            if not op.retrieve_ids:
                op.retrieve_ids = list(dec.retrieve_ids)
            if op.reason is None:
                op.reason = dec.reason
        return store.apply_ops(dec.ops, manager=self.name)


class AlwaysADD(MemoryManager):
    """Baseline: always insert a new row (v0 LoCoMo ingest)."""

    name = "AlwaysADD"

    def decide(self, inp: ManagerInput, store: HierStore) -> ManagerDecision:
        path = _default_path(inp.project, inp.path)
        fid = inp.fact_id or f"f_{uuid.uuid4().hex[:12]}"
        return ManagerDecision(
            ops=[
                Op(
                    op="ADD",
                    fact_id=fid,
                    text=inp.text,
                    path=path,
                    project=inp.project,
                    kind=inp.kind,
                    t=inp.t,
                    episode_id=inp.episode_id,
                    reason="always_add",
                )
            ],
            retrieve_ids=[],
            reason="always_add",
        )


class RuleV0(MemoryManager):
    """Scoped embed cosine: high sim → UPDATE; near-dupe → NOOP; else ADD."""

    name = "RuleV0"

    def __init__(
        self,
        embeddings: dict[str, np.ndarray] | None = None,
        embed_fn=None,
        *,
        top_m: int = 5,
        update_thresh: float = 0.88,
        noop_thresh: float = 0.97,
        model: str = "text-embedding-3-small",
        embedder: Any | None = None,
    ):
        self.embeddings = dict(embeddings or {})
        self.embedder = embedder
        if embed_fn is not None:
            self.embed_fn = embed_fn
        elif embedder is not None:
            self.embed_fn = lambda texts: embedder.embed_texts(list(texts))
        else:
            self.embed_fn = None
        self.top_m = top_m
        self.update_thresh = update_thresh
        self.noop_thresh = noop_thresh
        self.model = model

    def decide(self, inp: ManagerInput, store: HierStore) -> ManagerDecision:
        path = _default_path(inp.project, inp.path)
        cands = scoped_candidates(store, inp.project)

        if self.embed_fn is None:
            fid = inp.fact_id or f"f_{uuid.uuid4().hex[:12]}"
            return ManagerDecision(
                ops=[
                    Op(
                        op="ADD",
                        fact_id=fid,
                        text=inp.text,
                        path=path,
                        project=inp.project,
                        kind=inp.kind,
                        t=inp.t,
                        episode_id=inp.episode_id,
                        reason="rule_v0_no_embed_fn",
                    )
                ],
                retrieve_ids=[],
                reason="rule_v0_add_fallback",
            )

        q = np.asarray(self.embed_fn([inp.text])[0], dtype=np.float32)
        if not cands:
            fid = inp.fact_id or f"f_{uuid.uuid4().hex[:12]}"
            return ManagerDecision(
                ops=[
                    Op(
                        op="ADD",
                        fact_id=fid,
                        text=inp.text,
                        path=path,
                        project=inp.project,
                        kind=inp.kind,
                        t=inp.t,
                        episode_id=inp.episode_id,
                        reason="rule_v0_empty_bank",
                    )
                ],
                retrieve_ids=[],
                reason="rule_v0_add",
            )

        ranked = rank_by_embed(
            q,
            cands,
            store=store,
            model=self.model,
            embed_fn=self.embed_fn,
            cache=self.embeddings,
            top_m=self.top_m,
        )
        retrieve_ids = [f["id"] for _, f in ranked]
        if not ranked:
            fid = inp.fact_id or f"f_{uuid.uuid4().hex[:12]}"
            return ManagerDecision(
                ops=[
                    Op(
                        op="ADD",
                        fact_id=fid,
                        text=inp.text,
                        path=path,
                        project=inp.project,
                        kind=inp.kind,
                        t=inp.t,
                        episode_id=inp.episode_id,
                        reason="rule_v0_no_scored",
                        retrieve_ids=retrieve_ids,
                    )
                ],
                retrieve_ids=retrieve_ids,
                reason="rule_v0_add",
            )

        best_sim, best = ranked[0]
        nt_new = _norm_text(inp.text)
        nt_old = _norm_text(best["text"])
        contradict = bool(_NEG.search(inp.text) or _NEG.search(best["text"]))

        if nt_new == nt_old or (best_sim >= self.noop_thresh and not contradict):
            return ManagerDecision(
                ops=[
                    Op(
                        op="NOOP",
                        fact_id=best["id"],
                        reason=f"near_dupe_sim={best_sim:.3f}",
                        retrieve_ids=retrieve_ids,
                    )
                ],
                retrieve_ids=retrieve_ids,
                reason="rule_v0_noop",
            )

        # Supersede (negation + high sim): DELETE old + ADD new (locked control gold).
        # Complement / rewrite without negation cue: UPDATE in place.
        if contradict and best_sim >= self.update_thresh - 0.08:
            return ManagerDecision(
                ops=[
                    Op(
                        op="DELETE",
                        fact_id=best["id"],
                        reason=f"sim={best_sim:.3f}_supersede",
                        retrieve_ids=retrieve_ids,
                    ),
                    Op(
                        op="ADD",
                        fact_id=inp.fact_id or f"f_{uuid.uuid4().hex[:12]}",
                        text=inp.text,
                        path=path,
                        project=inp.project,
                        kind=inp.kind,
                        t=inp.t,
                        episode_id=inp.episode_id,
                        reason=f"sim={best_sim:.3f}_supersede_add",
                        retrieve_ids=retrieve_ids,
                    ),
                ],
                retrieve_ids=retrieve_ids,
                reason="rule_v0_delete_add",
            )

        if best_sim >= self.update_thresh:
            return ManagerDecision(
                ops=[
                    Op(
                        op="UPDATE",
                        fact_id=best["id"],
                        text=inp.text,
                        path=path,
                        project=inp.project,
                        episode_id=inp.episode_id,
                        reason=f"sim={best_sim:.3f}",
                        retrieve_ids=retrieve_ids,
                    )
                ],
                retrieve_ids=retrieve_ids,
                reason="rule_v0_update",
            )

        fid = inp.fact_id or f"f_{uuid.uuid4().hex[:12]}"
        return ManagerDecision(
            ops=[
                Op(
                    op="ADD",
                    fact_id=fid,
                    text=inp.text,
                    path=path,
                    project=inp.project,
                    kind=inp.kind,
                    t=inp.t,
                    episode_id=inp.episode_id,
                    reason=f"sim={best_sim:.3f}_below_update",
                    retrieve_ids=retrieve_ids,
                )
            ],
            retrieve_ids=retrieve_ids,
            reason="rule_v0_add",
        )


class ParseError(Exception):
    """Invalid LLM op JSON / targets."""

    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        super().__init__(f"{kind}: {detail}")


def _fallback_add(inp: ManagerInput, reason: str) -> list[Op]:
    path = _default_path(inp.project, inp.path)
    return [
        Op( 
            op="ADD",
            fact_id=inp.fact_id or f"f_{uuid.uuid4().hex[:12]}",
            text=inp.text,
            path=path,
            project=inp.project,
            kind=inp.kind,
            t=inp.t,
            episode_id=inp.episode_id,
            reason=reason,
        )
    ]


def parse_llm_ops(
    raw: str,
    *,
    related: Sequence[dict],
    inp: ManagerInput,
    allow_fallback: bool = False,
) -> list[Op]:
    """Parse LLM JSON into Ops.

    On invalid JSON/targets: raise ParseError unless allow_fallback → single ADD.
    """
    path = _default_path(inp.project, inp.path)
    related_ids = {f["id"] for f in related}
    related_by_id = {f["id"]: f for f in related}

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        if allow_fallback:
            return _fallback_add(inp, "llm_parse_fail_fallback_add")
        raise ParseError("parse_fail", str(e)) from e

    if isinstance(data, dict):
        data = data.get("ops") or data.get("operations") or [data]
    if not isinstance(data, list) or not data:
        if allow_fallback:
            return _fallback_add(inp, "llm_empty_fallback_add")
        raise ParseError("empty", "no ops list")

    ops: list[Op] = []
    add_i = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        if "event" not in item and "op" not in item and "operation" not in item:
            if allow_fallback:
                continue
            raise ParseError("missing_event", str(item))
        event = str(item.get("event") or item.get("op") or item.get("operation")).upper()
        event = event.replace("NO_OPERATION", "NOOP").replace("NO-OPERATION", "NOOP")
        if event not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
            if allow_fallback:
                continue
            raise ParseError("bad_event", event)
        fid = item.get("id") or item.get("fact_id")
        if fid is not None:
            fid = str(fid)
            if fid in ("null", "None", ""):
                fid = None
        mem_text = item.get("text")
        if mem_text is None and event in {"ADD", "UPDATE"}:
            mem_text = item.get("memory") or item.get("new_memory")

        if event == "ADD":
            if not mem_text:
                if allow_fallback:
                    mem_text = inp.text
                else:
                    raise ParseError("add_missing_text", "")
            if add_i == 0 and inp.fact_id:
                new_id = inp.fact_id
            else:
                new_id = f"f_{uuid.uuid4().hex[:12]}"
            add_i += 1
            ops.append(
                Op(
                    op="ADD",
                    fact_id=new_id,
                    text=str(mem_text),
                    path=path,
                    project=inp.project,
                    kind=inp.kind,
                    t=inp.t,
                    episode_id=inp.episode_id,
                    reason="llm_add",
                )
            )
        elif event == "UPDATE":
            if fid not in related_ids:
                old = item.get("old_memory") or item.get("old_text")
                if old:
                    for rid, f in related_by_id.items():
                        if _norm_text(f["text"]) == _norm_text(str(old)):
                            fid = rid
                            break
            if fid not in related_ids:
                if allow_fallback:
                    continue
                raise ParseError("update_bad_id", str(fid))
            if not mem_text:
                if allow_fallback:
                    mem_text = inp.text
                else:
                    raise ParseError("update_missing_text", fid or "")
            ops.append(
                Op(
                    op="UPDATE",
                    fact_id=fid,
                    text=str(mem_text),
                    path=path,
                    project=inp.project,
                    episode_id=inp.episode_id,
                    reason="llm_update",
                )
            )
        elif event == "DELETE":
            if fid not in related_ids:
                if allow_fallback:
                    continue
                raise ParseError("delete_bad_id", str(fid))
            ops.append(Op(op="DELETE", fact_id=fid, reason="llm_delete"))
        elif event == "NOOP":
            ops.append(
                Op(
                    op="NOOP",
                    fact_id=fid if fid in related_ids else None,
                    reason="llm_noop",
                )
            )

    if not ops:
        if allow_fallback:
            return _fallback_add(inp, "llm_no_valid_ops_fallback_add")
        raise ParseError("no_valid_ops", "")
    return ops


def make_chat_client(provider: str = "openai") -> OpenAI:
    """Chat client only. Embeddings always use OpenAI via Embedder + OPENAI_API_KEY.

    provider=openai     → api.openai.com + OPENAI_API_KEY
    provider=openrouter → openrouter.ai + OPENROUTER_API_KEY
    Never sets/reads OPENAI_BASE_URL for openai provider (keeps OpenAI tests pure).
    """
    p = (provider or "openai").strip().lower()
    if p in ("openrouter", "or"):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("LLMv0 openrouter needs OPENROUTER_API_KEY")
        return OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/ml-resarch/context-management",
                "X-Title": "context-management-phase0",
            },
        )
    if p in ("openai", "oa"):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("LLMv0 openai needs OPENAI_API_KEY")
        # Explicit default host — ignore any ambient OPENAI_BASE_URL for chat purity
        return OpenAI(api_key=key, base_url="https://api.openai.com/v1")
    raise ValueError(f"unknown chat provider: {provider}")


CHAT_PRICE = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
}


class LLMv0(MemoryManager):
    """Scoped embed retrieve → LLM ops → executor. Retry once then ADD fallback."""

    name = "LLMv0"

    def __init__(
        self,
        *,
        embedder: Any | None = None,
        embed_fn=None,
        client: OpenAI | None = None,
        chat_model: str = "gpt-4o-mini",
        chat_provider: str = "openai",
        embed_model: str = "text-embedding-3-small",
        top_m: int = 8,
        temperature: float = 0.0,
        embeddings: dict[str, np.ndarray] | None = None,
        max_retries: int = 1,
        prompt: str = "neutral",
    ):
        self.embedder = embedder
        if embed_fn is not None:
            self.embed_fn = embed_fn
        elif embedder is not None:
            self.embed_fn = lambda texts: embedder.embed_texts(list(texts))
        else:
            self.embed_fn = None
        self._client = client
        self.chat_provider = chat_provider
        self.chat_model = chat_model
        self.prompt_name = prompt
        self.system_prompt = PROMPTS.get(prompt, prompt)
        self.embed_model = embed_model
        self.top_m = top_m
        self.temperature = temperature
        self.embeddings: dict[str, np.ndarray] = dict(embeddings or {})
        self.max_retries = max_retries
        self.chat_calls = 0
        self.chat_prompt_tokens = 0
        self.chat_completion_tokens = 0
        self.last_raw: str | None = None
        self.n_decisions = 0
        self.n_invalid_outputs = 0  # parse failed after retries → fallback ADD
        self.n_retries = 0

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = make_chat_client(self.chat_provider)
        return self._client

    def _ensure_embed_fn(self):
        if self.embed_fn is not None:
            return
        from harness.embed import Embedder

        # Embeddings always OpenAI direct — never OpenRouter
        self.embedder = Embedder(model=self.embed_model)
        self.embed_fn = lambda texts: self.embedder.embed_texts(list(texts))

    def invalidate_embed_cache(self, fact_ids: Sequence[str]) -> None:
        for fid in fact_ids:
            self.embeddings.pop(fid, None)

    def _retrieve(
        self, inp: ManagerInput, store: HierStore
    ) -> tuple[list[str], list[dict]]:
        self._ensure_embed_fn()
        cands = scoped_candidates(store, inp.project)
        if not cands:
            return [], []
        q = np.asarray(self.embed_fn([inp.text])[0], dtype=np.float32)
        ranked = rank_by_embed(
            q,
            cands,
            store=store,
            model=self.embed_model,
            embed_fn=self.embed_fn,
            cache=self.embeddings,
            top_m=self.top_m,
        )
        related = [f for _, f in ranked]
        return [f["id"] for f in related], related

    def _call_llm(self, inp: ManagerInput, related: list[dict]) -> str:
        related_lines = []
        for f in related:
            related_lines.append(
                f'- id="{f["id"]}" project={f.get("project")!r} '
                f'path={json.dumps(f.get("path"))} text={json.dumps(f["text"])}'
            )
        related_block = "\n".join(related_lines) if related_lines else "(none)"
        in_path = _default_path(inp.project, inp.path)
        user = (
            f"NEW fact:\n{json.dumps(inp.text)}\n\n"
            f"incoming_project: {inp.project!r}\n"
            f"incoming_path: {json.dumps(in_path)}\n\n"
            f"RELATED memories (project + global):\n{related_block}\n\n"
            "JSON ops list:"
        )
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
        )
        self.chat_calls += 1
        usage = resp.usage
        if usage:
            self.chat_prompt_tokens += int(usage.prompt_tokens or 0)
            self.chat_completion_tokens += int(usage.completion_tokens or 0)
        raw = (resp.choices[0].message.content or "").strip()
        self.last_raw = raw
        return raw

    def decide(self, inp: ManagerInput, store: HierStore) -> ManagerDecision:
        self.n_decisions += 1
        retrieve_ids, related = self._retrieve(inp, store)
        attempts = 1 + max(0, self.max_retries)
        last_err: Exception | None = None
        raw = ""
        for attempt in range(attempts):
            raw = self._call_llm(inp, related)
            try:
                ops = parse_llm_ops(
                    raw, related=related, inp=inp, allow_fallback=False
                )
                for op in ops:
                    op.retrieve_ids = list(retrieve_ids)
                return ManagerDecision(
                    ops=ops,
                    retrieve_ids=retrieve_ids,
                    reason="llmv0",
                    raw_llm=raw,
                )
            except ParseError as e:
                last_err = e
                if attempt + 1 < attempts:
                    self.n_retries += 1
        self.n_invalid_outputs += 1
        ops = _fallback_add(
            inp, f"llm_invalid_fallback_add:{getattr(last_err, 'kind', 'unknown')}"
        )
        for op in ops:
            op.retrieve_ids = list(retrieve_ids)
        return ManagerDecision(
            ops=ops,
            retrieve_ids=retrieve_ids,
            reason="llmv0_invalid_fallback",
            raw_llm=raw,
        )

    def apply(self, inp: ManagerInput, store: HierStore) -> list[dict[str, Any]]:
        logs = super().apply(inp, store)
        touched = [
            l["fact_id"]
            for l in logs
            if l.get("fact_id") and l["op"] in ("UPDATE", "DELETE", "ADD")
        ]
        self.invalidate_embed_cache([t for t in touched if t])
        return logs

    def stats(self) -> dict[str, Any]:
        # Priced per model. An unknown model reports None, never a number —
        # this used to hardcode gpt-4o-mini rates and silently under-reported a
        # gpt-4o run by 17x ($0.41 vs $6.84).
        price = CHAT_PRICE.get(self.chat_model)
        cost = (
            self.chat_prompt_tokens / 1e6 * price["in"]
            + self.chat_completion_tokens / 1e6 * price["out"]
        ) if price else None
        inv_rate = (
            self.n_invalid_outputs / self.n_decisions if self.n_decisions else 0.0
        )
        out: dict[str, Any] = {
            "manager": self.name,
            "chat_provider": self.chat_provider,
            "chat_model": self.chat_model,
            "prompt": self.prompt_name,
            "prompt_sha256_16": prompt_sha(self.system_prompt),
            "chat_calls": self.chat_calls,
            "chat_prompt_tokens": self.chat_prompt_tokens,
            "chat_completion_tokens": self.chat_completion_tokens,
            "chat_cost_usd_est": round(cost, 6) if cost is not None else None,
            "n_decisions": self.n_decisions,
            "n_retries": self.n_retries,
            "n_invalid_outputs": self.n_invalid_outputs,
            "invalid_output_rate": round(inv_rate, 4),
        }
        if self.embedder is not None and hasattr(self.embedder, "stats"):
            out["embed"] = self.embedder.stats()
        return out


def get_manager(name: str, **kwargs) -> MemoryManager:
    key = name.strip().lower()
    if key in ("alwaysadd", "always_add", "append_only"):
        return AlwaysADD()
    if key in ("rulev0", "rule_v0", "rule"):
        return RuleV0(**kwargs)
    if key in ("llmv0", "llm_v0", "llm"):
        return LLMv0(**kwargs)
    raise ValueError(f"unknown manager: {name}")
