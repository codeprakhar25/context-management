#!/usr/bin/env python3
"""Template-heavy (plan C) hard corpus for Flat vs Hier.

Shared collision tokens across real ml-resarch projects; project-specific payloads.
Writes facts.jsonl + queries.jsonl under --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Real dirs under ~/ml-resarch (stable ids = folder names)
PROJECTS: dict[str, dict] = {
    "slm-lab": {
        "title": "SLM Lab / visual evidence compression",
        "metric_name": "dppmm",
        "bench": "LVBench-T1",
        "score": "0.50",
        "method_drop": "OMP-style frame pick",
        "fail_cause": "wrong fps probe on stitch sheets",
        "decision": "fork B region-recall paused; stay on keyframe track",
        "pack_meaning": "contact-sheet frame pack size",
        "alt_score": "0.44",
        "alt_method": "uniform",
    },
    "video-understanding": {
        "title": "Video understanding probes",
        "metric_name": "vmm-gpt",
        "bench": "VideoMME-picks",
        "score": "0.50",
        "method_drop": "OMP budget on clip sampling",
        "fail_cause": "empty subtitle track in probe batch",
        "decision": "hold Modal uniform16 until GPU quota returns",
        "pack_meaning": "clip sample count per video",
        "alt_score": "0.41",
        "alt_method": "uniform",
    },
    "prompt-compiler-pilot": {
        "title": "Prompt-compiler pilot",
        "metric_name": "pcomp",
        "bench": "pilot-judge",
        "score": "0.50",
        "method_drop": "OMP candidate expansion",
        "fail_cause": "empty system prompt in runner",
        "decision": "archive pilot DONE null; no more reformulator",
        "pack_meaning": "candidate prompt count in search",
        "alt_score": "0.51",
        "alt_method": "uniform",
    },
    "context-management": {
        "title": "Context management / memory store",
        "metric_name": "hier-scope",
        "bench": "phase0-toy",
        "score": "0.50",
        "method_drop": "OMP path beam",
        "fail_cause": "missing OPENAI_API_KEY in shell (empty env)",
        "decision": "Flat vs Hier before Graphiti or RL",
        "pack_meaning": "retrieve k for memory pack",
        "alt_score": "0.52",
        "alt_method": "flat-ann",
    },
    "experiment": {
        "title": "Experiment paper / arxiv track",
        "metric_name": "acl-draft",
        "bench": "internal-review",
        "score": "0.50",
        "method_drop": "OMP citation sweep",
        "fail_cause": "wrong venue deadline row in VENUE_DEADLINES",
        "decision": "arxiv revision before new method claims",
        "pack_meaning": "related-work snippet pack size",
        "alt_score": "0.47",
        "alt_method": "uniform",
    },
    "reward-hacking-context": {
        "title": "Reward-hacking context study",
        "metric_name": "rh-detect",
        "bench": "blog-findings",
        "score": "0.50",
        "method_drop": "OMP feature bag",
        "fail_cause": "label leakage in preference pairs",
        "decision": "freeze FINDINGS.md; no new detector until leakage fixed",
        "pack_meaning": "context snippet k for judge",
        "alt_score": "0.39",
        "alt_method": "uniform",
    },
    "avatar-lab": {
        "title": "Avatar lab",
        "metric_name": "lip-sync",
        "bench": "avatar-val",
        "score": "0.50",
        "method_drop": "OMP phoneme align",
        "fail_cause": "mismatched sample rate on voice track",
        "decision": "block release until lip-sync gate green",
        "pack_meaning": "frame pack for review sheet",
        "alt_score": "0.43",
        "alt_method": "uniform",
    },
    "real-voice": {
        "title": "Real-voice pipeline",
        "metric_name": "asr-wer",
        "bench": "voice-holdout",
        "score": "0.50",
        "method_drop": "OMP n-best fusion",
        "fail_cause": "codec mismatch on upload batch",
        "decision": "prefer short utterances for next eval slice",
        "pack_meaning": "n-best list size",
        "alt_score": "0.55",
        "alt_method": "uniform",
    },
}

# Shared collision surface (Flat traps)
SHARED_F1 = "0.50"  # every project reports *.score as 0.50 on purpose for hard collision
TOPK = 32
SEED = 7


def fact(fid: str, text: str, project: str | None, tags: list[str], t: str) -> dict:
    if project is None:
        path = ["global"]
    else:
        path = ["project", project]
    return {
        "id": fid,
        "text": text,
        "path": path,
        "kind": "claim",
        "project": project,
        "t": t,
        "tags": tags,
        "episode_id": None,
    }


def build_facts() -> list[dict]:
    facts: list[dict] = []
    # global
    globals_ = [
        ("f_gl01", "User prefers TypeScript over JavaScript for harness and eval scripts.", ["prefer", "TypeScript"]),
        ("f_gl02", "User runs Python experiments with uv, not pip, unless a tool forbids it.", ["uv", "pip"]),
        ("f_gl03", "User wants temperature=0 for all grading and memory-eval reader calls.", ["temperature"]),
        ("f_gl04", "User OpenAI embedding model for memory work is text-embedding-3-small via direct API.", ["embed"]),
        ("f_gl05", f"User never treats seed={SEED} as cross-project shared experimental state.", ["seed", "isolation"]),
        ("f_gl06", "User prefers short gold answers for EM metrics in Phase-0 style evals.", ["EM"]),
        ("f_gl07", "User workspace root for ML experiments is ~/ml-resarch; project ids match folder names.", ["workspace"]),
        ("f_gl08", "User will add Graphiti only after Flat vs Hier holds on a hard multi-project corpus.", ["graphiti", "order"]),
    ]
    for i, (fid, text, tags) in enumerate(globals_):
        facts.append(fact(fid, text, None, tags, f"2026-04-01T10:{i:02d}:00Z"))

    for pi, (pid, meta) in enumerate(PROJECTS.items()):
        base_t = f"2026-0{(pi % 6) + 1:d}-15"
        # Template block — same shape every project, different payload
        block = [
            (
                f"f_{pid}__score",
                f"{pid}: {meta['metric_name']} scored F1={meta['score']} on {meta['bench']} (held-out).",
                ["F1", SHARED_F1, "score", meta["metric_name"]],
            ),
            (
                f"f_{pid}__uniform",
                f"{pid}: {meta['alt_method']} topk={TOPK} baseline scored F1={meta['alt_score']} on {meta['bench']}.",
                ["uniform", "topk", "k32", "baseline", "F1"],
            ),
            (
                f"f_{pid}__omp",
                f"{pid}: {meta['method_drop']} was dropped after a failed / net-negative ablation.",
                ["OMP", "dropped", "failed", "net-negative"],
            ),
            (
                f"f_{pid}__failrun",
                f"{pid}: failed run root cause was {meta['fail_cause']}.",
                ["failed", "run"],
            ),
            (
                f"f_{pid}__decision",
                f"{pid}: decision log — {meta['decision']}.",
                ["decision"],
            ),
            (
                f"f_{pid}__topk_mean",
                f"{pid}: topk={TOPK} means {meta['pack_meaning']} in this project only.",
                ["topk", "k32", "definition"],
            ),
            (
                f"f_{pid}__baseline_fail",
                f"{pid}: baseline topk={TOPK} pack/check failed project-specific QA gate (see failrun).",
                ["baseline", "topk", "k32", "failed"],
            ),
            (
                f"f_{pid}__seed",
                f"{pid}: ablation used seed={SEED}; do not merge results with other ml-resarch projects.",
                ["seed", "ablation", "baseline"],
            ),
            (
                f"f_{pid}__title",
                f"{pid}: project title/theme is '{meta['title']}'.",
                ["title"],
            ),
            (
                f"f_{pid}__netneg",
                f"{pid}: latest scorer/method swap was net-negative vs the project baseline.",
                ["net-negative", "scorer", "ablation"],
            ),
            (
                f"f_{pid}__bench",
                f"{pid}: primary bench tag is {meta['bench']}; metrics are not comparable to other projects' F1={SHARED_F1} lines.",
                ["bench", "F1", SHARED_F1],
            ),
            (
                f"f_{pid}__next",
                f"{pid}: next action is implied by decision — {meta['decision']}",
                ["next", "decision"],
            ),
        ]
        for j, (fid, text, tags) in enumerate(block):
            facts.append(
                fact(fid, text, pid, tags, f"{base_t}T{12+j:02d}:00:00Z")
            )
    return facts


def build_queries(facts: list[dict]) -> list[dict]:
    by_id = {f["id"]: f for f in facts}
    qs: list[dict] = []
    n = 0

    def add(**kw):
        nonlocal n
        n += 1
        q = {"id": f"q_{n:03d}", **kw}
        # validate golds exist
        for g in q["gold_ids"]:
            assert g in by_id, g
        qs.append(q)

    # --- per-project locals + hard locals ---
    for pid, meta in PROJECTS.items():
        add(
            text=f"In {pid}, what F1 did {meta['metric_name']} get on {meta['bench']}?",
            type="local",
            project=pid,
            gold_ids=[f"f_{pid}__score"],
            gold_answer=meta["score"],
            distractor_ids=[f"f_{p}__score" for p in PROJECTS if p != pid],
            notes="Easy local score; all projects share F1=0.50 surface.",
        )
        add(
            text=f"What does F1={SHARED_F1} refer to in {pid}?",
            type="hard_local",
            project=pid,
            gold_ids=[f"f_{pid}__score", f"f_{pid}__bench"],
            gold_answer=f"{meta['metric_name']} on {meta['bench']}",
            distractor_ids=[f"f_{p}__score" for p in PROJECTS if p != pid],
            notes="HARD: every project has F1=0.50.",
        )
        add(
            text=f"In {pid}, why was baseline topk={TOPK} considered a failure?",
            type="hard_local",
            project=pid,
            gold_ids=[f"f_{pid}__baseline_fail", f"f_{pid}__failrun"],
            gold_answer=meta["fail_cause"],
            distractor_ids=[f"f_{p}__baseline_fail" for p in PROJECTS if p != pid],
            notes="HARD: baseline topk=32 failed everywhere.",
        )
        add(
            text=f"In {pid}, what does topk={TOPK} mean?",
            type="hard_local",
            project=pid,
            gold_ids=[f"f_{pid}__topk_mean"],
            gold_answer=meta["pack_meaning"],
            distractor_ids=[f"f_{p}__topk_mean" for p in PROJECTS if p != pid],
            notes="HARD: topk=32 different definition per project.",
        )
        add(
            text=f"What was dropped in {pid} after a failed/net-negative ablation (OMP-related)?",
            type="hard_local",
            project=pid,
            gold_ids=[f"f_{pid}__omp"],
            gold_answer=meta["method_drop"],
            distractor_ids=[f"f_{p}__omp" for p in PROJECTS if p != pid],
            notes="HARD: OMP token in every project.",
        )
        add(
            text=f"What is the decision-log next step for {pid}?",
            type="local",
            project=pid,
            gold_ids=[f"f_{pid}__decision"],
            gold_answer=meta["decision"],
            distractor_ids=[],
            notes="Local decision.",
        )
        add(
            text=f"What caused the failed run in {pid}?",
            type="hard_local",
            project=pid,
            gold_ids=[f"f_{pid}__failrun"],
            gold_answer=meta["fail_cause"],
            distractor_ids=[f"f_{p}__failrun" for p in PROJECTS if p != pid],
            notes="HARD: failed run in all projects.",
        )
        add(
            text=f"{pid}: {meta['alt_method']} topk={TOPK} baseline F1 on {meta['bench']}?",
            type="local",
            project=pid,
            gold_ids=[f"f_{pid}__uniform"],
            gold_answer=meta["alt_score"],
            distractor_ids=[f"f_{p}__uniform" for p in PROJECTS if p != pid],
            notes="Local alt baseline score.",
        )

    # globals
    add(
        text="What language does the user prefer for harness scripts?",
        type="global",
        project=None,
        gold_ids=["f_gl01"],
        gold_answer="TypeScript",
        distractor_ids=[],
        notes="global",
    )
    add(
        text="Does the user use pip or uv for Python experiments?",
        type="global",
        project=None,
        gold_ids=["f_gl02"],
        gold_answer="uv",
        distractor_ids=[],
        notes="global",
    )
    add(
        text="What OpenAI embedding model is used for memory work?",
        type="global",
        project=None,
        gold_ids=["f_gl04"],
        gold_answer="text-embedding-3-small",
        distractor_ids=[],
        notes="global",
    )
    add(
        text="Should seed=7 be treated as shared state across ml-resarch projects?",
        type="global",
        project=None,
        gold_ids=["f_gl05"],
        gold_answer="no",
        distractor_ids=[f"f_{p}__seed" for p in PROJECTS],
        notes="global isolation",
    )

    # mixed: project session + global
    for pid in ("slm-lab", "context-management", "prompt-compiler-pilot"):
        add(
            text=f"While working in {pid}, what package manager should Python runs use?",
            type="mixed",
            project=pid,
            gold_ids=["f_gl02"],
            gold_answer="uv",
            distractor_ids=[],
            notes="mixed global under project scope",
        )
        add(
            text=f"In a {pid} session, what temperature for grading/reader calls?",
            type="mixed",
            project=pid,
            gold_ids=["f_gl03"],
            gold_answer="0",
            distractor_ids=[],
            notes="mixed",
        )

    # adversarial
    add(
        text="Is seed=7 a shared experimental protocol across slm-lab, video-understanding, and prompt-compiler-pilot?",
        type="adversarial",
        project="slm-lab",
        gold_ids=["f_gl05", "f_slm-lab__seed"],
        gold_answer="no; coincidence / must not treat as shared state",
        distractor_ids=["f_video-understanding__seed", "f_prompt-compiler-pilot__seed"],
        notes="adversarial isolation",
    )
    add(
        text="slm-lab baseline ablation — should we reuse reward-hacking-context baseline numbers?",
        type="adversarial",
        project="slm-lab",
        gold_ids=["f_slm-lab__seed"],
        gold_answer="no; do not merge results with other ml-resarch projects",
        distractor_ids=["f_reward-hacking-context__uniform", "f_reward-hacking-context__score"],
        notes="adversarial cross-project refuse",
    )
    # intentional Hier miss: cross-project compare under one scope
    add(
        text="Compare uniform/flat topk=32 F1 for slm-lab vs prompt-compiler-pilot — give both numbers.",
        type="adversarial",
        project="slm-lab",
        gold_ids=["f_slm-lab__uniform", "f_prompt-compiler-pilot__uniform"],
        gold_answer=f"slm-lab {PROJECTS['slm-lab']['alt_score']}; prompt-compiler-pilot {PROJECTS['prompt-compiler-pilot']['alt_score']}",
        distractor_ids=["f_video-understanding__uniform"],
        notes="EXPECTED Hier recall miss on second gold; Flat may get both.",
    )
    add(
        text="Which project archived a reformulator/pilot as DONE null?",
        type="hard_local",
        project="prompt-compiler-pilot",
        gold_ids=["f_prompt-compiler-pilot__decision"],
        gold_answer="prompt-compiler-pilot",
        distractor_ids=["f_reward-hacking-context__decision", "f_context-management__decision"],
        notes="hard identity among decision logs",
    )
    add(
        text="Where is the user ML workspace root?",
        type="global",
        project=None,
        gold_ids=["f_gl07"],
        gold_answer="~/ml-resarch",
        distractor_ids=[],
        notes="global workspace",
    )

    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "hard_v1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    facts = build_facts()
    queries = build_queries(facts)

    with (args.out / "facts.jsonl").open("w") as f:
        for row in facts:
            f.write(json.dumps(row) + "\n")
    with (args.out / "queries.jsonl").open("w") as f:
        for row in queries:
            f.write(json.dumps(row) + "\n")

    meta = {
        "name": "hard_v1",
        "method": "template_C",
        "projects": list(PROJECTS.keys()),
        "n_facts": len(facts),
        "n_queries": len(queries),
        "shared_tokens": {"F1": SHARED_F1, "topk": TOPK, "seed": SEED, "OMP": True},
        "note": "All primary scores intentionally F1=0.50 for collision; payloads differ.",
    }
    (args.out / "META.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
