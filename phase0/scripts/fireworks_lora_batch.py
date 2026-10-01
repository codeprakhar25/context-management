#!/usr/bin/env python3
"""Orchestrate many Fireworks LoRA adapters: one per vault (R5) plus the A' pooled
folder-split adapter (R1), then serve them all on ONE multi-LoRA deployment.

Why a script: R5 is 43 vaults x 2 epoch settings (+ re-served pooled adapters),
far too many objects to drive by hand, and a half-finished batch must resume
without re-uploading or re-training. Every id and every API response worth
keeping lands in runs/fireworks_lora_batch_state.json.

Subcommands (mutating ones are DRY RUNS unless --yes is given; they print the
exact HTTP calls they would make):
  plan        what would be uploaded / trained, rows, token + cost estimate
  upload      create + upload + validate datasets (skips ones that exist)
  train       one SFT job per dataset with the corpus recipe, per --epochs
  status      poll jobs (read-only), refresh state
  deploy      one BF16 --enable-addons deployment, load adapters (+ pooled ones)
  model-map   write {vault: route} JSON for vault_lora_gate.py --model-map
  teardown    unload adapters, delete deployment, confirm the list is empty

REST only, so the API key in .env is all it needs. Uploads reuse
scripts/fireworks_upload_dataset.sh. Training data is third-party note
text, so datasets are only ever read from gitignored data/ dirs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "fireworks_lora_batch_state.json"
API = "https://api.fireworks.ai/v1"
ACCT = os.environ.get("FIREWORKS_ACCOUNT_ID", "prakharkhatri123-edp")

BASE_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
# The only version of this shape that advertises MULTI_LORA (BF16, 1xH200). The
# bare shape name has no validated version and is rejected with a 403.
SHAPE = "accounts/fireworks/deploymentShapes/rft-llama-v3p1-8b-instruct/versions/qcaqooud"
RECIPE = {"loraRank": 16, "learningRate": 1e-4, "maxContextLength": 16384}  # no earlyStop: rejected
ID_MAX = 63  # DNS-label style limit; NOT confirmed in the docs, assumed conservative
TAG = {"B": "b", "A_prime": "ap"}
PER_VAULT = ROOT / "data" / "sft_placer_per_vault"
FOLD = ROOT / "data" / "sft_placer_vaultsA_fold"
# Already-trained pooled adapters, re-scored on the same deployment as a control.
POOLED = {"B": "placer-vaultb-item-llama31-8b", "A_prime": "placer-vaultsa-item-llama31-8b"}
DEFAULT_DEP = "placer-multilora-llama31-8b"


def slug(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")


def legal_id(base: str, reserve: int, taken: dict[str, str], owner: str) -> str:
    """Lowercase [a-z0-9-], starts with a letter, base+reserve <= ID_MAX, and
    never shared between two owners (hash suffix on truncation or collision)."""
    base = slug(base)
    h = hashlib.sha1(owner.encode()).hexdigest()[:6]
    if len(base) + reserve > ID_MAX:
        base = base[: ID_MAX - reserve - 7].rstrip("-") + "-" + h
    if taken.get(base, owner) != owner:
        base = base[: ID_MAX - reserve - 7].rstrip("-") + "-" + h
    taken[base] = owner
    return base


def items(corpus: str, epochs: list[int]) -> list[dict]:
    """Every adapter to train: per-vault for B / A_prime, one pooled for A_prime_fold."""
    out, taken = [], {}
    for c in ("B", "A_prime"):
        if corpus not in (c, "all"):
            continue
        for d in sorted(p for p in (PER_VAULT / c).glob("*") if (p / "train.jsonl").exists()):
            base = legal_id(f"placer-pv-{TAG[c]}-{d.name}", len("-train") + 4, taken, f"{c}/{d.name}")
            out.append({"key": f"{c}/{d.name}", "corpus": c, "vault": d.name, "dir": d,
                        "train_ds": f"{base}-train", "val_ds": f"{base}-val",
                        "models": {e: f"{base}-e{e}" for e in epochs}})
    if corpus in ("A_prime_fold", "all"):
        # R1 LoRA arm: pooled A' folder-disjoint split, corpus recipe (epochs 1 only)
        out.append({"key": "A_prime_fold/pooled", "corpus": "A_prime_fold", "vault": "pooled",
                    "dir": FOLD, "train_ds": "placer-vaultsa-fold-train", "val_ds": "placer-vaultsa-fold-val",
                    "models": {1: "placer-vaultsa-fold-llama31-8b"}})
    return out


def load_key() -> str:
    for k in ("FIREWORKS_API_KEY", "FIREWORKS_API"):
        if os.environ.get(k):
            return os.environ[k]
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?(FIREWORKS_API_KEY|FIREWORKS_API)\s*=\s*(.*)", line)
            if m:
                return m.group(2).strip().strip("'\"")
    sys.exit("no FIREWORKS_API_KEY / FIREWORKS_API in env or .env")


def api(method: str, path: str, body: dict | None = None, query: dict | None = None,
        retries: int = 3) -> tuple[int, dict]:
    url = f"{API}/accounts/{ACCT}/{path}"
    if query:
        url += "?" + "&".join(f"{k}={v}" for k, v in query.items())
    data = json.dumps(body).encode() if body is not None else None
    for i in range(retries):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode()
                return r.status, json.loads(txt) if txt.strip() else {}
        except urllib.error.HTTPError as e:
            txt = e.read().decode()
            try:
                j = json.loads(txt)
            except ValueError:
                j = {"message": txt[:300]}
            if e.code >= 500 and i < retries - 1:
                time.sleep(2 ** i)
                continue
            return e.code, j
        except urllib.error.URLError:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def mutate(args, method: str, path: str, body=None, query=None):
    """Mutating call: print it; execute only with --yes. Returns (status, json) or None."""
    q = ("?" + "&".join(f"{k}={v}" for k, v in query.items())) if query else ""
    print(f"  {method} {API}/accounts/{ACCT}/{path}{q}"
          + (f"\n    {json.dumps(body)}" if body is not None else ""))
    if not args.yes:
        return None
    return api(method, path, body, query)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"account": ACCT, "datasets": {}, "jobs": {}, "deployment": None, "loaded": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2, sort_keys=True) + "\n")


def n_rows_tok(path: Path) -> tuple[int, int]:
    n = tok = 0
    with path.open() as f:
        for line in f:
            n += 1
            tok += len(line) // 4  # chars/4, same estimate the findings used
    return n, tok


def selected(args) -> list[dict]:
    its = items(args.corpus, args.epochs)
    if args.only:
        its = [i for i in its if args.only in i["key"]]
    return its[: args.limit] if args.limit else its


def remote_exists(coll: str, rid: str) -> dict | None:
    code, j = api("GET", f"{coll}/{rid}")
    return j if code == 200 else None


# --- subcommands ---

def cmd_plan(args) -> None:
    st = load_state()
    its = selected(args)
    tot_tok = 0
    print(f"{'adapter set':44s} {'train':>6s} {'val':>5s} {'tok~':>9s}  datasets / models")
    for it in its:
        ntr, tok = n_rows_tok(it["dir"] / ("train.jsonl"))
        nva, _ = n_rows_tok(it["dir"] / "val.jsonl")
        ep = [e for e in it["models"] if e in args.epochs or it["corpus"] == "A_prime_fold"]
        tot_tok += tok * sum(ep)  # tokens billed scale with epochs
        tag = ""
        if not args.offline:
            have = [d for d in (it["train_ds"], it["val_ds"]) if remote_exists("datasets", d)]
            tag = f" remote_datasets={len(have)}/2"
        print(f"{it['key']:44s} {ntr:6d} {nva:5d} {tok:9d}  {it['train_ds']} -> "
              f"{', '.join(it['models'][e] for e in ep)}{tag}")
    print(f"\n{len(its)} dataset pairs; training tokens ~{tot_tok/1e6:.1f}M over epochs {args.epochs} "
          f"(chars/4)\nSFT cost ~${tot_tok/1e6*args.price:.0f} at ${args.price}/M tokens "
          f"(page lists $0.50-$2.00/M for <=16B depending on method; confirm in the console)")
    print(f"state file: {STATE} ({len(st['datasets'])} datasets, {len(st['jobs'])} jobs recorded)")


def cmd_upload(args) -> None:
    st = load_state()
    todo = 0
    for it in selected(args):
        for ds, f in ((it["train_ds"], "train.jsonl"), (it["val_ds"], "val.jsonl")):
            r = remote_exists("datasets", ds)
            if r and r.get("state") == "READY":
                st["datasets"][ds] = {"path": str((it["dir"] / f).relative_to(ROOT)), "state": "READY"}
                continue
            if r:
                print(f"  {ds}: exists in state {r.get('state')}, not re-creating; delete or fix by hand")
                continue
            todo += 1
            cmd = ["bash", "scripts/fireworks_upload_dataset.sh", ds, str((it["dir"] / f).relative_to(ROOT))]
            print("  RUN", " ".join(cmd), "   (create -> getUploadEndpoint -> PUT -> validateUpload)")
            if args.yes:
                if subprocess.run(cmd, cwd=ROOT).returncode:
                    print(f"  FAILED {ds}", file=sys.stderr)
                    continue
                st["datasets"][ds] = {"path": str((it["dir"] / f).relative_to(ROOT)), "state": "READY"}
                save_state(st)
    print(f"{todo} datasets to upload" + ("" if args.yes else "  [DRY RUN: pass --yes]"))
    save_state(st) if args.yes else None


def cmd_train(args) -> None:
    st = load_state()
    n = 0
    for it in selected(args):
        for e, mid in it["models"].items():
            if it["corpus"] != "A_prime_fold" and e not in args.epochs:
                continue
            if mid in st["jobs"] or (args.yes and remote_exists("supervisedFineTuningJobs", mid)):
                print(f"  skip {mid}: job exists")
                continue
            if args.yes:
                bad = [d for d in (it["train_ds"], it["val_ds"])
                       if (remote_exists("datasets", d) or {}).get("state") != "READY"]
                if bad:
                    print(f"  skip {mid}: dataset not READY: {bad} (run upload)", file=sys.stderr)
                    continue
            # job id == output model id: outputModel is left unset ("job ID used if unspecified")
            body = {"dataset": f"accounts/{ACCT}/datasets/{it['train_ds']}",
                    "evaluationDataset": f"accounts/{ACCT}/datasets/{it['val_ds']}",
                    "baseModel": BASE_MODEL, "epochs": e, "displayName": mid, **RECIPE}
            n += 1
            r = mutate(args, "POST", "supervisedFineTuningJobs", body, {"supervisedFineTuningJobId": mid})
            if r:
                code, j = r
                if code != 200:
                    print(f"  FAILED {mid}: {code} {j.get('message')}", file=sys.stderr)
                    continue
                st["jobs"][mid] = {"key": it["key"], "epochs": e, "state": j.get("state"),
                                   "dataset": it["train_ds"], "model": mid}
                save_state(st)
    print(f"{n} jobs" + ("" if args.yes else "  [DRY RUN: pass --yes]"))


def cmd_status(args) -> None:
    st = load_state()
    counts: dict[str, int] = {}
    for jid, j in sorted(st["jobs"].items()):
        code, r = api("GET", f"supervisedFineTuningJobs/{jid}")
        s = r.get("state", f"HTTP{code}") if code == 200 else f"HTTP{code}"
        j["state"] = s
        j["outputModel"] = r.get("outputModel", j.get("outputModel"))
        counts[s] = counts.get(s, 0) + 1
        pct = (r.get("jobProgress") or {}).get("percent", "")
        print(f"  {jid:52s} {s} {pct}" + (f"  {(r.get('status') or {}).get('message','')}"
                                           if s.endswith("FAILED") else ""))
    save_state(st)
    print(counts or "no jobs recorded")


def deployed_models(dep: str) -> list[dict]:
    out, tok = [], ""
    while True:
        code, j = api("GET", "deployedModels", query={"pageSize": 200, **({"pageToken": tok} if tok else {})})
        out += [d for d in j.get("deployedModels", []) if d.get("deployment", "").endswith(f"/{dep}")]
        tok = j.get("nextPageToken")
        if code != 200 or not tok:
            return out


def cmd_deploy(args) -> None:
    st = load_state()
    dep = args.deployment
    body = {"baseModel": BASE_MODEL, "displayName": dep, "enableAddons": True,
            "deploymentShape": args.shape, "precision": "BF16",
            "minReplicaCount": 1, "maxReplicaCount": 1}
    if args.accelerator:  # H200 capacity fallback; docs say acceleratorType is meant for no-shape use
        body.update(acceleratorType=args.accelerator, acceleratorCount=1)
    if args.peft_cache:
        body["numPeftDeviceCached"] = args.peft_cache
    exists = remote_exists("deployments", dep)
    if exists:
        print(f"  deployment {dep} exists: {exists.get('state')}")
    else:
        q = {"deploymentId": dep, **({"validateOnly": "true"} if args.validate_only else {})}
        r = mutate(args, "POST", "deployments", body, q)
        if r and r[0] != 200:
            sys.exit(f"deployment create failed: {r[0]} {r[1].get('message')}")
        if r and not args.validate_only:
            st["deployment"] = {"id": dep, "created": time.strftime("%FT%TZ", time.gmtime())}
            save_state(st)
            while True:
                s = (remote_exists("deployments", dep) or {})
                print(f"  {time.strftime('%H:%M:%S')} {s.get('state')} {(s.get('status') or {}).get('message','')}")
                if s.get("state") == "READY":
                    break
                if s.get("state") in ("FAILED", "DELETED"):
                    sys.exit("deployment failed; for H200 shortage retry with --accelerator NVIDIA_H100_80GB")
                time.sleep(15)
        if args.validate_only:
            return
    models = []
    if not args.no_trained:
        for it in selected(args):
            for e, m in it["models"].items():
                if it["corpus"] != "A_prime_fold" and e not in args.epochs:
                    continue
                if not args.yes or st["jobs"].get(m, {}).get("state") == "JOB_STATE_COMPLETED":
                    models.append(m)
    if not args.no_pooled:
        models += list(POOLED.values())
    models += args.extra_model
    have = {d.get("model", "").split("/")[-1] for d in deployed_models(dep)} if exists or args.yes else set()
    for m in dict.fromkeys(models):
        if m in have:
            print(f"  already loaded: {m}")
            continue
        r = mutate(args, "POST", "deployedModels",
                   {"model": f"accounts/{ACCT}/models/{m}", "deployment": f"accounts/{ACCT}/deployments/{dep}"})
        if r:
            if r[0] != 200:
                print(f"  load FAILED {m}: {r[0]} {r[1].get('message')}", file=sys.stderr)
                continue
            st["loaded"][m] = r[1].get("name", "")
            save_state(st)
    if not args.yes:
        print("  [DRY RUN: pass --yes]  (real run loads only adapters whose job is COMPLETED in state)")


def route(model: str, dep: str) -> str:
    return f"accounts/{ACCT}/models/{model}#accounts/{ACCT}/deployments/{dep}"


def cmd_model_map(args) -> None:
    dep = args.deployment
    for c in ("B", "A_prime"):
        if args.corpus not in (c, "all"):
            continue
        m = {it["vault"]: route(it["models"][args.map_epochs], dep)
             for it in items(c, [args.map_epochs])}
        out = args.out or ROOT / "runs" / f"fireworks_lora_vault_map_{c}_e{args.map_epochs}.json"
        out.write_text(json.dumps(m, indent=2) + "\n")
        print(f"wrote {out} ({len(m)} vaults)\n  pooled control route for --model: {route(POOLED[c], dep)}")


def cmd_teardown(args) -> None:
    st = load_state()
    dep = args.deployment
    for d in deployed_models(dep):
        mutate(args, "DELETE", f"deployedModels/{d['name'].split('/')[-1]}")  # unload-lora
    mutate(args, "DELETE", f"deployments/{dep}", query={"ignoreChecks": "true"})
    if args.yes:
        for _ in range(40):
            code, j = api("GET", "deployments")
            left = [d["name"].split("/")[-1] + ":" + d.get("state", "") for d in j.get("deployments", [])
                    if d.get("state") != "DELETED"]
            if not left:
                break
            time.sleep(15)
        print("deployments remaining:", left or "none (confirmed empty)")
        st["deployment"], st["loaded"] = None, {}
        save_state(st)
        sys.exit(1 if left else 0)
    code, j = api("GET", "deployments")
    print(f"  [DRY RUN: pass --yes]  current deployments: {[d['name'] for d in j.get('deployments', [])]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, mutating=True, sel=True):
        if sel:
            p.add_argument("--corpus", choices=["B", "A_prime", "A_prime_fold", "all"], default="all")
            p.add_argument("--epochs", type=int, nargs="+", default=[1], help="e.g. --epochs 1 3")
            p.add_argument("--only", help="substring filter on corpus/vault key")
            p.add_argument("--limit", type=int, help="first N items (smoke)")
        if mutating:
            p.add_argument("--yes", action="store_true", help="actually call the API (default: dry run)")

    p = sub.add_parser("plan"); common(p, False)
    p.add_argument("--offline", action="store_true", help="skip read-only remote existence checks")
    p.add_argument("--price", type=float, default=0.5, help="$/M training tokens for the estimate")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("upload"); common(p); p.set_defaults(fn=cmd_upload)
    p = sub.add_parser("train"); common(p); p.set_defaults(fn=cmd_train)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("deploy"); common(p)
    p.add_argument("--deployment", default=DEFAULT_DEP)
    p.add_argument("--shape", default=SHAPE)
    p.add_argument("--accelerator", help="e.g. NVIDIA_H100_80GB when H200 capacity is short")
    p.add_argument("--peft-cache", type=int, help="numPeftDeviceCached (GPU-resident adapters)")
    p.add_argument("--validate-only", action="store_true", help="server-side validateOnly create, nothing made")
    p.add_argument("--no-pooled", action="store_true", help="skip the two existing pooled adapters")
    p.add_argument("--no-trained", action="store_true", help="load only pooled/extra adapters")
    p.add_argument("--extra-model", action="append", default=[], help="more existing model ids to load")
    p.set_defaults(fn=cmd_deploy)
    p = sub.add_parser("model-map")
    p.add_argument("--corpus", choices=["B", "A_prime", "all"], default="all")
    p.add_argument("--map-epochs", type=int, default=1)
    p.add_argument("--deployment", default=DEFAULT_DEP)
    p.add_argument("--out", type=Path, help="default runs/fireworks_lora_vault_map_<corpus>_e<N>.json")
    p.set_defaults(fn=cmd_model_map)
    p = sub.add_parser("teardown"); common(p, sel=False)
    p.add_argument("--deployment", default=DEFAULT_DEP)
    p.set_defaults(fn=cmd_teardown)

    args = ap.parse_args()
    if args.cmd == "model-map" and args.out and args.corpus == "all":
        ap.error("--out needs a single --corpus")
    args.fn(args)


if __name__ == "__main__":
    main()
