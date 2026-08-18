#!/usr/bin/env python3
"""Upload placer SFT JSONL and fine-tune gpt-4o-mini via OpenAI API.

3050 6GB too small for local 8B LoRA — OpenAI FT is the runnable SFT path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from openai import OpenAI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "sft_placer")
    ap.add_argument("--base-model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--suffix", default="placer-v0")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "sft_placer_openai")
    ap.add_argument("--poll-s", type=int, default=30)
    ap.add_argument("--max-wait-s", type=int, default=3600)
    ap.add_argument(
        "--job-id",
        default=None,
        help="If set, only poll an existing job (no upload)",
    )
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    if args.job_id:
        job_id = args.job_id
        print("polling existing", job_id, flush=True)
    else:
        train_path = args.data / "train.jsonl"
        val_path = args.data / "val.jsonl"
        if not train_path.exists():
            raise SystemExit(f"missing {train_path} — run build_sft_placer.py")

        print("uploading train…", flush=True)
        train_file = client.files.create(
            file=train_path.open("rb"), purpose="fine-tune"
        )
        val_file = None
        if val_path.exists() and val_path.stat().st_size > 0:
            print("uploading val…", flush=True)
            val_file = client.files.create(
                file=val_path.open("rb"), purpose="fine-tune"
            )

        kwargs = {
            "training_file": train_file.id,
            "model": args.base_model,
            "suffix": args.suffix,
        }
        if val_file is not None:
            kwargs["validation_file"] = val_file.id

        print("creating fine-tune job…", flush=True)
        job = client.fine_tuning.jobs.create(**kwargs)
        job_id = job.id
        (args.out / "job_create.json").write_text(
            json.dumps(job.model_dump(), indent=2, default=str)
        )
        print("job_id=", job_id, flush=True)

    t0 = time.time()
    last_status = None
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        if status != last_status:
            print(f"status={status} model={job.fine_tuned_model}", flush=True)
            last_status = status
            (args.out / "job_status.json").write_text(
                json.dumps(job.model_dump(), indent=2, default=str)
            )
        if status in ("succeeded", "failed", "cancelled"):
            break
        if time.time() - t0 > args.max_wait_s:
            print("timeout waiting for job", flush=True)
            sys.exit(2)
        time.sleep(args.poll_s)

    result = {
        "job_id": job_id,
        "status": job.status,
        "fine_tuned_model": job.fine_tuned_model,
        "base_model": args.base_model,
        "error": str(job.error) if job.error else None,
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if job.status != "succeeded":
        sys.exit(1)


if __name__ == "__main__":
    main()
