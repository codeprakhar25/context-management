"""Modal LoRA SFT for path placer (OpenAI FT unavailable for this org).

Base: Qwen2.5-1.5B-Instruct (ungated, fits A10 quickly).
Data: data/sft_placer/train.jsonl + val.jsonl (OpenAI messages format).

  modal run modal_sft_placer.py
  modal run modal_sft_placer.py --eval-only
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import modal

APP = modal.App("context-mgmt-placer-sft")
VOL = modal.Volume.from_name("placer-sft-vol", create_if_missing=True)
ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "peft==0.13.2",
        "trl==0.12.1",
        "bitsandbytes==0.44.1",
        "sentencepiece",
        "protobuf",
    )
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _parse_path(raw: str) -> list[str] | None:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    path = obj.get("path")
    if not isinstance(path, list) or not all(isinstance(x, str) for x in path):
        return None
    return path


def _soft(gold: list[str], pred: list[str] | None) -> dict:
    if not pred:
        return {
            "exact": False,
            "soft_hit": False,
            "branch_ok": False,
            "soft_relation": "empty",
        }
    exact = gold == pred
    same_root = bool(gold) and bool(pred) and gold[0] == pred[0]
    if exact:
        rel = "exact"
    elif len(gold) <= len(pred) and pred[: len(gold)] == gold:
        rel = "gold_prefix"
    elif len(pred) <= len(gold) and gold[: len(pred)] == pred:
        rel = "pred_prefix"
    elif same_root:
        rel = "same_root"
    else:
        rel = "diff_root"
    return {
        "exact": exact,
        "soft_hit": rel in ("exact", "gold_prefix", "pred_prefix"),
        "branch_ok": same_root,
        "soft_relation": rel,
        "pred_path": pred,
    }


@APP.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    memory=65536,
    volumes={"/vol": VOL},
)
def train_and_eval(
    train_jsonl: str,
    val_jsonl: str,
    val_meta_jsonl: str,
    epochs: int = 3,
    lr: float = 2e-4,
    eval_only: bool = False,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    out_dir = Path("/vol/placer_qwen15_lora")
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_msgs(text: str) -> list[dict]:
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line)["messages"])
        return rows

    train_msgs = load_msgs(train_jsonl)
    val_msgs = load_msgs(val_jsonl)
    val_meta = [
        json.loads(l) for l in val_meta_jsonl.splitlines() if l.strip()
    ]

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def to_text(messages: list[dict]) -> str:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    def tokenize_rows(msgs: list[list[dict]]) -> Dataset:
        texts = [to_text(m) for m in msgs]
        ds = Dataset.from_list([{"text": t} for t in texts])

        def _tok(batch):
            return tok(
                batch["text"],
                truncation=True,
                max_length=1536,
                padding=False,
            )

        return ds.map(_tok, batched=True, remove_columns=["text"])

    if not eval_only:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        peft_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_cfg)
        train_ds = tokenize_rows(train_msgs)
        collator = DataCollatorForLanguageModeling(tok, mlm=False)

        targs = TrainingArguments(
            output_dir=str(out_dir / "ckpt"),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=lr,
            logging_steps=5,
            save_strategy="epoch",
            # skip trainer.evaluate — long seqs OOM on logits; we gen-eval below
            eval_strategy="no",
            bf16=True,
            report_to=[],
            remove_unused_columns=False,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            data_collator=collator,
        )
        trainer.train()
        trainer.model.save_pretrained(str(out_dir / "adapter"))
        tok.save_pretrained(str(out_dir / "adapter"))
        del trainer
        del model
        torch.cuda.empty_cache()
        VOL.commit()

    # ---- eval ----
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(out_dir / "adapter"))
    model.eval()

    rows = []
    for messages, meta in zip(val_msgs, val_meta):
        # prompt = all but assistant
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt = tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = _parse_path(gen)
        gold = meta["gold_path"]
        sc = _soft(gold, pred)
        rows.append(
            {
                "id": meta["id"],
                "gold_path": gold,
                "pred_path": pred,
                "raw": gen,
                **sc,
            }
        )

    n = len(rows) or 1
    summary = {
        "base_model": BASE_MODEL,
        "n_val": len(rows),
        "path_exact": sum(r["exact"] for r in rows) / n,
        "path_soft": sum(r["soft_hit"] for r in rows) / n,
        "branch_ok": sum(r["branch_ok"] for r in rows) / n,
        "adapter": str(out_dir / "adapter"),
        "epochs": epochs,
        "eval_only": eval_only,
    }
    (out_dir / "val_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (out_dir / "val_summary.json").write_text(json.dumps(summary, indent=2))
    VOL.commit()
    return {"summary": summary, "results": rows}


@APP.local_entrypoint()
def main(eval_only: bool = False, epochs: int = 3):
    data = ROOT / "data" / "sft_placer"
    train = (data / "train.jsonl").read_text()
    val = (data / "val.jsonl").read_text()
    meta = (data / "val_meta.jsonl").read_text()
    out = train_and_eval.remote(
        train, val, meta, epochs=epochs, eval_only=eval_only
    )
    local_out = ROOT / "runs" / "sft_placer_modal"
    local_out.mkdir(parents=True, exist_ok=True)
    (local_out / "val_summary.json").write_text(
        json.dumps(out["summary"], indent=2)
    )
    (local_out / "val_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out["results"]) + "\n"
    )
    # baseline note: compare to gpt-4o with dirs on same val ids later
    print(json.dumps(out["summary"], indent=2))
    print("wrote", local_out)
