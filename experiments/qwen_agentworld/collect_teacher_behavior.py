"""Cache BF16 teacher behavior for AgentWorld low-bit calibration.

This is local and free of external judge/API calls. The cache stores tokenized
AgentWorld prompts and top-k BF16 teacher logits for the last N positions, so
packed ternary/binary students can be evaluated without rerunning the teacher.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measure_group_damage import load_jobs, messages_for_job


def load_balanced_jobs(data_dir: Path, per_domain_limit: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*_test.jsonl")):
        taken = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                jobs.append(json.loads(line))
                taken += 1
                if taken >= per_domain_limit:
                    break
    return jobs


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--data-dir", default="/workspace/AgentWorldBench_smoke_1x7", type=Path)
    parser.add_argument("--out", default="/workspace/agentworld_runs/teacher_behavior_smoke_1x7_last8_top256.pt", type=Path)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit-jobs", type=int, default=7)
    parser.add_argument(
        "--per-domain-limit",
        type=int,
        help="take this many rows from each *_test.jsonl file; overrides --limit-jobs",
    )
    parser.add_argument("--last-n", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=256)
    args = parser.parse_args()

    if args.last_n <= 0:
        raise ValueError("--last-n must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    # AgentWorld prompts often have long system/tool state. Keep the latest user
    # state/action context when a calibration prompt exceeds max_length.
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = first_parameter_device(model)

    jobs = (
        load_balanced_jobs(args.data_dir, args.per_domain_limit)
        if args.per_domain_limit is not None
        else load_jobs(args.data_dir, args.limit_jobs)
    )

    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        text = tokenizer.apply_chat_template(messages_for_job(job), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=args.max_length)
        model_inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            keep = min(args.last_n, int(inputs["input_ids"].shape[-1]))
            outputs = model(**model_inputs, use_cache=False, logits_to_keep=keep)
            logits = outputs.logits[0, -keep:, :].float()
            top_values, top_indices = torch.topk(logits, k=min(args.top_k, logits.shape[-1]), dim=-1)
        rows.append(
            {
                "task": job.get("task"),
                "id": job.get("id"),
                "index": index,
                "tokens": int(inputs["input_ids"].shape[-1]),
                "input_ids": inputs["input_ids"][0].cpu().to(torch.int32),
                "attention_mask": inputs["attention_mask"][0].cpu().to(torch.int8),
                "top_values": top_values.cpu().to(torch.float16),
                "top_indices": top_indices.cpu().to(torch.int32),
            }
        )
        print(
            json.dumps(
                {
                    "index": index,
                    "task": job.get("task"),
                    "id": job.get("id"),
                    "tokens": int(inputs["input_ids"].shape[-1]),
                    "positions": int(top_values.shape[0]),
                    "top_k": int(top_values.shape[-1]),
                }
            ),
            flush=True,
        )

    payload = {
        "format": "qwen_agentworld_teacher_behavior_v1",
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "max_length": args.max_length,
        "limit_jobs": args.limit_jobs,
        "per_domain_limit": args.per_domain_limit,
        "last_n": args.last_n,
        "top_k": args.top_k,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    meta_path = args.out.with_suffix(args.out.suffix + ".json")
    meta_path.write_text(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "rows"
            }
            | {
                "rows": [
                    {
                        "task": row["task"],
                        "id": row["id"],
                        "tokens": row["tokens"],
                        "positions": int(row["top_values"].shape[0]),
                        "top_k": int(row["top_values"].shape[-1]),
                    }
                    for row in rows
                ]
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"out": str(args.out), "meta": str(meta_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
