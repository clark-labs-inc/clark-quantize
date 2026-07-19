"""Run long-generation survival gates for Qwen AgentWorld-style prompts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import transformers

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_onpolicy_survival_cache import (  # noqa: E402
    collapse_summary,
    first_parameter_device,
    load_model,
    messages_for_job,
)
from weight_patch import apply_weight_patch  # noqa: E402


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counts = Counter(row["collapse"]["failure_kind"] for row in rows)
    valid = sum(1 for row in rows if row["collapse"].get("valid_observation"))
    parseable = sum(1 for row in rows if row["collapse"].get("official_parseable"))
    generated_tokens = [int(row["generated_tokens"]) for row in rows]
    return {
        "rows": len(rows),
        "valid_observation": valid,
        "parseable": parseable,
        "failure_counts": dict(failure_counts),
        "min_generated_tokens": min(generated_tokens) if generated_tokens else 0,
        "max_generated_tokens": max(generated_tokens) if generated_tokens else 0,
        "avg_generated_tokens": sum(generated_tokens) / max(len(generated_tokens), 1),
    }


def load_agentworld_rows(data_dir: Path, max_rows: int) -> list[dict[str, Any]]:
    """Load AgentWorld-style JSONL rows without importing the paid eval harness."""
    if data_dir.is_file():
        files = [data_dir]
    else:
        files = sorted(data_dir.glob("*_test.jsonl"))
    if not files:
        raise FileNotFoundError(f"no *_test.jsonl files under {data_dir}")

    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= max_rows:
                    return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--data-dir", default="/workspace/AgentWorldBench_round8_fixed10_20260629", type=Path)
    parser.add_argument("--weight-patch", action="append", type=Path)
    parser.add_argument("--patch-exclude-regex")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lengths", default="4096,8192,16384,32768")
    parser.add_argument("--max-rows", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--stop-on-first-length-failure", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_agentworld_rows(args.data_dir, args.max_rows)
    if not jobs:
        raise RuntimeError(f"no AgentWorld gate rows loaded from {args.data_dir}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.truncation_side = args.truncation_side

    model = load_model(args.model_dir)
    patch_reports = []
    for patch in args.weight_patch or []:
        patch_reports.append(apply_weight_patch(model, patch, exclude_regex=args.patch_exclude_regex))
    for param in model.parameters():
        param.requires_grad_(False)
    device = first_parameter_device(model)

    all_summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for length in parse_int_csv(args.lengths):
        length_rows: list[dict[str, Any]] = []
        out_jsonl = args.out_dir / f"long_gate_{length}.jsonl"
        print(json.dumps({"event": "length_start", "length": length, "rows": len(jobs)}), flush=True)
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for index, job in enumerate(jobs, start=1):
                prompt_text = apply_chat_template(tokenizer, messages_for_job(job))
                inputs = tokenizer(
                    [prompt_text],
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_model_len,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                prompt_tokens = int(inputs["input_ids"].shape[-1])
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=length,
                        do_sample=args.temperature > 0,
                        temperature=max(args.temperature, 1e-5),
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = output[0, prompt_tokens:]
                generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
                row = {
                    "length": length,
                    "index": index,
                    "task": job.get("task"),
                    "id": job.get("id"),
                    "turn_idx": job.get("turn_idx"),
                    "total_turns": job.get("total_turns"),
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": int(new_tokens.numel()),
                    "generation": generation,
                    "collapse": collapse_summary(generation),
                }
                length_rows.append(row)
                all_rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "length": length,
                            "index": index,
                            "task": row["task"],
                            "id": row["id"],
                            "prompt_tokens": prompt_tokens,
                            "generated_tokens": row["generated_tokens"],
                            "collapse": row["collapse"],
                        }
                    ),
                    flush=True,
                )
        summary = summarize_rows(length_rows)
        all_summaries[str(length)] = summary
        (args.out_dir / f"summary_{length}.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.stop_on_first_length_failure and summary["valid_observation"] < len(length_rows):
            print(json.dumps({"event": "length_failure_stop", "length": length, "summary": summary}), flush=True)
            break

    summary_payload = {
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "weight_patches": [str(path) for path in args.weight_patch or []],
        "patch_exclude_regex": args.patch_exclude_regex,
        "patch_reports": patch_reports,
        "lengths": parse_int_csv(args.lengths),
        "max_rows": args.max_rows,
        "max_model_len": args.max_model_len,
        "summaries": all_summaries,
        "overall": summarize_rows(all_rows),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "overall": summary_payload["overall"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
