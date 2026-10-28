#!/usr/bin/env python3
"""Freeze exact MATH-500, GSM8K, and MMLU rows for the Qwen 27B pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "large_scale_ternary"))

from gate_b_protocol import (  # noqa: E402
    digest_json,
    make_stratified_support_manifest,
    make_task_manifest,
    seal_manifest,
)


PROMPT_CONTRACT = {
    "version": "qwen38-capability-prompts-v1",
    "chat_template": {"add_generation_prompt": True, "enable_thinking": False},
    "generation": {"max_new_tokens": 96, "batch_size": 4, "do_sample": False},
    "math500": {"shots": 0, "answer_instruction": "Return only the final answer enclosed in \\boxed{}; do not show work.", "scorer": "boxed_normalized_exact_v1"},
    "gsm8k": {"shots": 4, "answer_instruction": "Return only the final numeric answer.", "scorer": "last_numeric_exact_v1"},
    "mmlu": {"shots": 5, "scorer": "next_token_abcd_v1"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math-revision", required=True)
    parser.add_argument("--gsm8k-revision", required=True)
    parser.add_argument("--mmlu-revision", required=True)
    parser.add_argument("--math-limit", type=int, default=32)
    parser.add_argument("--gsm8k-limit", type=int, default=32)
    parser.add_argument("--mmlu-limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from datasets import load_dataset

    math = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=args.math_revision)
    gsm = load_dataset("openai/gsm8k", "main", split="test", revision=args.gsm8k_revision)
    mmlu_test = load_dataset("cais/mmlu", "all", split="test", revision=args.mmlu_revision)
    mmlu_dev = load_dataset("cais/mmlu", "all", split="dev", revision=args.mmlu_revision)

    math_spec = make_task_manifest(
        math,
        task="math500",
        dataset="HuggingFaceH4/MATH-500",
        config=None,
        split="test",
        revision=args.math_revision,
        fingerprint=math._fingerprint,
        limit=args.math_limit,
        seed=args.seed,
        shots=0,
        stratum_key="subject",
    )
    gsm_spec = make_task_manifest(
        gsm,
        task="gsm8k",
        dataset="openai/gsm8k",
        config="main",
        split="test",
        revision=args.gsm8k_revision,
        fingerprint=gsm._fingerprint,
        limit=args.gsm8k_limit,
        seed=args.seed,
        shots=4,
    )
    mmlu_spec = make_task_manifest(
        mmlu_test,
        task="mmlu",
        dataset="cais/mmlu",
        config="all",
        split="test",
        revision=args.mmlu_revision,
        fingerprint=mmlu_test._fingerprint,
        limit=args.mmlu_limit,
        seed=args.seed,
        shots=5,
        stratum_key="subject",
    )
    mmlu_spec["support"] = make_stratified_support_manifest(
        mmlu_dev,
        dataset="cais/mmlu",
        config="all",
        split="dev",
        revision=args.mmlu_revision,
        fingerprint=mmlu_dev._fingerprint,
        shots_per_stratum=5,
        stratum_key="subject",
    )
    manifest = seal_manifest(
        [math_spec, gsm_spec, mmlu_spec],
        seed=args.seed,
        prompt_contract_hash=digest_json(PROMPT_CONTRACT),
    )
    manifest["prompt_contract"] = PROMPT_CONTRACT
    # seal_manifest cannot include the expanded contract without changing its
    # hash, so reseal the complete immutable body here.
    manifest["content_sha256"] = digest_json(
        {key: value for key, value in manifest.items() if key not in {"created_at", "content_sha256"}}
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"out": str(args.out), "content_sha256": manifest["content_sha256"], "samples": {spec["task"]: len(spec["samples"]) for spec in manifest["tasks"]}}, indent=2))


if __name__ == "__main__":
    main()
