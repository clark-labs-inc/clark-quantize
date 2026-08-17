"""Freeze exact GSM8K and subject-balanced MMLU samples before evaluation.

This command downloads datasets but never loads or calls a language model.
Dataset revisions are mandatory so an alias moving upstream cannot silently
change the scientific gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_b_protocol import (
    make_stratified_support_manifest,
    make_task_manifest,
    prompt_contract_sha256,
    seal_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        choices=["both", "gsm8k", "mmlu"],
        default="both",
        help=(
            "tasks sealed into the manifest; task selection is frozen here "
            "rather than left to the evaluator"
        ),
    )
    parser.add_argument("--gsm8k-revision")
    parser.add_argument("--mmlu-revision")
    parser.add_argument("--gsm8k-limit", type=int, default=1_000)
    parser.add_argument("--mmlu-limit", type=int, default=1_000)
    parser.add_argument("--gsm8k-shots", type=int, default=4, choices=range(5))
    parser.add_argument("--mmlu-shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    include_gsm8k = args.tasks in {"both", "gsm8k"}
    include_mmlu = args.tasks in {"both", "mmlu"}
    if include_gsm8k and not args.gsm8k_revision:
        parser.error("--gsm8k-revision is required for the selected tasks")
    if include_mmlu and not args.mmlu_revision:
        parser.error("--mmlu-revision is required for the selected tasks")

    from datasets import load_dataset

    task_specs = []
    if include_gsm8k:
        gsm = load_dataset(
            "openai/gsm8k",
            "main",
            split="test",
            revision=args.gsm8k_revision,
        )
        task_specs.append(
            make_task_manifest(
                gsm,
                task="gsm8k",
                dataset="openai/gsm8k",
                config="main",
                split="test",
                revision=args.gsm8k_revision,
                fingerprint=gsm._fingerprint,
                limit=args.gsm8k_limit,
                seed=args.seed,
                shots=args.gsm8k_shots,
            )
        )
    if include_mmlu:
        mmlu_test = load_dataset(
            "cais/mmlu",
            "all",
            split="test",
            revision=args.mmlu_revision,
        )
        mmlu_dev = load_dataset(
            "cais/mmlu",
            "all",
            split="dev",
            revision=args.mmlu_revision,
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
            shots=args.mmlu_shots,
            stratum_key="subject",
        )
        mmlu_spec["support"] = make_stratified_support_manifest(
            mmlu_dev,
            dataset="cais/mmlu",
            config="all",
            split="dev",
            revision=args.mmlu_revision,
            fingerprint=mmlu_dev._fingerprint,
            shots_per_stratum=args.mmlu_shots,
            stratum_key="subject",
        )
        task_specs.append(mmlu_spec)
    manifest = seal_manifest(
        task_specs,
        seed=args.seed,
        prompt_contract_hash=prompt_contract_sha256(),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({manifest['content_sha256']})")


if __name__ == "__main__":
    main()
