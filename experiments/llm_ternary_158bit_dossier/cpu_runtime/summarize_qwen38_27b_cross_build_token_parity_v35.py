#!/usr/bin/env python3
"""Diagnose cross-build token parity in the frozen v35 serving rotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ARMS = ("parent_a", "candidate_a", "candidate_b", "parent_b")
MODES = ("target", "mtp3")
PROMPTS = tuple(range(4))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def summarize(run_dir: Path) -> dict[str, Any]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    sources = []
    for arm in ARMS:
        for mode in MODES:
            for prompt in PROMPTS:
                path = run_dir / arm / mode / f"prompt-{prompt:02d}.json"
                record = json.loads(path.read_text())
                tokens = record.get("response", {}).get("tokens")
                if (
                    record.get("terminal") is not True
                    or record.get("prompt_index") != prompt
                    or record.get("configuration") != mode
                    or record.get("n_predict") != 96
                    or not isinstance(tokens, list)
                    or len(tokens) != 96
                    or not all(isinstance(token, int) for token in tokens)
                ):
                    raise ValueError(f"invalid v35 raw response: {path}")
                records[(arm, mode, prompt)] = record
                sources.append(
                    {
                        "arm": arm,
                        "mode": mode,
                        "prompt": prompt,
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                )

    prompts = []
    for prompt in PROMPTS:
        prompt_records = {
            f"{arm}/{mode}": records[(arm, mode, prompt)]
            for arm in ARMS
            for mode in MODES
        }
        prompt_hashes = {
            record["prompt_sha256"] for record in prompt_records.values()
        }
        if len(prompt_hashes) != 1:
            raise ValueError("v35 prompt identity differs across arms")
        tokens = {
            label: record["response"]["tokens"]
            for label, record in prompt_records.items()
        }
        parent = tokens["parent_a/target"]
        candidate = tokens["candidate_a/target"]
        within_build = {
            arm: tokens[f"{arm}/target"] == tokens[f"{arm}/mtp3"]
            for arm in ARMS
        }
        prompts.append(
            {
                "prompt": prompt,
                "prompt_sha256": prompt_hashes.pop(),
                "tokens_per_response": len(parent),
                "within_build_target_mtp3_exact_parity": within_build,
                "parent_a_parent_b_exact_parity": (
                    tokens["parent_a/target"] == tokens["parent_b/target"]
                ),
                "candidate_a_candidate_b_exact_parity": (
                    tokens["candidate_a/target"] == tokens["candidate_b/target"]
                ),
                "candidate_parent_exact_parity": candidate == parent,
                "candidate_parent_first_mismatch_token_index": first_mismatch(
                    parent, candidate
                ),
            }
        )

    cross_build_exact = sum(
        int(item["candidate_parent_exact_parity"]) for item in prompts
    )
    within_build_exact = sum(
        int(passed)
        for item in prompts
        for passed in item["within_build_target_mtp3_exact_parity"].values()
    )
    parent_repeat_exact = all(
        item["parent_a_parent_b_exact_parity"] for item in prompts
    )
    candidate_repeat_exact = all(
        item["candidate_a_candidate_b_exact_parity"] for item in prompts
    )
    return {
        "schema": "qwen38_27b_cross_build_token_parity_diagnostic_v35",
        "posthoc_diagnostic": True,
        "model": "Qwen/Qwen3.8-27B",
        "prompt_count": len(PROMPTS),
        "tokens_per_prompt": 96,
        "prompts": prompts,
        "aggregate": {
            "within_build_target_mtp3_exact_pairs": within_build_exact,
            "within_build_target_mtp3_pair_denominator": len(PROMPTS) * len(ARMS),
            "cross_build_candidate_parent_exact_prompts": cross_build_exact,
            "cross_build_candidate_parent_prompt_denominator": len(PROMPTS),
            "parent_ab_repeat_exact_every_prompt": parent_repeat_exact,
            "candidate_ab_repeat_exact_every_prompt": candidate_repeat_exact,
        },
        "classification": (
            "candidate_mtp_acceptance_gain_is_sequence_confounded_and_cannot_be_"
            "attributed_as_a_clean_kernel_runtime_speedup"
            if cross_build_exact != len(PROMPTS)
            else "cross_build_sequences_match"
        ),
        "interpretation": (
            "Target-vs-MTP equality inside each build proves speculative parity "
            "for that build. It does not prove the candidate compiler/runtime "
            "matches the parent. A changed greedy sequence can change draft "
            "acceptance and therefore confound cross-build MTP throughput."
        ),
        "gate_effect": "diagnostic_only; does not alter the frozen v35 gates",
        "source_files": sources,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "complete": True,
        "terminal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
