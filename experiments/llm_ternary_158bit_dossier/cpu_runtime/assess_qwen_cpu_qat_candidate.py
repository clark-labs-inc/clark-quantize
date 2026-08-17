#!/usr/bin/env python3
"""Produce a fail-closed decision receipt for an exact CPU QAT candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hashed_report(path: Path, digest_key: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    claimed = report.get(digest_key)
    body = {key: value for key, value in report.items() if key != digest_key}
    if not claimed or canonical_digest(body) != claimed:
        raise ValueError(f"report content hash mismatch: {path}")
    return report


def task_summary(report: dict[str, Any], task: str) -> dict[str, Any]:
    matches = [row for row in report["summaries"] if row["task"] == task]
    if len(matches) != 1:
        raise ValueError(f"expected one {task} summary, found {len(matches)}")
    return matches[0]


def assess(
    *,
    reconstruction_path: Path,
    scope_path: Path,
    cute_path: Path,
    math_code_path: Path,
    ppl_path: Path,
    min_reconstruction_improvement: float,
    min_cute_correct: int,
    min_math_correct: int,
    min_code_correct: int,
    max_ppl: float,
) -> dict[str, Any]:
    reconstruction = load_hashed_report(
        reconstruction_path, "report_content_sha256"
    )
    scope = load_hashed_report(scope_path, "receipt_content_sha256")
    cute = load_hashed_report(cute_path, "report_content_sha256")
    math_code = load_hashed_report(math_code_path, "report_content_sha256")
    ppl = load_hashed_report(ppl_path, "report_content_sha256")

    candidate_model_sha = scope["models"]["candidate"]["sha256"]
    report_model_shas = {
        "scope": candidate_model_sha,
        "cute": cute["model_sha256"],
        "math_code": math_code["model_sha256"],
        "ppl": ppl["model_sha256"],
    }
    if len(set(report_model_shas.values())) != 1:
        raise ValueError(f"evaluation reports refer to different models: {report_model_shas}")

    cute_summary = task_summary(cute, "cute_quick_20")
    math_summary = task_summary(math_code, "math500")
    code_summary = task_summary(math_code, "code_trace")
    expected_denominators = {
        "cute_quick_20": 20,
        "math500": 8,
        "code_trace": 6,
    }
    observed_denominators = {
        "cute_quick_20": int(cute_summary["n"]),
        "math500": int(math_summary["n"]),
        "code_trace": int(code_summary["n"]),
    }
    if observed_denominators != expected_denominators:
        raise ValueError(
            f"frozen evaluation denominator mismatch: {observed_denominators}"
        )

    test = reconstruction["splits"]["test"]
    reconstruction_improvement = float(
        test["candidate_relative_improvement_vs_direct_ptq_percent"]
    )
    gates = {
        "exact_ternary_patch": bool(reconstruction["candidate_exact_ternary"]),
        "sealed_test_nonempty": int(test["positions"]) > 0,
        "sealed_reconstruction_improvement": reconstruction_improvement
        >= min_reconstruction_improvement,
        "gguf_scope": bool(scope["valid"])
        and int(scope["target_count"]) == 3
        and int(scope["non_target_count"]) == 438,
        "cute_quick": int(cute_summary["correct"]) >= min_cute_correct,
        "math500": int(math_summary["correct"]) >= min_math_correct,
        "code_trace": int(code_summary["correct"]) >= min_code_correct,
        "wikitext": bool(ppl["valid"])
        and ppl["status"] == "completed"
        and float(ppl["final_ppl"]) <= max_ppl,
    }
    passed = all(gates.values())
    result = {
        "schema": "qwen_cpu_qat_candidate_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "pass": passed,
        "decision": (
            "replicated_one_layer_gate_pass_no_depth_authorization"
            if passed
            else "candidate_rejected"
        ),
        "claim_boundary": "one-layer Qwen3.5-4B replication; not a whole-model result and not a breakthrough",
        "depth_expansion_authorized": False,
        "next_authorized_experiment": (
            "distinct frozen capability replication such as a subject-balanced MMLU slice"
            if passed
            else "diagnose the failed gate without adding quantized depth"
        ),
        "candidate_model_sha256": candidate_model_sha,
        "candidate_patch_sha256": reconstruction["candidate_patch_sha256"],
        "thresholds": {
            "minimum_sealed_reconstruction_improvement_percent": min_reconstruction_improvement,
            "minimum_cute_correct": [min_cute_correct, 20],
            "minimum_math500_correct": [min_math_correct, 8],
            "minimum_code_trace_correct": [min_code_correct, 6],
            "maximum_wikitext_ppl": max_ppl,
        },
        "observed": {
            "sealed_test_positions": int(test["positions"]),
            "sealed_reconstruction_improvement_percent": reconstruction_improvement,
            "cute_quick_correct": [int(cute_summary["correct"]), 20],
            "math500_correct": [int(math_summary["correct"]), 8],
            "code_trace_correct": [int(code_summary["correct"]), 6],
            "wikitext_ppl": float(ppl["final_ppl"]),
            "wikitext_uncertainty": float(ppl["final_uncertainty"]),
            "gguf_target_tensors": int(scope["target_count"]),
            "byte_identical_non_target_tensors": int(scope["non_target_count"]),
        },
        "gates": gates,
        "inputs": {
            label: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for label, path in (
                ("reconstruction", reconstruction_path),
                ("scope", scope_path),
                ("cute", cute_path),
                ("math_code", math_code_path),
                ("ppl", ppl_path),
            )
        },
        "evaluator_sha256": sha256_file(Path(__file__)),
    }
    result["report_content_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", required=True, type=Path)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--cute", required=True, type=Path)
    parser.add_argument("--math-code", required=True, type=Path)
    parser.add_argument("--ppl", required=True, type=Path)
    parser.add_argument("--min-reconstruction-improvement", type=float, default=10.0)
    parser.add_argument("--min-cute-correct", type=int, default=7)
    parser.add_argument("--min-math-correct", type=int, default=5)
    parser.add_argument("--min-code-correct", type=int, default=4)
    parser.add_argument("--max-ppl", type=float, default=8.6975)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = assess(
        reconstruction_path=args.reconstruction,
        scope_path=args.scope,
        cute_path=args.cute,
        math_code_path=args.math_code,
        ppl_path=args.ppl,
        min_reconstruction_improvement=args.min_reconstruction_improvement,
        min_cute_correct=args.min_cute_correct,
        min_math_correct=args.min_math_correct,
        min_code_correct=args.min_code_correct,
        max_ppl=args.max_ppl,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
