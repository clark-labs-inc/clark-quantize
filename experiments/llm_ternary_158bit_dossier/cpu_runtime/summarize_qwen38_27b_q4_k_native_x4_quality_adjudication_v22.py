#!/usr/bin/env python3
"""Summarize the frozen quality adjudication for the fast native-x4 Q4_K kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def task_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["task"]): row for row in rows}


def require_exact_baseline(prereg: dict[str, Any], baseline: dict[str, Any]) -> None:
    frozen = prereg["baseline"]
    cute = task_rows(baseline["cute"])["cute_quick_20"]
    math_code = task_rows(baseline["math_code"])
    observed = {
        "model_sha256": baseline["model_sha256"],
        "cute_quick_correct": int(cute["correct"]),
        "cute_quick_n": int(cute["n"]),
        "math500_correct": int(math_code["math500"]["correct"]),
        "math500_n": int(math_code["math500"]["n"]),
        "code_trace_correct": int(math_code["code_trace"]["correct"]),
        "code_trace_n": int(math_code["code_trace"]["n"]),
        "wikitext_ppl": float(baseline["wikitext_ppl"]),
        "mmlu_correct": int(baseline["mmlu"]["correct"]),
        "mmlu_n": int(baseline["mmlu"]["n"]),
    }
    expected = {key: frozen[key] for key in observed if key != "model_sha256"}
    expected["model_sha256"] = prereg["scope"]["model_sha256"]
    if observed != expected:
        raise ValueError(f"baseline differs from preregistration: {observed!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--manifest-amendment", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--predecessor", required=True, type=Path)
    parser.add_argument("--cute", required=True, type=Path)
    parser.add_argument("--math-code", required=True, type=Path)
    parser.add_argument("--wikitext", required=True, type=Path)
    parser.add_argument("--mmlu", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prereg = json.loads(args.preregistration.read_text())
    amendment = json.loads(args.manifest_amendment.read_text())
    baseline = json.loads(args.baseline.read_text())
    predecessor = json.loads(args.predecessor.read_text())
    cute = json.loads(args.cute.read_text())
    math_code = json.loads(args.math_code.read_text())
    wiki = json.loads(args.wikitext.read_text())
    model_sha = prereg["scope"]["model_sha256"]

    if prereg["schema"] != "qwen38_27b_q4_k_native_x4_quality_adjudication_preregistration_v22":
        raise ValueError("unexpected preregistration schema")
    if amendment["experiment_id"] != prereg["experiment_id"]:
        raise ValueError("manifest amendment targets another experiment")
    if predecessor.get("decision") != "reject_system_gate":
        raise ValueError("v22 requires the preserved terminal v21 rejection")
    require_exact_baseline(prereg, baseline)

    cute_rows = task_rows(cute["summaries"])
    math_rows = task_rows(math_code["summaries"])
    candidate = {
        "cute_correct": int(cute_rows["cute_quick_20"]["correct"]),
        "cute_n": int(cute_rows["cute_quick_20"]["n"]),
        "math500_correct": int(math_rows["math500"]["correct"]),
        "math500_n": int(math_rows["math500"]["n"]),
        "code_trace_correct": int(math_rows["code_trace"]["correct"]),
        "code_trace_n": int(math_rows["code_trace"]["n"]),
        "wikitext_ppl": float(wiki["final_ppl"]),
    }
    candidate["wikitext_ppl_ratio"] = candidate["wikitext_ppl"] / float(
        baseline["wikitext_ppl"]
    )

    receipt_checks = {
        "cute_receipt_valid_20": (
            cute.get("schema") == "cpu_cute_quick_result_v1"
            and cute.get("model_sha256") == model_sha
            and len(cute.get("records", [])) == 20
            and candidate["cute_n"] == 20
        ),
        "math_code_receipt_valid_8_plus_6": (
            math_code.get("schema") == "cpu_math_code_canary_result_v1"
            and math_code.get("model_sha256") == model_sha
            and len(math_code.get("records", [])) == 14
            and candidate["math500_n"] == 8
            and candidate["code_trace_n"] == 6
        ),
        "wikitext_receipt_valid_4_chunks": (
            wiki.get("schema") == "cpu_wikitext_quick_result_v1"
            and wiki.get("valid") is True
            and wiki.get("status") == "completed"
            and wiki.get("model_sha256") == model_sha
            and len(wiki.get("chunks", [])) == 4
            and wiki.get("parameters", {}).get("context") == 512
        ),
    }
    gate_1_frozen = prereg["sequential_gates"][
        "gate_1_always_run_character_math_and_perplexity"
    ]
    gate_1_checks = {
        "cute_at_least_9_of_20": candidate["cute_correct"]
        >= int(gate_1_frozen["cute_quick_minimum_correct"]),
        "math500_at_least_5_of_8": candidate["math500_correct"]
        >= int(gate_1_frozen["math500_minimum_correct"]),
        "code_trace_at_least_5_of_6": candidate["code_trace_correct"]
        >= int(gate_1_frozen["code_trace_minimum_correct"]),
        "wikitext_ratio_at_most_1_005": candidate["wikitext_ppl_ratio"]
        <= float(gate_1_frozen["maximum_candidate_over_baseline_wikitext_ppl_ratio"]),
    }
    gate_1_passed = all(receipt_checks.values()) and all(gate_1_checks.values())

    mmlu_payload: dict[str, Any] | None = None
    mmlu_check: bool | None = None
    mmlu_receipt_valid: bool | None = None
    if args.mmlu is not None:
        mmlu = json.loads(args.mmlu.read_text())
        gate_2 = prereg["sequential_gates"]["gate_2_conditional_mmlu"]
        mmlu_payload = {
            "correct": int(mmlu["summary"]["correct"]),
            "n": int(mmlu["summary"]["n"]),
            "accuracy": float(mmlu["summary"]["accuracy"]),
            "loss_vs_baseline_correct": int(baseline["mmlu"]["correct"])
            - int(mmlu["summary"]["correct"]),
        }
        mmlu_receipt_valid = (
            mmlu.get("valid") is True
            and mmlu.get("status") == "completed"
            and mmlu.get("contract", {}).get("model_sha256") == model_sha
            and mmlu_payload["n"] == int(gate_2["denominator"])
        )
        mmlu_check = (
            mmlu_payload["correct"] >= int(gate_2["minimum_correct"])
            and mmlu_payload["loss_vs_baseline_correct"]
            <= int(gate_2["maximum_loss_vs_baseline_correct"])
        )

    if not gate_1_passed:
        terminal = True
        passed = False
        decision = "reject_quality_gate_1"
        exit_code = 2
    elif args.mmlu is None:
        terminal = False
        passed = False
        decision = "advance_to_mmlu_64"
        exit_code = 0
    else:
        terminal = True
        passed = bool(mmlu_receipt_valid and mmlu_check)
        decision = "quality_qualified_same_weights_kernel" if passed else "reject_quality_gate_2"
        exit_code = 0 if passed else 2

    provenance_paths = {
        "preregistration": args.preregistration,
        "manifest_amendment": args.manifest_amendment,
        "baseline": args.baseline,
        "predecessor_system_decision": args.predecessor,
        "cute": args.cute,
        "math_code": args.math_code,
        "wikitext": args.wikitext,
    }
    if args.mmlu is not None:
        provenance_paths["mmlu"] = args.mmlu

    result = {
        "schema": "qwen38_27b_q4_k_native_x4_quality_adjudication_result_v22",
        "experiment_id": prereg["experiment_id"],
        "model": "Qwen/Qwen3.8-27B",
        "model_sha256": model_sha,
        "infrastructure_valid": all(receipt_checks.values())
        and (mmlu_receipt_valid is not False),
        "terminal": terminal,
        "passed": passed,
        "decision": decision,
        "denominators": {
            "cute": 20,
            "math500": 8,
            "code_trace": 6,
            "wikitext_chunks": 4,
            "mmlu": 64 if args.mmlu is not None else None,
        },
        "candidate": candidate,
        "mmlu": mmlu_payload,
        "receipt_checks": receipt_checks,
        "gate_1_checks": gate_1_checks,
        "gate_1_passed": gate_1_passed,
        "gate_2_checks": (
            {
                "mmlu_receipt_valid_64": mmlu_receipt_valid,
                "mmlu_at_least_51_and_loss_at_most_1": mmlu_check,
            }
            if args.mmlu is not None
            else None
        ),
        "retained_speed_signal": {
            "mtp3_candidate_over_control_accepted_tps": prereg["predecessor"][
                "retained_signals"
            ]["mtp3_candidate_over_control_accepted_tps"],
            "candidate_mtp3_accepted_tps": prereg["predecessor"]["retained_signals"]
            ["candidate_mtp3_accepted_tps"],
            "v21_exact_token_gate_remains_failed": True,
        },
        "provenance": {
            name + "_sha256": sha256_file(path)
            for name, path in provenance_paths.items()
        },
        "claim_boundary": (
            "A pass quality-qualifies the same-weights v21 AVX2 kernel for its measured "
            "MTP3 speed signal. It does not reverse v21 token-identity rejection, establish "
            "the 2.75 tok/s conventional-decode breakthrough gate, reach 100 tok/s, or "
            "support novelty or scientific-breakthrough claims."
        ),
    }
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "event": "quality_terminal" if terminal else "quality_gate_1_passed",
                "decision": decision,
                "passed": passed,
                "candidate": candidate,
                "mmlu": mmlu_payload,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
