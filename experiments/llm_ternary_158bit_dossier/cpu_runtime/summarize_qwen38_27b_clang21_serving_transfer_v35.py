#!/usr/bin/env python3
"""Reduce paired server and capability receipts for Qwen3.8-27B v35."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


RUN_ORDER = ("parent_a", "candidate_a", "candidate_b", "parent_b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def config(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return next(
        row for row in summary["configurations"] if row["configuration"] == label
    )


def token_rows(root: Path, label: str) -> list[list[int]]:
    return [
        load(root / label / f"prompt-{index:02d}.json")["response"]["tokens"]
        for index in range(4)
    ]


def mad_fraction(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values) / center


def frequency(path: Path) -> dict[str, Any]:
    values = [
        float(line.split()[1])
        for line in path.read_text().splitlines()
        if len(line.split()) == 2
    ]
    if not values:
        raise ValueError("frequency log is empty")
    return {
        "samples": len(values),
        "minimum_mhz": min(values),
        "maximum_mhz": max(values),
        "range_mhz": max(values) - min(values),
        "median_mhz": statistics.median(values),
        "sha256": sha256(path),
    }


def task_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["task"]): row for row in rows}


def summarize_performance(
    prereg: dict[str, Any], state: Path, run_dir: Path
) -> tuple[dict[str, Any], bool]:
    expected_model = prereg["scope"]["target_model_sha256"]
    expected_draft = prereg["scope"]["draft_model_sha256"]
    expected_servers = {
        "parent": prereg["scope"]["parent_server_sha256"],
        "candidate": prereg["scope"]["candidate_server_sha256"],
    }
    runs: dict[str, Any] = {}
    identities_valid = True
    for label in RUN_ORDER:
        arm = label.split("_")[0]
        root = run_dir / label
        summary_path = root / "summary.json"
        summary = load(summary_path)
        target = config(summary, "target")
        mtp3 = config(summary, "mtp3")
        target_tokens = token_rows(root, "target")
        mtp3_tokens = token_rows(root, "mtp3")
        identities_valid = identities_valid and (
            summary.get("terminal") is True
            and summary.get("target", {}).get("sha256") == expected_model
            and summary.get("draft", {}).get("sha256") == expected_draft
            and summary.get("runtime", {}).get("sha256") == expected_servers[arm]
            and summary.get("generation", {}).get("prompt_count") == 4
            and summary.get("generation", {}).get("tokens_per_prompt") == 96
            and target.get("output_tokens_including_prompt_eval_token") == 384
            and mtp3.get("output_tokens_including_prompt_eval_token") == 384
        )
        runs[label] = {
            "arm": arm,
            "target_tps": float(target["accepted_tokens_per_second_weighted"]),
            "mtp3_tps": float(mtp3["accepted_tokens_per_second_weighted"]),
            "mtp3_speedup_vs_own_target": float(
                mtp3["accepted_decode_speedup_vs_target"]
            ),
            "mtp3_draft_n": int(mtp3["draft_n"]),
            "mtp3_draft_n_accepted": int(mtp3["draft_n_accepted"]),
            "mtp3_exact_token_parity_with_own_target": target_tokens == mtp3_tokens,
            "target_token_sha256": [
                hashlib.sha256(
                    json.dumps(row, separators=(",", ":")).encode()
                ).hexdigest()
                for row in target_tokens
            ],
            "summary_sha256": sha256(summary_path),
        }

    arms: dict[str, Any] = {}
    for arm in ("parent", "candidate"):
        selected = [runs[label] for label in RUN_ORDER if runs[label]["arm"] == arm]
        target_values = [row["target_tps"] for row in selected]
        mtp3_values = [row["mtp3_tps"] for row in selected]
        arms[arm] = {
            "target_run_tps": target_values,
            "target_median_tps": statistics.median(target_values),
            "target_run_median_mad_fraction": mad_fraction(target_values),
            "mtp3_run_tps": mtp3_values,
            "mtp3_median_tps": statistics.median(mtp3_values),
            "mtp3_run_median_mad_fraction": mad_fraction(mtp3_values),
        }

    target_ratio = arms["candidate"]["target_median_tps"] / arms["parent"][
        "target_median_tps"
    ]
    mtp3_ratio = arms["candidate"]["mtp3_median_tps"] / arms["parent"][
        "mtp3_median_tps"
    ]
    freq = frequency(state / "frequency.log")
    gate = prereg["gates"]["serving_transfer"]
    checks = {
        "all_receipt_identities_and_denominators_exact": identities_valid,
        "candidate_over_parent_conventional_at_least_1_04": target_ratio
        >= float(gate["minimum_candidate_over_parent_conventional_tps"]),
        "candidate_over_parent_mtp3_at_least_1_02": mtp3_ratio
        >= float(gate["minimum_candidate_over_parent_mtp3_tps"]),
        "candidate_conventional_at_least_2_35_tps": arms["candidate"][
            "target_median_tps"
        ]
        >= float(gate["minimum_candidate_conventional_tps"]),
        "candidate_mtp3_at_least_3_5_tps": arms["candidate"]["mtp3_median_tps"]
        >= float(gate["minimum_candidate_mtp3_tps"]),
        "all_arm_run_median_mad_fractions_at_most_0_05": all(
            arms[arm][key] <= float(gate["maximum_run_median_mad_fraction"])
            for arm in arms
            for key in (
                "target_run_median_mad_fraction",
                "mtp3_run_median_mad_fraction",
            )
        ),
        "candidate_target_mtp3_exact_token_parity_every_prompt": all(
            runs[label]["mtp3_exact_token_parity_with_own_target"]
            for label in ("candidate_a", "candidate_b")
        ),
        "frequency_range_at_most_100_mhz": freq["range_mhz"]
        <= float(gate["maximum_frequency_range_mhz"]),
    }
    cross_runtime_parity = all(
        runs[label]["target_token_sha256"] == runs["parent_a"]["target_token_sha256"]
        for label in RUN_ORDER
    )
    payload = {
        "denominator": {
            "runs": 4,
            "runs_per_arm": 2,
            "configurations_per_run": 2,
            "prompts_per_configuration": 4,
            "generated_tokens_per_prompt": 96,
        },
        "runs": runs,
        "arms": arms,
        "ratios": {
            "candidate_over_parent_conventional": target_ratio,
            "candidate_over_parent_mtp3": mtp3_ratio,
        },
        "cross_runtime_target_token_parity_diagnostic": cross_runtime_parity,
        "frequency": freq,
        "checks": checks,
        "passed": all(checks.values()),
        "program_target_diagnostics": {
            "candidate_conventional_at_least_2_75_tps": arms["candidate"][
                "target_median_tps"
            ]
            >= float(prereg["program_targets"]["conventional_decode_tps"]),
            "candidate_over_parent_at_least_1_40x": target_ratio
            >= float(prereg["program_targets"]["minimum_speedup_vs_q4"]),
            "candidate_at_least_100_tps": arms["candidate"]["target_median_tps"]
            >= float(prereg["program_targets"]["stretch_tps"]),
        },
    }
    return payload, payload["passed"]


def summarize_quality(
    prereg: dict[str, Any], baseline: dict[str, Any], state: Path
) -> tuple[dict[str, Any], bool]:
    cute = load(state / "cute.json")
    math_code = load(state / "math-code.json")
    wiki = load(state / "wikitext.json")
    cute_rows = task_rows(cute["summaries"])
    math_rows = task_rows(math_code["summaries"])
    observed = {
        "cute_correct": int(cute_rows["cute_quick_20"]["correct"]),
        "cute_n": int(cute_rows["cute_quick_20"]["n"]),
        "math500_correct": int(math_rows["math500"]["correct"]),
        "math500_n": int(math_rows["math500"]["n"]),
        "code_trace_correct": int(math_rows["code_trace"]["correct"]),
        "code_trace_n": int(math_rows["code_trace"]["n"]),
        "wikitext_ppl": float(wiki["final_ppl"]),
    }
    observed["wikitext_ppl_ratio_vs_stock"] = observed["wikitext_ppl"] / float(
        baseline["wikitext_ppl"]
    )
    expected_model = prereg["scope"]["target_model_sha256"]
    gate = prereg["gates"]["always_run_quick_quality"]
    receipt_checks = {
        "cute_receipt_valid_20": cute.get("schema") == "cpu_cute_quick_result_v1"
        and cute.get("model_sha256") == expected_model
        and len(cute.get("records", [])) == 20
        and observed["cute_n"] == 20,
        "math_code_receipt_valid_8_plus_6": math_code.get("schema")
        == "cpu_math_code_canary_result_v1"
        and math_code.get("model_sha256") == expected_model
        and len(math_code.get("records", [])) == 14
        and observed["math500_n"] == 8
        and observed["code_trace_n"] == 6,
        "wikitext_receipt_valid_4_chunks": wiki.get("schema")
        == "cpu_wikitext_quick_result_v1"
        and wiki.get("valid") is True
        and wiki.get("model_sha256") == expected_model
        and len(wiki.get("chunks", [])) == 4,
    }
    quality_checks = {
        "cute_at_least_9_of_20": observed["cute_correct"]
        >= int(gate["cute_quick_minimum_correct"]),
        "math500_at_least_5_of_8": observed["math500_correct"]
        >= int(gate["math500_minimum_correct"]),
        "code_trace_at_least_5_of_6": observed["code_trace_correct"]
        >= int(gate["code_trace_minimum_correct"]),
        "wikitext_ratio_at_most_1_01": observed["wikitext_ppl_ratio_vs_stock"]
        <= float(gate["maximum_wikitext_ppl_ratio_vs_stock"]),
    }
    passed = all(receipt_checks.values()) and all(quality_checks.values())
    return {
        "denominator": {
            "cute": 20,
            "math500": 8,
            "code_trace": 6,
            "wikitext_chunks": 4,
        },
        "observed": observed,
        "receipt_checks": receipt_checks,
        "quality_checks": quality_checks,
        "passed": passed,
        "provenance": {
            "cute_sha256": sha256(state / "cute.json"),
            "math_code_sha256": sha256(state / "math-code.json"),
            "wikitext_sha256": sha256(state / "wikitext.json"),
        },
    }, passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mmlu", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prereg = load(args.preregistration)
    baseline = load(args.baseline)
    performance, performance_passed = summarize_performance(
        prereg, args.state, args.run_dir
    )
    quality, quality_passed = summarize_quality(prereg, baseline, args.state)
    prior_passed = performance_passed and quality_passed

    mmlu_result = None
    mmlu_checks = None
    if args.mmlu:
        mmlu = load(args.mmlu)
        gate = prereg["gates"]["conditional_mmlu"]
        correct = int(mmlu["summary"]["correct"])
        n = int(mmlu["summary"]["n"])
        loss = int(gate["stock_correct"]) - correct
        mmlu_result = {
            "correct": correct,
            "n": n,
            "accuracy": float(mmlu["summary"]["accuracy"]),
            "loss_vs_stock_correct": loss,
            "sha256": sha256(args.mmlu),
        }
        mmlu_checks = {
            "receipt_valid_64": mmlu.get("valid") is True
            and mmlu.get("status") == "completed"
            and mmlu.get("contract", {}).get("model_sha256")
            == prereg["scope"]["target_model_sha256"]
            and n == int(gate["denominator"]),
            "correct_at_least_49": correct >= int(gate["minimum_correct"]),
            "loss_vs_stock_at_most_3": loss
            <= int(gate["maximum_loss_vs_stock_correct"]),
        }

    if not prior_passed:
        terminal = True
        passed = False
        decision = "reject_clang_serving_composition_before_mmlu"
        exit_code = 2
    elif args.mmlu is None:
        terminal = False
        passed = False
        decision = "advance_to_mmlu_64"
        exit_code = 0
    else:
        terminal = True
        passed = all(mmlu_checks.values())
        decision = (
            "retain_clang_active_as_representation_kernel_component"
            if passed
            else "reject_clang_serving_composition_on_mmlu"
        )
        exit_code = 0 if passed else 2

    infrastructure_valid = (
        performance["checks"]["all_receipt_identities_and_denominators_exact"]
        and all(quality["receipt_checks"].values())
        and (mmlu_checks is None or mmlu_checks["receipt_valid_64"])
    )
    result = {
        "schema": "qwen38_27b_clang21_serving_transfer_result_v35",
        "experiment_id": prereg["experiment_id"],
        "terminal": terminal,
        "infrastructure_valid": infrastructure_valid,
        "passed": passed,
        "decision": decision,
        "performance": performance,
        "quick_quality": quality,
        "mmlu": mmlu_result,
        "mmlu_checks": mmlu_checks,
        "provenance": {
            "preregistration_sha256": sha256(args.preregistration),
            "baseline_sha256": sha256(args.baseline),
            "sealed_inputs_sha256": sha256(args.state / "sealed-inputs.sha256"),
        },
        "next_lane": (
            "Combine the retained compiler composition with a validation-gated, activation-optimal residual-assisted three-bit representation."
            if passed
            else "Keep the representation/kernel lane independent of this rejected compiler composition."
        ),
        "claim_boundary": prereg["claim_boundary"],
    }
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "event": "terminal" if terminal else "mmlu_admitted",
                "decision": decision,
                "performance_passed": performance_passed,
                "quality_passed": quality_passed,
                "mmlu_passed": None
                if mmlu_checks is None
                else all(mmlu_checks.values()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
