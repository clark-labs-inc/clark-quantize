#!/usr/bin/env python3
"""Authorize or permanently reject one final-evaluated packed q1 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_promotion import evaluate
from evaluate_scale_final_candidate import (
    FORMAT as EVALUATION_FORMAT,
    PRECOMMIT_FORMAT,
    selected_candidate,
    validate_candidate_artifacts,
)
from full_surface_provenance import file_sha256
from prepare_scale_runtime_fixture import FORMAT as FIXTURE_FORMAT, SOURCE_FILES as FIXTURE_SOURCES


FORMAT = "backdropkit-swin2sr-scale-final-promotion-decision-v1"
SOURCE_FILES = tuple(dict.fromkeys((
    "experiments/browser_swin2sr/decide_scale_final_promotion.py",
    "experiments/browser_swin2sr/evaluate_promotion.py",
    "experiments/browser_swin2sr/evaluate_scale_final_candidate.py",
    "experiments/browser_swin2sr/full_surface_provenance.py",
    *FIXTURE_SOURCES,
)))


def validate_final_chain(
    candidate_path: Path,
    receipt_path: Path,
    policy_path: Path,
    reference_path: Path,
) -> tuple[dict, dict]:
    candidate = selected_candidate(candidate_path)
    validate_candidate_artifacts(candidate)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != EVALUATION_FORMAT:
        raise ValueError("unsupported packed-q1 final evaluation")
    precommit_path = Path(receipt["precommit"])
    if (
        not precommit_path.is_file()
        or receipt.get("precommit_sha256") != file_sha256(precommit_path)
    ):
        raise ValueError("packed-q1 final precommit identity changed")
    precommit = json.loads(precommit_path.read_text())
    package = candidate["runtime_package"]
    expected_precommit = {
        "format": PRECOMMIT_FORMAT,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "scale_decision": candidate["scale_decision"],
        "scale_decision_sha256": candidate["scale_decision_sha256"],
        "lane": candidate["lane"],
        "hard_state_sha256": candidate["hard_state"]["sha256"],
        "runtime_package_sha256": package["runtime_package_sha256"],
        "promotion_policy": str(policy_path),
        "promotion_policy_sha256": file_sha256(policy_path),
        "final_q4": str(reference_path),
        "final_q4_sha256": file_sha256(reference_path),
    }
    if any(precommit.get(key) != value for key, value in expected_precommit.items()):
        raise ValueError("packed-q1 final evaluation and candidate differ")
    report_path = Path(receipt["candidate_metrics"])
    if (
        not report_path.is_file()
        or receipt.get("candidate_metrics_sha256") != file_sha256(report_path)
    ):
        raise ValueError("packed-q1 final metrics identity changed")
    report = json.loads(report_path.read_text())
    expected_report = {
        "selection_source": "development",
        "selection_lane": candidate["lane"],
        "hard_state_sha256": candidate["hard_state"]["sha256"],
        "materialization_sha256": candidate["materialization"]["sha256"],
        "runtime_state_sha256": candidate["runtime_state"]["sha256"],
        "runtime_artifact_sha256": package["runtime_package_sha256"],
        "runtime_artifact_bytes": package["runtime_package_bytes"],
        "runtime_artifact_compression": package["runtime_package_compression"],
        "runtime_expanded_artifact_sha256": package["expanded_bundle_sha256"],
        "runtime_expanded_artifact_bytes": package["expanded_bundle_bytes"],
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise ValueError("packed-q1 final report and candidate differ")
    if (
        receipt.get("samples") != len(report.get("samples", []))
        or receipt.get("overall") != report.get("overall")
    ):
        raise ValueError("packed-q1 final receipt summary differs")
    return candidate, report


def validate_runtime_fixture(
    candidate_path: Path,
    candidate: dict,
    fixture_path: Path,
    runtime_path: Path,
    policy: dict,
) -> None:
    fixture = json.loads(fixture_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    package = candidate["runtime_package"]
    expected = {
        "format": FIXTURE_FORMAT,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "hard_state_sha256": candidate["hard_state"]["sha256"],
        "model": package["runtime_package"],
        "model_sha256": package["runtime_package_sha256"],
        "model_bytes": package["runtime_package_bytes"],
        "model_compression": package["runtime_package_compression"],
        "expanded_model_sha256": package["expanded_bundle_sha256"],
        "expanded_model_bytes": package["expanded_bundle_bytes"],
        "artifact_manifest_sha256": candidate["artifact_manifest"]["sha256"],
        "artifact_binary_sha256": candidate["artifact_binary"]["sha256"],
        "input_shape": policy["runtime_input_shape"],
    }
    if any(fixture.get(key) != value for key, value in expected.items()):
        raise ValueError("packed-q1 runtime fixture and candidate differ")
    repository = Path(__file__).resolve().parents[2]
    expected_sources = {
        relative: file_sha256(repository / relative)
        for relative in FIXTURE_SOURCES
    }
    if fixture.get("source_files") != expected_sources:
        raise ValueError("packed-q1 runtime fixture source closure differs")
    fixture_sha = file_sha256(fixture_path)
    if (
        runtime.get("model_sha256") != package["runtime_package_sha256"]
        or runtime.get("fixture_sha256") != fixture_sha
        or runtime.get("fixture_hard_state_sha256")
        != candidate["hard_state"]["sha256"]
    ):
        raise ValueError("packed-q1 runtime receipts use another fixture")


def promotion_record(
    candidate_path: Path,
    receipt_path: Path,
    policy_path: Path,
    reference_path: Path,
    fixture_path: Path,
    runtime_path: Path,
    result: dict,
    source_files: dict,
) -> dict:
    quality = bool(
        all(gate["passed"] for gate in result.get("paired_quality", {}).values())
        and all(gate["passed"] for gate in result.get("stratified_quality", []))
    )
    artifact = bool((result.get("artifact") or {}).get("passed"))
    runtime = bool((result.get("runtime") or {}).get("passed"))
    passed = bool(result.get("passed") and quality and artifact and runtime)
    return {
        "format": FORMAT,
        "selection_source": "development_then_one_final_heldout",
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "final_evaluation_receipt": str(receipt_path),
        "final_evaluation_receipt_sha256": file_sha256(receipt_path),
        "promotion_policy": str(policy_path),
        "promotion_policy_sha256": file_sha256(policy_path),
        "q4_reference": str(reference_path),
        "q4_reference_sha256": file_sha256(reference_path),
        "runtime_fixture": str(fixture_path),
        "runtime_fixture_sha256": file_sha256(fixture_path),
        "runtime_receipts": str(runtime_path),
        "runtime_receipts_sha256": file_sha256(runtime_path),
        "source_files": source_files,
        "quality_passed": quality,
        "artifact_passed": artifact,
        "runtime_passed": runtime,
        "promotion_result": result,
        "final_evaluation_complete": True,
        "model_promotion_authorized": passed,
        "required_next_gate": (
            "Integrate this exact packed q1 artifact into BackdropKit and run product regression."
            if passed
            else "Reject this candidate without tuning against final heldout evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--final-evaluation-receipt", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runtime-fixture", type=Path, required=True)
    parser.add_argument("--runtime-receipts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidate, report = validate_final_chain(
        args.candidate_manifest,
        args.final_evaluation_receipt,
        args.policy,
        args.reference,
    )
    policy = json.loads(args.policy.read_text())
    reference = json.loads(args.reference.read_text())
    validate_runtime_fixture(
        args.candidate_manifest,
        candidate,
        args.runtime_fixture,
        args.runtime_receipts,
        policy,
    )
    result = evaluate(policy, reference, report, args.runtime_receipts)
    repository = Path(__file__).resolve().parents[2]
    source_files = {
        relative: file_sha256(repository / relative)
        for relative in SOURCE_FILES
    }
    record = promotion_record(
        args.candidate_manifest,
        args.final_evaluation_receipt,
        args.policy,
        args.reference,
        args.runtime_fixture,
        args.runtime_receipts,
        result,
        source_files,
    )
    if args.out.exists() and json.loads(args.out.read_text()) != record:
        raise ValueError("existing packed-q1 final promotion decision differs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "quality_passed": record["quality_passed"],
        "artifact_passed": record["artifact_passed"],
        "runtime_passed": record["runtime_passed"],
        "model_promotion_authorized": record["model_promotion_authorized"],
    }, indent=2))
    if not record["model_promotion_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
