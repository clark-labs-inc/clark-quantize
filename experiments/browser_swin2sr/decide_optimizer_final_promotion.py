#!/usr/bin/env python3
"""Authorize one optimizer-selected Swin2SR artifact, or reject it forever."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_promotion import evaluate
from final_artifact_contract import file_sha256, validate_artifacts
from prepare_full_onnx_smoke import SOURCE_FILES as FIXTURE_SOURCE_FILES


SOURCE_FILES = tuple(dict.fromkeys((
    "experiments/browser_swin2sr/decide_optimizer_final_promotion.py",
    "experiments/browser_swin2sr/evaluate_promotion.py",
    "experiments/browser_swin2sr/final_artifact_contract.py",
    *FIXTURE_SOURCE_FILES,
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--final-evaluation-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runtime-fixture", type=Path, required=True)
    parser.add_argument("--runtime-receipts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def validate_final_chain(
    candidate_path: Path,
    receipt_path: Path,
    policy_path: Path,
    reference_path: Path,
) -> tuple[dict, Path, dict]:
    candidate = json.loads(candidate_path.read_text())
    if candidate.get("format") != "backdropkit-swin2sr-final-candidate-v1":
        raise ValueError("unsupported optimizer final candidate")
    if not candidate.get("final_evaluation_eligible"):
        raise ValueError("candidate was not eligible for final evaluation")
    if candidate.get("model_promotion_authorized"):
        raise ValueError("candidate manifest improperly authorizes promotion")
    validate_artifacts(candidate)

    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != "backdropkit-swin2sr-final-evaluation-v1":
        raise ValueError("unsupported final evaluation receipt")
    precommit_path = receipt_path.parent / "final-precommit.json"
    if (
        not precommit_path.exists()
        or receipt.get("precommit_sha256") != file_sha256(precommit_path)
    ):
        raise ValueError("final evaluation precommit identity changed")
    precommit = json.loads(precommit_path.read_text())
    if precommit.get("format") != "backdropkit-swin2sr-final-precommit-v2":
        raise ValueError("final evaluation did not use optimizer precommit v2")
    expected_precommit = {
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "optimizer_decision": candidate["optimizer_decision"],
        "optimizer_decision_sha256": candidate[
            "optimizer_decision_sha256"
        ],
        "selected_optimizer": candidate["lane"],
        "materialization": candidate["materialization"],
        "materialization_sha256": candidate["materialization_sha256"],
        "hard_state": candidate["hard_state"],
        "hard_state_sha256": candidate["hard_state_sha256"],
        "runtime_state": candidate["runtime_state"],
        "runtime_state_sha256": candidate["runtime_state_sha256"],
        "runtime_package": candidate["runtime_package"],
        "runtime_package_sha256": candidate["runtime_package_sha256"],
        "runtime_package_bytes": candidate["runtime_package_bytes"],
        "artifact_precommit": candidate["artifact_precommit"],
        "artifact_precommit_sha256": candidate[
            "artifact_precommit_sha256"
        ],
        "promotion_policy": str(policy_path),
        "promotion_policy_sha256": file_sha256(policy_path),
        "final_q4": str(reference_path),
        "final_q4_sha256": file_sha256(reference_path),
    }
    if any(
        precommit.get(key) != value
        for key, value in expected_precommit.items()
    ):
        raise ValueError("final evaluation and selected candidate differ")

    report_path = Path(receipt.get("candidate_metrics", ""))
    if (
        not report_path.exists()
        or receipt.get("candidate_metrics_sha256")
        != file_sha256(report_path)
    ):
        raise ValueError("final candidate metrics identity changed")
    report = json.loads(report_path.read_text())
    report_identity = {
        "selection_source": "development",
        "selection_lane": candidate["lane"],
        "state_sha256": candidate["hard_state_sha256"],
        "hard_state_sha256": candidate["hard_state_sha256"],
        "materialization_sha256": candidate["materialization_sha256"],
        "runtime_state_sha256": candidate["runtime_state_sha256"],
        "runtime_artifact_sha256": candidate["runtime_package_sha256"],
        "runtime_artifact_bytes": candidate["runtime_package_bytes"],
        "runtime_artifact_compression": candidate[
            "runtime_package_compression"
        ],
        "runtime_expanded_artifact_sha256": candidate[
            "portable_onnx_sha256"
        ],
        "runtime_expanded_artifact_bytes": candidate[
            "portable_onnx_bytes"
        ],
    }
    if any(report.get(key) != value for key, value in report_identity.items()):
        raise ValueError("final metrics and canonical artifact differ")
    if receipt.get("samples") != len(report.get("samples", [])):
        raise ValueError("final receipt sample count differs")
    if receipt.get("overall") != report.get("overall"):
        raise ValueError("final receipt summary differs")
    return candidate, report_path, report


def validate_runtime_fixture(
    candidate: dict,
    fixture_path: Path,
    runtime_path: Path,
    policy: dict,
) -> None:
    fixture = json.loads(fixture_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    expected_fixture = {
        "hard_state_sha256": candidate["hard_state_sha256"],
        "model": candidate["portable_onnx"],
        "model_sha256": candidate["portable_onnx_sha256"],
        "model_package": candidate["runtime_package"],
        "model_package_compression": candidate[
            "runtime_package_compression"
        ],
        "model_package_bytes": candidate["runtime_package_bytes"],
        "model_package_sha256": candidate["runtime_package_sha256"],
        "input_shape": policy["runtime_input_shape"],
    }
    if any(fixture.get(key) != value for key, value in expected_fixture.items()):
        raise ValueError("runtime fixture and canonical candidate differ")
    repository = Path(__file__).resolve().parents[2]
    expected_sources = {
        name: file_sha256(repository / name)
        for name in FIXTURE_SOURCE_FILES
    }
    if fixture.get("source_files") != expected_sources:
        raise ValueError("runtime fixture source closure differs")
    fixture_sha256 = file_sha256(fixture_path)
    if (
        runtime.get("fixture_sha256") != fixture_sha256
        or runtime.get("fixture_hard_state_sha256")
        != candidate["hard_state_sha256"]
        or runtime.get("model_sha256")
        != candidate["runtime_package_sha256"]
    ):
        raise ValueError("runtime receipt bundle and fixture differ")
    for receipt in runtime.get("receipts", []):
        if (
            receipt.get("fixtureSha256") != fixture_sha256
            or receipt.get("fixtureHardStateSha256")
            != candidate["hard_state_sha256"]
        ):
            raise ValueError("runtime surface and fixture identity differ")


def promotion_record(
    *,
    candidate_path: Path,
    receipt_path: Path,
    policy_path: Path,
    reference_path: Path,
    fixture_path: Path,
    runtime_path: Path,
    result: dict,
    source_files: dict[str, str],
) -> dict:
    passed = bool(result.get("passed"))
    return {
        "format": "backdropkit-swin2sr-final-promotion-decision-v1",
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
        "quality_passed": bool(
            all(
                gate["passed"]
                for gate in result.get("paired_quality", {}).values()
            )
            and all(
                gate["passed"]
                for gate in result.get("stratified_quality", [])
            )
        ),
        "artifact_passed": bool(
            (result.get("artifact") or {}).get("passed")
        ),
        "runtime_passed": bool(
            (result.get("runtime") or {}).get("passed")
        ),
        "promotion_result": result,
        "final_evaluation_complete": True,
        "model_promotion_authorized": passed,
        "required_next_gate": (
            "Integrate this exact artifact into BackdropKit without changing "
            "its bytes, then run product end-to-end regression tests."
            if passed
            else "Reject this candidate. Do not tune against the final "
            "heldout result; begin a new development study and reserve a new "
            "final set before any later promotion attempt."
        ),
    }


def write_or_verify(path: Path, record: dict) -> dict:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != record:
            raise ValueError("existing final promotion decision differs")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    args = parse_args()
    candidate, _, report = validate_final_chain(
        args.candidate_manifest,
        args.final_evaluation_receipt,
        args.policy,
        args.reference,
    )
    policy = json.loads(args.policy.read_text())
    reference = json.loads(args.reference.read_text())
    validate_runtime_fixture(
        candidate,
        args.runtime_fixture,
        args.runtime_receipts,
        policy,
    )
    result = evaluate(policy, reference, report, args.runtime_receipts)
    repository = Path(__file__).resolve().parents[2]
    source_files = {
        name: file_sha256(repository / name)
        for name in SOURCE_FILES
    }
    record = promotion_record(
        candidate_path=args.candidate_manifest,
        receipt_path=args.final_evaluation_receipt,
        policy_path=args.policy,
        reference_path=args.reference,
        fixture_path=args.runtime_fixture,
        runtime_path=args.runtime_receipts,
        result=result,
        source_files=source_files,
    )
    write_or_verify(args.out, record)
    print(json.dumps({
        "candidate": str(args.candidate_manifest),
        "quality_passed": record["quality_passed"],
        "artifact_passed": record["artifact_passed"],
        "runtime_passed": record["runtime_passed"],
        "model_promotion_authorized": record[
            "model_promotion_authorized"
        ],
        "required_next_gate": record["required_next_gate"],
    }, indent=2))
    if not record["model_promotion_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
