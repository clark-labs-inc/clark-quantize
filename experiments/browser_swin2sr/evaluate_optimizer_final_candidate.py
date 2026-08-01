#!/usr/bin/env python3
"""Evaluate the one optimizer-selected canonical artifact on final heldout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate_final_candidate import (
    candidate_report,
    verify_completed_receipt,
    write_or_verify_precommit,
)
from final_artifact_contract import file_sha256, validate_artifacts


FORMAT = "backdropkit-swin2sr-final-candidate-v1"
SOURCE_FILES = (
    "clark_air/quant/binary.py",
    "clark_air/quant/optimal_ternary.py",
    "clark_quantize/bonsai_muon.py",
    "clark_quantize/bonsai_qat8.py",
    "clark_quantize/qat_optimizer_variants.py",
    "clark_quantize/ternary_kd.py",
    "experiments/browser_swin2sr/artifact_state.py",
    "experiments/browser_swin2sr/data_pipeline.py",
    "experiments/browser_swin2sr/evaluate_final_candidate.py",
    "experiments/browser_swin2sr/evaluate_optimizer_final_candidate.py",
    "experiments/browser_swin2sr/evaluate_stratified.py",
    "experiments/browser_swin2sr/final_artifact_contract.py",
    "experiments/browser_swin2sr/optimizer_controls.py",
    "experiments/browser_swin2sr/qat_objective.py",
    "experiments/browser_swin2sr/quantizer_controls.py",
    "experiments/browser_swin2sr/rotation_control.py",
    "experiments/browser_swin2sr/run_state.py",
    "experiments/browser_swin2sr/train_swin2sr_qat.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, required=True)
    parser.add_argument("--final-bundle", type=Path, required=True)
    parser.add_argument("--final-q4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def selected_candidate(manifest: dict) -> dict:
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported final candidate manifest")
    if manifest.get("selection_source") != "development":
        raise ValueError("final candidate was not development-selected")
    if not manifest.get("final_evaluation_eligible"):
        raise ValueError("final candidate failed its development artifact gate")
    if manifest.get("model_promotion_authorized"):
        raise ValueError("candidate manifest improperly authorizes promotion")
    if not manifest.get("final_heldout_locked"):
        raise ValueError("candidate manifest does not lock final heldout")
    if not (manifest.get("development_quality_gate") or {}).get("passed"):
        raise ValueError("final candidate failed development quality")
    if not (manifest.get("artifact_size_gate") or {}).get("passed"):
        raise ValueError("final candidate failed artifact size")
    required = (
        "lane",
        "optimizer_decision",
        "optimizer_decision_sha256",
        "optimizer_study_manifest",
        "optimizer_study_manifest_sha256",
        "development_bundle",
        "development_bundle_sha256",
        "development_q4",
        "development_q4_sha256",
        "promotion_policy",
        "promotion_policy_sha256",
        "artifact_precommit",
        "artifact_precommit_sha256",
        "base_onnx",
        "base_onnx_sha256",
        "materialization",
        "materialization_sha256",
        "hard_state",
        "hard_state_sha256",
        "runtime_state",
        "runtime_state_sha256",
        "portable_onnx",
        "portable_onnx_bytes",
        "portable_onnx_sha256",
        "portable_ledger",
        "portable_ledger_sha256",
        "runtime_package",
        "runtime_package_bytes",
        "runtime_package_sha256",
        "runtime_package_compression",
    )
    missing = [key for key in required if manifest.get(key) is None]
    if missing:
        raise ValueError(f"final candidate is missing fields: {missing}")
    return manifest


def validate_development_selection(
    candidate: dict,
    policy_path: Path,
    final_bundle_path: Path,
    final_bundle: dict,
    final_q4_path: Path,
    final_q4: dict,
) -> None:
    decision_path = Path(candidate["optimizer_decision"])
    study_path = Path(candidate["optimizer_study_manifest"])
    development_bundle_path = Path(candidate["development_bundle"])
    development_q4_path = Path(candidate["development_q4"])
    artifact_precommit_path = Path(candidate["artifact_precommit"])
    for current, expected, label in (
        (
            decision_path,
            candidate["optimizer_decision_sha256"],
            "optimizer decision",
        ),
        (
            study_path,
            candidate["optimizer_study_manifest_sha256"],
            "optimizer study",
        ),
        (
            development_bundle_path,
            candidate["development_bundle_sha256"],
            "development bundle",
        ),
        (
            development_q4_path,
            candidate["development_q4_sha256"],
            "development q4",
        ),
        (
            policy_path,
            candidate["promotion_policy_sha256"],
            "promotion policy",
        ),
        (
            artifact_precommit_path,
            candidate["artifact_precommit_sha256"],
            "artifact precommit",
        ),
    ):
        if not current.exists() or file_sha256(current) != expected:
            raise ValueError(f"{label} identity changed")
    if Path(candidate["promotion_policy"]).resolve() != policy_path.resolve():
        raise ValueError("candidate names another promotion policy")

    decision = json.loads(decision_path.read_text())
    study = json.loads(study_path.read_text())
    development_bundle = json.loads(development_bundle_path.read_text())
    development_q4 = json.loads(development_q4_path.read_text())
    policy = json.loads(policy_path.read_text())
    artifact_precommit = json.loads(artifact_precommit_path.read_text())
    if (
        decision.get("format")
        != "backdropkit-swin2sr-optimizer-transfer-decision-v1"
        or not decision.get("optimizer_recipe_selected")
    ):
        raise ValueError("candidate does not descend from optimizer selection")
    selected = {
        "lane": decision["selected_optimizer"],
        "materialization": decision["selected_materialization"],
        "materialization_sha256": decision[
            "selected_materialization_sha256"
        ],
        "hard_state": decision["selected_dense_state"],
        "hard_state_sha256": decision["selected_dense_state_sha256"],
        "runtime_state": decision["selected_runtime_state"],
        "runtime_state_sha256": decision["selected_runtime_state_sha256"],
    }
    if any(candidate.get(key) != value for key, value in selected.items()):
        raise ValueError("candidate differs from selected optimizer identity")
    if (
        study.get("format")
        != "backdropkit-swin2sr-optimizer-transfer-v1"
        or study.get("selection_source") != "development"
    ):
        raise ValueError("candidate optimizer study is invalid")
    if (
        study.get("bundle") != candidate["development_bundle"]
        or study.get("bundle_sha256")
        != candidate["development_bundle_sha256"]
        or study.get("q4_reference") != candidate["development_q4"]
        or study.get("q4_reference_sha256")
        != candidate["development_q4_sha256"]
    ):
        raise ValueError("candidate and optimizer study inputs differ")
    if (
        development_bundle["manifest_sha256"]
        != development_q4["bundle_manifest_sha256"]
    ):
        raise ValueError("development bundle and q4 differ")
    if (
        development_bundle_path.resolve() == final_bundle_path.resolve()
        or development_bundle["manifest_sha256"]
        == final_bundle["manifest_sha256"]
        or development_q4_path.resolve() == final_q4_path.resolve()
    ):
        raise ValueError("development selection reused final evidence")
    reference = policy["reference"]
    final_checks = {
        "model": final_q4.get("model"),
        "revision": final_q4.get("model_revision"),
        "model_sha256": final_q4.get("state_sha256"),
        "artifact_bytes": final_q4.get("artifact_bytes"),
        "bundle_manifest_sha256": final_bundle.get("manifest_sha256"),
        "samples": int(final_bundle.get("samples", -1)),
    }
    for key, actual in final_checks.items():
        if reference.get(key) != actual:
            raise ValueError(f"final policy {key} differs")
    if final_q4.get("bundle_manifest_sha256") != final_bundle.get(
        "manifest_sha256"
    ):
        raise ValueError("final q4 and bundle differ")
    expected_precommit = {
        "format": "backdropkit-swin2sr-artifact-precommit-v1",
        "optimizer_decision": candidate["optimizer_decision"],
        "optimizer_decision_sha256": candidate[
            "optimizer_decision_sha256"
        ],
        "promotion_policy": candidate["promotion_policy"],
        "promotion_policy_sha256": candidate["promotion_policy_sha256"],
        "base_onnx": candidate["base_onnx"],
        "base_onnx_sha256": candidate["base_onnx_sha256"],
        "materialization": candidate["materialization"],
        "materialization_sha256": candidate["materialization_sha256"],
        "portable_onnx": candidate["portable_onnx"],
        "final_heldout_locked": True,
    }
    if any(
        artifact_precommit.get(key) != value
        for key, value in expected_precommit.items()
    ):
        raise ValueError("candidate artifact precommit differs")
    repository = Path(__file__).resolve().parents[2]
    for relative, expected in artifact_precommit.get(
        "source_files",
        {},
    ).items():
        source = repository / relative
        if not source.exists() or file_sha256(source) != expected:
            raise ValueError(
                f"candidate artifact source identity changed: {relative}"
            )


def evaluate_command(
    candidate: dict,
    bundle_path: Path,
    destination: Path,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("evaluate_stratified.py")),
        candidate["hard_state"],
        "--bundle",
        str(bundle_path),
        "--out",
        str(destination / "heldout"),
        "--device",
        device,
    ]


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    completed_path = args.out / "final-evaluation-receipt.json"
    candidate = selected_candidate(
        json.loads(args.candidate_manifest.read_text())
    )
    validate_artifacts(candidate)
    final_bundle = json.loads(args.final_bundle.read_text())
    final_q4 = json.loads(args.final_q4.read_text())
    validate_development_selection(
        candidate,
        args.promotion_policy,
        args.final_bundle,
        final_bundle,
        args.final_q4,
        final_q4,
    )
    repository = Path(__file__).resolve().parents[2]
    source_hashes = {
        name: file_sha256(repository / name)
        for name in SOURCE_FILES
    }
    stable_precommit = {
        "format": "backdropkit-swin2sr-final-precommit-v2",
        "selection_source": "development",
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": file_sha256(args.candidate_manifest),
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
        "promotion_policy": str(args.promotion_policy),
        "promotion_policy_sha256": file_sha256(args.promotion_policy),
        "final_bundle": str(args.final_bundle),
        "final_bundle_sha256": file_sha256(args.final_bundle),
        "final_bundle_manifest_sha256": final_bundle["manifest_sha256"],
        "final_q4": str(args.final_q4),
        "final_q4_sha256": file_sha256(args.final_q4),
        "final_q4_state_sha256": final_q4["state_sha256"],
        "final_device": final_q4["device"],
        "source_files": source_hashes,
    }
    precommit_path = args.out / "final-precommit.json"
    write_or_verify_precommit(precommit_path, stable_precommit)
    report_path = args.out / "candidate-metrics.json"
    if completed_path.exists():
        receipt = verify_completed_receipt(
            completed_path,
            precommit_path,
            report_path,
            repository,
            SOURCE_FILES,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    command = evaluate_command(
        candidate,
        args.final_bundle,
        args.out,
        final_q4["device"],
    )
    (args.out / "command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    with (args.out / "evaluation.log").open("ab") as output:
        subprocess.run(
            command,
            check=True,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    metrics = json.loads(
        (args.out / "heldout" / "metrics.json").read_text()
    )
    report = candidate_report(metrics, candidate, final_bundle, final_q4)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    receipt = {
        "format": "backdropkit-swin2sr-final-evaluation-v1",
        "precommit_sha256": file_sha256(precommit_path),
        "candidate_metrics": str(report_path),
        "candidate_metrics_sha256": file_sha256(report_path),
        "selected_lane": candidate["lane"],
        "samples": len(report["samples"]),
        "overall": report["overall"],
        "source_files": source_hashes,
    }
    completed_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
