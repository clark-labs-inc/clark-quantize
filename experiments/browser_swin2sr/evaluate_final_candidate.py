#!/usr/bin/env python3
"""Evaluate one development-selected Swin2SR candidate on the final bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from final_artifact_contract import file_sha256, validate_artifacts


SOURCE_FILES = (
    "evaluate_final_candidate.py",
    "evaluate_stratified.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-summary", type=Path, required=True)
    parser.add_argument("--final-bundle", type=Path, required=True)
    parser.add_argument("--final-q4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def selected_candidate(summary: dict) -> dict:
    if summary.get("format") != "backdropkit-swin2sr-full-surface-summary-v1":
        raise ValueError("unsupported full-surface comparison summary")
    if summary.get("failed") or not summary.get("ranking"):
        raise ValueError("full-surface comparison has no unambiguous winner")
    winner = summary["ranking"][0]
    required = (
        "lane",
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
    missing = [key for key in required if winner.get(key) is None]
    if missing:
        raise ValueError(f"comparison winner is missing fields: {missing}")
    return winner


def validate_development_selection(
    summary: dict,
    final_bundle_path: Path,
    final_bundle: dict,
    final_q4_path: Path,
) -> None:
    required = (
        "study_manifest",
        "study_manifest_sha256",
        "bundle",
        "bundle_sha256",
        "bundle_manifest_sha256",
        "q4_reference",
        "q4_reference_sha256",
        "q4_reference_state_sha256",
    )
    if summary.get("selection_source") != "development":
        raise ValueError("final candidate was not selected on development data")
    missing = [key for key in required if summary.get(key) is None]
    if missing:
        raise ValueError(f"development selection is missing provenance: {missing}")
    study_path = Path(summary["study_manifest"])
    bundle_path = Path(summary["bundle"])
    q4_path = Path(summary["q4_reference"])
    for path, expected, label in (
        (study_path, summary["study_manifest_sha256"], "study manifest"),
        (bundle_path, summary["bundle_sha256"], "development bundle"),
        (q4_path, summary["q4_reference_sha256"], "development q4"),
    ):
        if not path.exists() or file_sha256(path) != expected:
            raise ValueError(f"{label} hash mismatch")
    study = json.loads(study_path.read_text())
    development_bundle = json.loads(bundle_path.read_text())
    development_q4 = json.loads(q4_path.read_text())
    if study.get("format") != "backdropkit-swin2sr-full-surface-comparison-v1":
        raise ValueError("unsupported development study manifest")
    if study.get("selection_source") != "development":
        raise ValueError("study manifest is not development-only")
    expected_study = {
        "bundle": summary["bundle"],
        "bundle_sha256": summary["bundle_sha256"],
        "q4_reference": summary["q4_reference"],
        "q4_reference_sha256": summary["q4_reference_sha256"],
    }
    if any(study.get(key) != value for key, value in expected_study.items()):
        raise ValueError("development summary and study manifest differ")
    if (
        development_bundle.get("manifest_sha256")
        != summary["bundle_manifest_sha256"]
        or development_q4.get("bundle_manifest_sha256")
        != summary["bundle_manifest_sha256"]
    ):
        raise ValueError("development bundle and q4 reference do not match")
    if (
        development_q4.get("state_sha256")
        != summary["q4_reference_state_sha256"]
    ):
        raise ValueError("development q4 state identity mismatch")
    if (
        bundle_path.resolve() == final_bundle_path.resolve()
        or summary["bundle_manifest_sha256"] == final_bundle.get("manifest_sha256")
    ):
        raise ValueError("development selection used the final evaluation bundle")
    if q4_path.resolve() == final_q4_path.resolve():
        raise ValueError("development selection used the final q4 evaluation")


def stable_precommit(
    summary_path: Path,
    summary: dict,
    candidate: dict,
    bundle_path: Path,
    bundle: dict,
    q4_path: Path,
    q4: dict,
) -> dict:
    return {
        "format": "backdropkit-swin2sr-final-precommit-v1",
        "selection_source": "development",
        "comparison_summary": str(summary_path),
        "comparison_summary_sha256": file_sha256(summary_path),
        "development_study_manifest": summary["study_manifest"],
        "development_study_manifest_sha256": summary["study_manifest_sha256"],
        "development_bundle": summary["bundle"],
        "development_bundle_sha256": summary["bundle_sha256"],
        "development_bundle_manifest_sha256": summary[
            "bundle_manifest_sha256"
        ],
        "development_q4": summary["q4_reference"],
        "development_q4_sha256": summary["q4_reference_sha256"],
        "selected_lane": candidate["lane"],
        "materialization": candidate["materialization"],
        "materialization_sha256": candidate["materialization_sha256"],
        "hard_state": candidate["hard_state"],
        "hard_state_sha256": candidate["hard_state_sha256"],
        "runtime_state": candidate["runtime_state"],
        "runtime_state_sha256": candidate["runtime_state_sha256"],
        "portable_onnx": candidate["portable_onnx"],
        "portable_onnx_bytes": candidate["portable_onnx_bytes"],
        "portable_onnx_sha256": candidate["portable_onnx_sha256"],
        "portable_ledger": candidate["portable_ledger"],
        "portable_ledger_sha256": candidate["portable_ledger_sha256"],
        "runtime_package": candidate["runtime_package"],
        "runtime_package_bytes": candidate["runtime_package_bytes"],
        "runtime_package_sha256": candidate["runtime_package_sha256"],
        "runtime_package_compression": candidate["runtime_package_compression"],
        "final_bundle": str(bundle_path),
        "final_bundle_sha256": file_sha256(bundle_path),
        "final_bundle_manifest_sha256": bundle["manifest_sha256"],
        "final_q4": str(q4_path),
        "final_q4_sha256": file_sha256(q4_path),
        "final_q4_state_sha256": q4["state_sha256"],
        "final_device": q4["device"],
    }


def write_or_verify_precommit(path: Path, stable: dict) -> dict:
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = {
            key: value
            for key, value in existing.items()
            if key != "created_at_unix"
        }
        if comparable != stable:
            raise ValueError("final precommit differs from the existing selection")
        return existing
    receipt = {**stable, "created_at_unix": time.time()}
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_completed_receipt(
    path: Path,
    precommit_path: Path,
    report_path: Path,
    source_root: Path,
    source_files: tuple[str, ...] = SOURCE_FILES,
) -> dict:
    receipt = json.loads(path.read_text())
    if receipt.get("format") != "backdropkit-swin2sr-final-evaluation-v1":
        raise ValueError("unsupported completed final-evaluation receipt")
    if not precommit_path.exists():
        raise ValueError("completed final evaluation is missing its precommit")
    if receipt.get("precommit_sha256") != file_sha256(precommit_path):
        raise ValueError("completed final-evaluation precommit hash mismatch")
    recorded_report = Path(receipt.get("candidate_metrics", ""))
    if recorded_report.resolve() != report_path.resolve():
        raise ValueError("completed final-evaluation report path mismatch")
    if not report_path.exists():
        raise ValueError("completed final evaluation is missing candidate metrics")
    if receipt.get("candidate_metrics_sha256") != file_sha256(report_path):
        raise ValueError("completed final-evaluation metrics hash mismatch")
    report = json.loads(report_path.read_text())
    if receipt.get("samples") != len(report.get("samples", [])):
        raise ValueError("completed final-evaluation sample count mismatch")
    if receipt.get("overall") != report.get("overall"):
        raise ValueError("completed final-evaluation metrics summary mismatch")
    expected_sources = {
        name: file_sha256(source_root / name)
        for name in source_files
    }
    if receipt.get("source_files") != expected_sources:
        raise ValueError("completed final-evaluation source hash mismatch")
    return receipt


def evaluate_command(
    candidate: dict,
    bundle_path: Path,
    out: Path,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("evaluate_stratified.py")),
        candidate["hard_state"],
        "--bundle",
        str(bundle_path),
        "--out",
        str(out / "heldout"),
        "--device",
        device,
    ]


def candidate_report(
    metrics: dict,
    candidate: dict,
    bundle: dict,
    q4: dict,
) -> dict:
    if metrics.get("state_sha256") != candidate["hard_state_sha256"]:
        raise ValueError(
            "candidate metrics reference a different canonical hard state"
        )
    if metrics.get("bundle_manifest_sha256") != bundle["manifest_sha256"]:
        raise ValueError("candidate metrics reference a different final bundle")
    if metrics.get("device") != q4.get("device"):
        raise ValueError("candidate and q4 reference used different devices")
    if len(metrics.get("samples", [])) != int(bundle["samples"]):
        raise ValueError("candidate final evaluation sample count mismatch")
    report = dict(metrics)
    report["runtime_artifact_bytes"] = int(candidate["runtime_package_bytes"])
    report["runtime_artifact_sha256"] = candidate["runtime_package_sha256"]
    report["runtime_artifact_compression"] = candidate[
        "runtime_package_compression"
    ]
    report["runtime_expanded_artifact_bytes"] = int(
        candidate["portable_onnx_bytes"]
    )
    report["runtime_expanded_artifact_sha256"] = candidate[
        "portable_onnx_sha256"
    ]
    report["selection_source"] = "development"
    report["selection_lane"] = candidate["lane"]
    report["hard_state_sha256"] = candidate["hard_state_sha256"]
    report["materialization_sha256"] = candidate["materialization_sha256"]
    report["runtime_state_sha256"] = candidate["runtime_state_sha256"]
    report["portable_ledger_sha256"] = candidate["portable_ledger_sha256"]
    return report


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    completed_path = args.out / "final-evaluation-receipt.json"
    summary = json.loads(args.comparison_summary.read_text())
    bundle = json.loads(args.final_bundle.read_text())
    q4 = json.loads(args.final_q4.read_text())
    candidate = selected_candidate(summary)
    validate_artifacts(candidate)
    validate_development_selection(
        summary,
        args.final_bundle,
        bundle,
        args.final_q4,
    )
    if bundle.get("manifest_sha256") != q4.get("bundle_manifest_sha256"):
        raise ValueError("final bundle and q4 reference do not match")
    precommit = stable_precommit(
        args.comparison_summary,
        summary,
        candidate,
        args.final_bundle,
        bundle,
        args.final_q4,
        q4,
    )
    precommit_path = args.out / "final-precommit.json"
    write_or_verify_precommit(precommit_path, precommit)
    report_path = args.out / "candidate-metrics.json"
    source_root = Path(__file__).resolve().parent
    if completed_path.exists():
        receipt = verify_completed_receipt(
            completed_path,
            precommit_path,
            report_path,
            source_root,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    command = evaluate_command(candidate, args.final_bundle, args.out, q4["device"])
    (args.out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (args.out / "evaluation.log").open("ab") as output:
        subprocess.run(
            command,
            check=True,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    metrics_path = args.out / "heldout" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    report = candidate_report(metrics, candidate, bundle, q4)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    receipt = {
        "format": "backdropkit-swin2sr-final-evaluation-v1",
        "precommit_sha256": file_sha256(precommit_path),
        "candidate_metrics": str(report_path),
        "candidate_metrics_sha256": file_sha256(report_path),
        "selected_lane": candidate["lane"],
        "samples": len(report["samples"]),
        "overall": report["overall"],
        "source_files": {
            name: file_sha256(source_root / name)
            for name in SOURCE_FILES
        },
    }
    completed_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
