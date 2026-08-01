#!/usr/bin/env python3
"""Evaluate one precommitted q1 scale artifact on the locked final bundle."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

from full_surface_provenance import file_sha256
from pack_browser_v3_bundle import unpack_raw_bundle
from prepare_scale_final_candidate import (
    DECISION_FORMATS,
    FORMAT as CANDIDATE_FORMAT,
)


FORMAT = "backdropkit-swin2sr-scale-final-evaluation-v1"
PRECOMMIT_FORMAT = "backdropkit-swin2sr-scale-final-precommit-v1"
SOURCE_FILES = (
    "experiments/browser_swin2sr/evaluate_scale_final_candidate.py",
    "experiments/browser_swin2sr/evaluate_stratified.py",
    "experiments/browser_swin2sr/full_surface_provenance.py",
    "experiments/browser_swin2sr/pack_browser_v3_bundle.py",
    "experiments/browser_swin2sr/prepare_scale_final_candidate.py",
)


def selected_candidate(path: Path) -> dict:
    candidate = json.loads(path.read_text())
    if candidate.get("format") != CANDIDATE_FORMAT:
        raise ValueError("unsupported scale final candidate")
    if candidate.get("selection_source") != "development":
        raise ValueError("scale candidate was not development-selected")
    if not candidate.get("final_evaluation_eligible"):
        raise ValueError("scale candidate failed its development gates")
    if candidate.get("model_promotion_authorized") or not candidate.get(
        "final_heldout_locked"
    ):
        raise ValueError("scale candidate violates final-evaluation isolation")
    return candidate


def verify_record(record: dict, label: str) -> Path:
    path = Path(record["path"])
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise ValueError(f"{label} identity changed")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{label} size changed")
    return path


def validate_candidate_artifacts(candidate: dict) -> None:
    verify_record(candidate["checkpoint"], "scale checkpoint")
    verify_record(candidate["materialization"], "scale materialization")
    verify_record(candidate["hard_state"], "scale hard state")
    verify_record(candidate["runtime_state"], "scale runtime state")
    manifest = verify_record(candidate["artifact_manifest"], "browser-v3 manifest")
    binary = verify_record(candidate["artifact_binary"], "browser-v3 binary")
    package = candidate["runtime_package"]
    package_path = Path(package["runtime_package"])
    if (
        package.get("runtime_package_compression") != "gzip"
        or not package_path.is_file()
        or file_sha256(package_path) != package["runtime_package_sha256"]
        or package_path.stat().st_size != int(package["runtime_package_bytes"])
    ):
        raise ValueError("scale runtime package identity changed")
    raw = gzip.decompress(package_path.read_bytes())
    manifest_bytes, binary_bytes, graph_bytes, layout_bytes = unpack_raw_bundle(raw)
    if manifest_bytes != manifest.read_bytes() or binary_bytes != binary.read_bytes():
        raise ValueError("scale runtime package components differ")
    if package.get("external_graph"):
        if (
            graph_bytes != Path(package["external_graph"]).read_bytes()
            or layout_bytes != Path(package["external_layout"]).read_bytes()
        ):
            raise ValueError("scale WebGPU package components differ")
    if (
        file_sha256(Path(package["artifact_manifest"]))
        != package["artifact_manifest_sha256"]
        or file_sha256(Path(package["artifact_binary"]))
        != package["artifact_binary_sha256"]
    ):
        raise ValueError("scale package ledger component identity changed")


def validate_development_selection(
    candidate: dict,
    policy_path: Path,
    final_bundle_path: Path,
    final_bundle: dict,
    final_q4_path: Path,
    final_q4: dict,
) -> None:
    decision_path = Path(candidate["scale_decision"])
    study_path = Path(candidate["scale_study_manifest"])
    development_bundle = Path(candidate["development_bundle"])
    development_q4 = Path(candidate["development_q4"])
    for path, expected, label in (
        (decision_path, candidate["scale_decision_sha256"], "scale decision"),
        (study_path, candidate["scale_study_manifest_sha256"], "scale study"),
        (development_bundle, candidate["development_bundle_sha256"], "development bundle"),
        (development_q4, candidate["development_q4_sha256"], "development q4"),
        (policy_path, candidate["promotion_policy_sha256"], "promotion policy"),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{label} identity changed")
    if Path(candidate["promotion_policy"]).resolve() != policy_path.resolve():
        raise ValueError("scale candidate names another promotion policy")
    decision = json.loads(decision_path.read_text())
    study = json.loads(study_path.read_text())
    if (
        decision.get("format") not in DECISION_FORMATS
        or decision.get("endpoint_candidate") != candidate["lane"]
        or not decision.get("approximately_1_58_endpoint_eligible")
    ):
        raise ValueError("scale candidate differs from its development decision")
    configured = (decision.get("candidates") or {}).get(candidate["lane"], {})
    if not configured.get("endpoint_eligible"):
        raise ValueError("scale decision endpoint was not eligible")
    if (
        study.get("bundle") != candidate["development_bundle"]
        or study.get("bundle_sha256") != candidate["development_bundle_sha256"]
        or study.get("q4_reference") != candidate["development_q4"]
        or study.get("q4_reference_sha256") != candidate["development_q4_sha256"]
    ):
        raise ValueError("scale candidate and study development inputs differ")
    if (
        development_bundle.resolve() == final_bundle_path.resolve()
        or development_q4.resolve() == final_q4_path.resolve()
    ):
        raise ValueError("scale development selection reused final evidence")
    policy = json.loads(policy_path.read_text())
    reference = policy["reference"]
    checks = {
        "model": final_q4.get("model"),
        "revision": final_q4.get("model_revision"),
        "model_sha256": final_q4.get("state_sha256"),
        "artifact_bytes": final_q4.get("artifact_bytes"),
        "bundle_manifest_sha256": final_bundle.get("manifest_sha256"),
        "samples": int(final_bundle.get("samples", -1)),
    }
    for key, actual in checks.items():
        if reference.get(key) != actual:
            raise ValueError(f"final policy {key} differs")
    if final_q4.get("bundle_manifest_sha256") != final_bundle.get("manifest_sha256"):
        raise ValueError("final q4 and bundle differ")
    repository = Path(__file__).resolve().parents[2]
    for relative, expected in candidate["source_files"].items():
        if file_sha256(repository / relative) != expected:
            raise ValueError(f"scale candidate source changed: {relative}")


def evaluate_command(candidate: dict, bundle: Path, out: Path, device: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("evaluate_stratified.py")),
        candidate["hard_state"]["path"],
        "--bundle", str(bundle),
        "--out", str(out / "heldout"),
        "--device", device,
    ]


def candidate_report(metrics: dict, candidate: dict, bundle: dict, q4: dict) -> dict:
    if metrics.get("state_sha256") != candidate["hard_state"]["sha256"]:
        raise ValueError("final metrics use another scale hard state")
    if metrics.get("bundle_manifest_sha256") != bundle["manifest_sha256"]:
        raise ValueError("final metrics use another bundle")
    if metrics.get("device") != q4.get("device"):
        raise ValueError("scale candidate and q4 used different devices")
    package = candidate["runtime_package"]
    report = dict(metrics)
    report.update({
        "runtime_artifact_bytes": int(package["runtime_package_bytes"]),
        "runtime_artifact_sha256": package["runtime_package_sha256"],
        "runtime_artifact_compression": package["runtime_package_compression"],
        "runtime_expanded_artifact_bytes": int(package["expanded_bundle_bytes"]),
        "runtime_expanded_artifact_sha256": package["expanded_bundle_sha256"],
        "selection_source": "development",
        "selection_lane": candidate["lane"],
        "hard_state_sha256": candidate["hard_state"]["sha256"],
        "materialization_sha256": candidate["materialization"]["sha256"],
        "runtime_state_sha256": candidate["runtime_state"]["sha256"],
    })
    return report


def write_or_verify(path: Path, payload: dict, volatile: str | None = None) -> dict:
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = {key: value for key, value in existing.items() if key != volatile}
        if comparable != payload:
            raise ValueError(f"existing final scale receipt differs: {path}")
        return existing
    value = dict(payload)
    if volatile:
        value[volatile] = time.time()
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def verify_completed_receipt(
    receipt_path: Path,
    precommit_path: Path,
    report_path: Path,
    source_files: dict,
) -> dict:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != FORMAT:
        raise ValueError("unsupported completed scale final evaluation")
    if receipt.get("precommit_sha256") != file_sha256(precommit_path):
        raise ValueError("completed scale final precommit identity changed")
    if Path(receipt.get("candidate_metrics", "")).resolve() != report_path.resolve():
        raise ValueError("completed scale final metrics path changed")
    if not report_path.is_file() or receipt.get(
        "candidate_metrics_sha256"
    ) != file_sha256(report_path):
        raise ValueError("completed scale final metrics identity changed")
    report = json.loads(report_path.read_text())
    if (
        receipt.get("samples") != len(report.get("samples", []))
        or receipt.get("overall") != report.get("overall")
        or receipt.get("source_files") != source_files
    ):
        raise ValueError("completed scale final receipt differs")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, required=True)
    parser.add_argument("--final-bundle", type=Path, required=True)
    parser.add_argument("--final-q4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidate = selected_candidate(args.candidate_manifest)
    validate_candidate_artifacts(candidate)
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
    precommit = {
        "format": PRECOMMIT_FORMAT,
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": file_sha256(args.candidate_manifest),
        "scale_decision": candidate["scale_decision"],
        "scale_decision_sha256": candidate["scale_decision_sha256"],
        "lane": candidate["lane"],
        "hard_state_sha256": candidate["hard_state"]["sha256"],
        "runtime_package_sha256": candidate["runtime_package"]["runtime_package_sha256"],
        "promotion_policy": str(args.promotion_policy),
        "promotion_policy_sha256": file_sha256(args.promotion_policy),
        "final_bundle": str(args.final_bundle),
        "final_bundle_sha256": file_sha256(args.final_bundle),
        "final_q4": str(args.final_q4),
        "final_q4_sha256": file_sha256(args.final_q4),
        "source_files": {
            relative: file_sha256(repository / relative)
            for relative in SOURCE_FILES
        },
    }
    precommit_path = args.out / "final-precommit.json"
    write_or_verify(precommit_path, precommit, "created_at_unix")
    receipt_path = args.out / "final-evaluation-receipt.json"
    report_path = args.out / "candidate-metrics.json"
    if receipt_path.exists():
        receipt = verify_completed_receipt(
            receipt_path,
            precommit_path,
            report_path,
            precommit["source_files"],
        )
        print(json.dumps(receipt, indent=2))
        return
    command = evaluate_command(candidate, args.final_bundle, args.out, final_q4["device"])
    (args.out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (args.out / "evaluation.log").open("ab") as output:
        subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
    metrics = json.loads((args.out / "heldout" / "metrics.json").read_text())
    report = candidate_report(metrics, candidate, final_bundle, final_q4)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    receipt = {
        "format": FORMAT,
        "precommit": str(precommit_path),
        "precommit_sha256": file_sha256(precommit_path),
        "candidate_metrics": str(report_path),
        "candidate_metrics_sha256": file_sha256(report_path),
        "samples": len(report["samples"]),
        "overall": report["overall"],
        "source_files": precommit["source_files"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
