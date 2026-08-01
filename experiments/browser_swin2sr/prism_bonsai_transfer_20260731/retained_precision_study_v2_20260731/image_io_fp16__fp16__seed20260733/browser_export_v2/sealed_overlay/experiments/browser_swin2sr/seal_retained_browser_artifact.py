#!/usr/bin/env python3
"""Seal browser materialization/export of a confirmed retained-precision arm."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-retained-browser-export-study-v1"
OVERLAY_SOURCES = (
    "experiments/browser_swin2sr/retained_precision_islands.py",
    "experiments/browser_swin2sr/retained_precision_state.py",
    "experiments/browser_swin2sr/train_retained_fp16_island.py",
    "experiments/browser_swin2sr/materialize_retained_precision_checkpoint.py",
    "experiments/browser_swin2sr/materialize_surface_states.py",
    "experiments/browser_swin2sr/artifact_ledger.py",
    "experiments/browser_swin2sr/export_browser_artifact.py",
    "experiments/browser_swin2sr/seal_retained_browser_artifact.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-materialization", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--base-sealed-sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def command(overlay: Path, base: Path, script: str, *arguments: str) -> list[str]:
    overlay_browser = overlay / "experiments/browser_swin2sr"
    base_browser = base / "experiments/browser_swin2sr"
    executable = overlay / script if (overlay / script).exists() else base / script
    return [
        "/usr/bin/env",
        f"PYTHONPATH={overlay_browser}:{base_browser}:{base}",
        sys.executable,
        str(executable),
        *arguments,
    ]


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"retained browser-export study exists: {args.out}")
    source = json.loads(args.source_materialization.read_text())
    if source.get("format") != (
        "backdropkit-swin2sr-retained-precision-materialization-v1"
    ):
        raise ValueError("unsupported retained-precision materialization")
    precision = source["retained_precision_training"]["precision"]
    if precision not in {"ternary", "fp16"}:
        raise RuntimeError("unsupported retained-precision browser arm")
    checkpoint = Path(source["checkpoint"])
    if file_sha256(checkpoint) != source["checkpoint_sha256"]:
        raise RuntimeError("retained-precision checkpoint changed")
    report = json.loads(args.development_report.read_text())
    if (
        report["state_sha256"] != source["dense_state"]["file_sha256"]
        or len(report.get("samples", [])) != 3040
    ):
        raise RuntimeError("development report does not evaluate the source state")
    if not args.base_sealed_sources.is_dir():
        raise FileNotFoundError(args.base_sealed_sources)
    repository = Path(__file__).resolve().parents[2]
    overlay = args.out / "sealed_overlay"
    source_sha256 = {}
    for relative in OVERLAY_SOURCES:
        destination = overlay / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, destination)
        source_sha256[relative] = file_sha256(destination)
    materialized = args.out / "browser_materialized"
    artifact = args.out / "browser_artifact"
    manifest = {
        "format": FORMAT,
        "status": "sealed_before_browser_materialization",
        "selection_source": "confirmed_development_leader",
        "precision": precision,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "source_materialization": str(args.source_materialization),
        "source_materialization_sha256": file_sha256(args.source_materialization),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "development_report": str(args.development_report),
        "development_report_sha256": file_sha256(args.development_report),
        "development_psnr_vs_ground_truth": report["overall"][
            "psnr_vs_ground_truth"
        ]["mean"],
        "base_sealed_sources": str(args.base_sealed_sources),
        "sealed_overlay": str(overlay),
        "source_sha256": source_sha256,
        "materialization_command": command(
            overlay,
            args.base_sealed_sources,
            "experiments/browser_swin2sr/materialize_retained_precision_checkpoint.py",
            str(checkpoint),
            "--out", str(materialized),
            "--source-device", "cpu",
        ),
        "materialization": str(materialized / "materialization.json"),
        "artifact_command": command(
            overlay,
            args.base_sealed_sources,
            "experiments/browser_swin2sr/export_browser_artifact.py",
            str(materialized / "materialization.json"),
            "--out", str(artifact),
        ),
        "artifact": str(artifact),
        "required_next_gates": [
            "materialization_dense_state_exactly_matches_source",
            "packed_runtime_numerical_conformance",
            "cross_browser_webgpu_wasm_cpu_latency_and_memory",
            "robustness",
            "official_q4_quality",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "study-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "study_manifest": str(path),
        "study_manifest_sha256": file_sha256(path),
        "materialization_command": manifest["materialization_command"],
        "artifact_command": manifest["artifact_command"],
    }, indent=2))


if __name__ == "__main__":
    main()
