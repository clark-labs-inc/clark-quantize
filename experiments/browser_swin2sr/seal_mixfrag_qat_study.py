#!/usr/bin/env python3
"""Seal progressive QAT for the selected MixFrag per-surface scale map."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from full_surface_provenance import file_sha256
from seal_scale_codebook_qat_study import SOURCE_FILES as SCALE_QAT_SOURCE_FILES
from training_data_contract import dataset_fingerprints


FORMAT = "backdropkit-swin2sr-mixfrag-scale-qat-study-v1"
TARGET_MAP_FORMAT = "backdropkit-swin2sr-per-surface-scale-map-v1"
SOURCE_FILES = tuple(dict.fromkeys((
    *SCALE_QAT_SOURCE_FILES,
    "experiments/browser_swin2sr/seal_mixfrag_qat_study.py",
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction-result", type=Path, required=True)
    parser.add_argument("--development-bundle", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--master-learning-rate", type=float, default=1e-6)
    parser.add_argument("--center-learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def sealed_command(sealed_root: Path, script: str, *arguments: str) -> list[str]:
    browser = sealed_root / "experiments/browser_swin2sr"
    return [
        "/usr/bin/env",
        f"PYTHONPATH={browser}:{sealed_root}",
        sys.executable,
        str(sealed_root / script),
        *arguments,
    ]


def verify_source_closure(repository: Path) -> dict[str, list[str]]:
    source_paths = {
        relative: (repository / relative).resolve() for relative in SOURCE_FILES
    }
    included = set(source_paths.values())
    search_roots = (repository, repository / "experiments/browser_swin2sr")
    resolved = {}
    missing = {}
    for relative, source in source_paths.items():
        tree = ast.parse(source.read_text(), filename=str(source))
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                parts = module.split(".")
                candidates = []
                for root in search_roots:
                    candidates.extend((
                        root.joinpath(*parts).with_suffix(".py"),
                        root.joinpath(*parts, "__init__.py"),
                    ))
                dependency = next(
                    (path.resolve() for path in candidates if path.exists()), None
                )
                if dependency is None:
                    continue
                dependency_relative = str(dependency.relative_to(repository))
                dependencies.add(dependency_relative)
                if dependency not in included:
                    missing.setdefault(relative, []).append(dependency_relative)
        resolved[relative] = sorted(dependencies)
    if missing:
        details = "; ".join(
            f"{source}: {sorted(set(dependencies))}"
            for source, dependencies in sorted(missing.items())
        )
        raise RuntimeError(f"sealed source dependency closure is incomplete: {details}")
    return resolved


def load_selection(path: Path, repository: Path) -> tuple[dict, dict, dict, dict]:
    result = json.loads(path.read_text())
    if result.get("format") != (
        "backdropkit-swin2sr-scale-allocation-interaction-result-v1"
    ):
        raise ValueError("unsupported interaction result")
    if (
        result.get("model_promotion_authorized")
        or result.get("development_accessed")
        or result.get("final_heldout_accessed")
        or not result.get("final_heldout_locked")
    ):
        raise RuntimeError("interaction result violates training-only isolation")
    if result["decision"]["selected_candidate"] != "mixfrag_10012":
        raise RuntimeError("the preregistered 10,012-byte candidate was not selected")
    candidate = result["candidates"]["mixfrag_10012"]
    storage = candidate["storage"]
    if (
        int(candidate["budget_bytes"]) != 10012
        or int(storage["calibration_storage_bytes"]) != 10012
        or int(storage["physical_aligned_span_bytes"]) != 10012
        or int(storage["surface_count"]) != 204
    ):
        raise RuntimeError("selected candidate violates the sealed storage contract")
    protocol_path = Path(result["protocol"])
    if not protocol_path.is_absolute():
        protocol_path = repository / protocol_path
    if file_sha256(protocol_path) != result["protocol_sha256"]:
        raise RuntimeError("interaction protocol changed")
    protocol = json.loads(protocol_path.read_text())
    report_path = Path(protocol["allocation_report"])
    if not report_path.is_absolute():
        report_path = repository / report_path
    if file_sha256(report_path) != protocol["allocation_report_sha256"]:
        raise RuntimeError("allocation report changed")
    report = json.loads(report_path.read_text())
    allocation = report["allocations"]["10012"]
    if allocation["choice_counts"] != storage["choice_counts"]:
        raise RuntimeError("allocation labels differ from the interaction receipt")
    parent = Path(protocol["parent_checkpoint"])
    if not parent.is_absolute():
        parent = repository / parent
    if file_sha256(parent) != protocol["parent_checkpoint_sha256"]:
        raise RuntimeError("interaction parent changed")
    if protocol["parent_checkpoint_sha256"] != result["parent_checkpoint_sha256"]:
        raise RuntimeError("interaction result names a different parent")
    protocol = {
        **protocol,
        "allocation_report": str(report_path.resolve()),
        "parent_checkpoint": str(parent.resolve()),
    }
    return result, protocol, allocation, storage


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"MixFrag QAT study exists: {args.out}")
    if args.steps != 4000 or args.device != "mps":
        raise ValueError("initial MixFrag QAT is sealed at 4,000 MPS steps")
    if args.master_learning_rate != 1e-6 or args.center_learning_rate != 1e-5:
        raise ValueError("initial MixFrag QAT learning rates are frozen")
    repository = Path(__file__).resolve().parents[2]
    result, protocol, allocation, storage = load_selection(
        args.interaction_result, repository
    )
    if not args.development_bundle.exists():
        raise FileNotFoundError(args.development_bundle)
    choices = allocation["choices"]
    target_bits = {}
    for name, label in choices.items():
        if not isinstance(label, str) or not label.startswith("q"):
            raise RuntimeError(f"selected allocation is not integer codebook: {name}")
        bits = int(label[1:])
        if bits < 1 or bits > 8:
            raise RuntimeError(f"selected allocation has invalid bits: {name}")
        target_bits[name] = bits
    counts = {f"q{bits}": count for bits, count in sorted(Counter(target_bits.values()).items())}
    if len(target_bits) != 204 or counts != storage["choice_counts"]:
        raise RuntimeError("target map does not reproduce the storage receipt")
    source_closure = verify_source_closure(repository)
    sealed_root = args.out / "sealed_sources"
    for relative in SOURCE_FILES:
        destination = sealed_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, destination)
    args.out.mkdir(parents=True, exist_ok=True)
    target_map_path = args.out / "target-map.json"
    target_map = {
        "format": TARGET_MAP_FORMAT,
        "selection_source": "training_only",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "interaction_result": str(args.interaction_result),
        "interaction_result_sha256": file_sha256(args.interaction_result),
        "allocation_report": protocol["allocation_report"],
        "allocation_report_sha256": protocol["allocation_report_sha256"],
        "parent_checkpoint": protocol["parent_checkpoint"],
        "parent_checkpoint_sha256": protocol["parent_checkpoint_sha256"],
        "budget_bytes": 10012,
        "storage": storage,
        "target_bits_by_surface": dict(sorted(target_bits.items())),
    }
    target_map_path.write_text(json.dumps(target_map, indent=2) + "\n")
    run = args.out / "run"
    checkpoint = run / f"checkpoint-scale-{args.steps}.pt"
    parent = Path(protocol["parent_checkpoint"])
    training = sealed_command(
        sealed_root,
        "experiments/browser_swin2sr/train_scale_codebook_recovery.py",
        "--parent", str(parent),
        "--target-map", str(target_map_path),
        "--out", str(run),
        "--validation-data", str(args.validation_data),
        "--steps", str(args.steps),
        "--master-learning-rate", str(args.master_learning_rate),
        "--center-learning-rate", str(args.center_learning_rate),
        "--validate-every", "500",
        "--checkpoint-every", "500",
        "--hard-shadow-every", "2",
        "--hard-shadow-weight-start", "0.05",
        "--hard-shadow-weight-end", "0.25",
        "--device", args.device,
    )
    for root in args.data:
        training.extend(("--data", str(root)))
    materialized = args.out / "materialized"
    development = args.out / "development"
    artifact = args.out / "artifact"
    source_sha256 = {
        relative: file_sha256(sealed_root / relative) for relative in SOURCE_FILES
    }
    manifest = {
        "format": FORMAT,
        "status": "sealed_before_mixfrag_qat_results",
        "selection_source": "training_only_until_development_evaluation",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "interaction_result": str(args.interaction_result),
        "interaction_result_sha256": file_sha256(args.interaction_result),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": file_sha256(parent),
        "target_map": str(target_map_path),
        "target_map_sha256": file_sha256(target_map_path),
        "scale_storage_bytes": 10012,
        "target_bit_counts": counts,
        "training_data": dataset_fingerprints(args.data),
        "validation_data": str(args.validation_data),
        "development_bundle": str(args.development_bundle),
        "development_bundle_sha256": file_sha256(args.development_bundle),
        "steps": args.steps,
        "batch_size": 8,
        "master_learning_rate": args.master_learning_rate,
        "center_learning_rate": args.center_learning_rate,
        "optimizer": "adamw_mechanism_control",
        "training_command": training,
        "inspect_command": [*training, "--inspect-only"],
        "final_checkpoint": str(checkpoint),
        "materialization_command": sealed_command(
            sealed_root,
            "experiments/browser_swin2sr/materialize_scale_codebook.py",
            str(checkpoint), "--out", str(materialized), "--source-device", "cpu",
        ),
        "materialization": str(materialized / "materialization.json"),
        "development_evaluation_command": sealed_command(
            sealed_root,
            "experiments/browser_swin2sr/evaluate_stratified.py",
            str(materialized / "dense-hard.pt"),
            "--bundle", str(args.development_bundle),
            "--out", str(development), "--device", args.device,
        ),
        "development_report": str(development / "metrics.json"),
        "artifact_command": sealed_command(
            sealed_root,
            "experiments/browser_swin2sr/export_browser_artifact.py",
            str(materialized / "materialization.json"), "--out", str(artifact),
        ),
        "artifact": str(artifact),
        "advancement_rule": (
            "Advance only if exact storage is preserved and full-development "
            "quality improves on the identical refined parent; no final-heldout access."
        ),
        "source_sha256": source_sha256,
        "source_closure": {
            "verified": True,
            "resolved_local_imports": source_closure,
        },
    }
    manifest_path = args.out / "study-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "study_manifest": str(manifest_path),
        "study_manifest_sha256": file_sha256(manifest_path),
        "target_map": str(target_map_path),
        "target_map_sha256": file_sha256(target_map_path),
        "target_bit_counts": counts,
        "training_command": training,
    }, indent=2))


if __name__ == "__main__":
    main()
