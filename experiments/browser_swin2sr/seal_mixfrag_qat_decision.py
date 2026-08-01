#!/usr/bin/env python3
"""Seal MixFrag-QAT quality/storage gates before the first major checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-mixfrag-scale-qat-decision-protocol-v1"
SOURCES = (
    "experiments/browser_swin2sr/decide_mixfrag_qat.py",
    "experiments/browser_swin2sr/evaluate_promotion.py",
    "experiments/browser_swin2sr/full_surface_provenance.py",
    "experiments/browser_swin2sr/training_data_contract.py",
    "experiments/browser_swin2sr/seal_mixfrag_qat_decision.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, required=True)
    parser.add_argument("--refined-parent-report", type=Path, required=True)
    parser.add_argument("--released-ternary-report", type=Path, required=True)
    parser.add_argument("--sensitive-bridge-report", type=Path, required=True)
    parser.add_argument("--leader-report", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def input_record(name: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"name": name, "path": str(path.resolve()), "sha256": file_sha256(path)}


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"decision protocol exists: {args.out}")
    if len(args.leader_report) != 3:
        raise ValueError("exactly three all-ternary leader reports are required")
    study = json.loads(args.study_manifest.read_text())
    if study.get("format") != "backdropkit-swin2sr-mixfrag-scale-qat-study-v1":
        raise ValueError("unsupported MixFrag QAT study")
    if study.get("model_promotion_authorized") or not study.get("final_heldout_locked"):
        raise RuntimeError("MixFrag QAT study violates promotion isolation")
    target_map = Path(study["target_map"])
    if file_sha256(target_map) != study["target_map_sha256"]:
        raise RuntimeError("MixFrag target map changed")
    repository = Path(__file__).resolve().parents[2]
    sealed_root = args.out / "decision_sealed_sources"
    source_sha256 = {}
    for relative in SOURCES:
        destination = sealed_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, destination)
        source_sha256[relative] = file_sha256(destination)
    leader_seeds = [20260731, 20260732, 20260733]
    inputs = [
        input_record("study_manifest", args.study_manifest),
        input_record("target_map", target_map),
        input_record("q4_reference", args.q4_reference),
        input_record("promotion_policy", args.promotion_policy),
        input_record("refined_parent_report", args.refined_parent_report),
        input_record("released_ternary_report", args.released_ternary_report),
        input_record("sensitive_bridge_report", args.sensitive_bridge_report),
    ]
    inputs.extend(
        input_record(f"all_ternary_leader_seed_{seed}", path)
        for seed, path in zip(leader_seeds, args.leader_report)
    )
    protocol = {
        "format": FORMAT,
        "status": "sealed_before_first_major_qat_checkpoint",
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "leader_seeds": leader_seeds,
        "exact_scale_bytes": 10012,
        "mechanism_rule": (
            "Exact 10,012 scale bytes, official paired and stratified "
            "noninferiority versus the refined parent, and positive mean "
            "ground-truth PSNR delta versus both released ternary and the "
            "prior sensitive bridge."
        ),
        "leader_rule": (
            "The mechanism rule plus positive mean ground-truth and edge PSNR "
            "delta versus the three-seed all-ternary leader mean."
        ),
        "inputs": inputs,
        "sealed_source_root": str(sealed_root.resolve()),
        "source_sha256": source_sha256,
    }
    decision = args.out / "decision.json"
    browser = sealed_root / "experiments/browser_swin2sr"
    protocol["decision_command"] = [
        "/usr/bin/env",
        f"PYTHONPATH={browser}:{sealed_root}",
        sys.executable,
        str(browser / "decide_mixfrag_qat.py"),
        "--protocol", str(args.out / "decision-protocol.json"),
        "--out", str(decision),
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "decision-protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "protocol": str(path),
        "protocol_sha256": file_sha256(path),
        "decision_command": protocol["decision_command"],
    }, indent=2))


if __name__ == "__main__":
    main()
