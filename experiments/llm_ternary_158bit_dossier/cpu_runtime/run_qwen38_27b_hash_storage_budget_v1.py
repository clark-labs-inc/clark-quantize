#!/usr/bin/env python3
"""Build an honest complete-model byte projection for the selected hash family."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_hash_storage_budget import (
    HashResidualStorageStrategy,
    complete_hash_storage_ledger,
)
from clark_quantize.qwen38_storage_budget import (
    canonical_inventory_digest,
    inventory_from_safetensors,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources(preregistration: dict) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen hash storage source changed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--hash-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-hash-residual-screen-v1.json"),
    )
    parser.add_argument(
        "--codec-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-fixed-support-codec-v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-hash-storage-budget-v1.json"),
    )
    args = parser.parse_args()

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_hash_storage_budget_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    hash_parent = json.loads(args.hash_receipt.read_text())
    codec_parent = json.loads(args.codec_receipt.read_text())
    if (
        hash_parent.get("schema") != "qwen38_27b_hash_residual_screen_receipt_v1"
        or hash_parent.get("gate", {}).get("passed") is not True
        or codec_parent.get("schema") != "qwen38_27b_fixed_support_codec_receipt_v1"
        or codec_parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("hash storage budget requires both passing parent gates")
    selected = hash_parent["gate"]["selected_candidate"]
    if codec_parent["gate"]["selected_candidate"] != selected:
        raise RuntimeError("hash storage parents selected different candidates")
    parent_prereg_path = runtime / "qwen38_27b_hash_residual_screen_preregistration_v1.json"
    parent_prereg = json.loads(parent_prereg_path.read_text())
    candidates = {
        candidate["name"]: candidate
        for candidate in parent_prereg["candidate_family"]["candidates"]
    }
    if selected not in candidates:
        raise RuntimeError("hash storage selected candidate is unknown")
    candidate = candidates[selected]
    strategy = HashResidualStorageStrategy(
        name=selected,
        group_size=int(parent_prereg["candidate_family"]["group_size"]),
        nonzero_per_group=int(candidate["nonzero_per_group"]),
        input_to_sketch_ratio=int(candidate["input_to_sketch_ratio"]),
        hashes_per_input=int(parent_prereg["candidate_family"]["hashes_per_input"]),
    )

    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if download != {
        "bytes": 55_563_006_776,
        "index_sha256": INDEX_SHA256,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "schema": "qwen38_model_download_v1",
        "shards": 18,
    }:
        raise RuntimeError("Qwen3.8-27B checkpoint identity changed")
    tensors, source = inventory_from_safetensors(args.model_dir)
    expected = prereg["expected_inventory"]
    observed_inventory = {
        "tensor_count": len(tensors),
        "total_parameters": sum(tensor.parameters for tensor in tensors),
        "inventory_sha256": canonical_inventory_digest(tensors),
        "index_sha256": source["index_sha256"],
    }
    if any(observed_inventory[key] != expected[key] for key in expected):
        raise RuntimeError(f"Qwen3.8 hash storage inventory changed: {observed_inventory}")
    ledger = complete_hash_storage_ledger(
        tensors, strategy, auxiliary_bytes=source["auxiliary_bytes"]
    )
    gate_spec = prereg["gate"]
    eligible_fraction = (
        ledger["parameters"]["fixed_support_hash"] / ledger["parameters"]["total"]
    )
    gate = {
        **gate_spec,
        "selected_candidate": selected,
        "observed_complete_bundle_bits_per_total_parameter": ledger[
            "effective_bits_per_total_parameter"
        ],
        "observed_fixed_support_parameter_fraction": eligible_fraction,
        "observed_total_parameters": ledger["parameters"]["total"],
        "observed_total_tensors": ledger["tensors"]["total"],
        "observed_accounting_identity": (
            ledger["parameters"]["fixed_support_hash"]
            + ledger["parameters"]["retained_exact"]
            == ledger["parameters"]["total"]
        ),
    }
    gate["passed"] = bool(
        gate["observed_accounting_identity"]
        and ledger["parameters"]["total"] == int(gate_spec["required_total_parameters"])
        and ledger["tensors"]["total"] == int(gate_spec["required_total_tensors"])
        and ledger["effective_bits_per_total_parameter"]
        <= float(gate_spec["maximum_complete_bundle_bits_per_total_parameter"])
        and eligible_fraction >= float(gate_spec["minimum_fixed_support_parameter_fraction"])
    )
    receipt = {
        "schema": "qwen38_27b_hash_storage_budget_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "complete": True,
        "terminal": True,
        "preregistration_sha256": sha256_file(prereg_path),
        "hash_parent_receipt_sha256": sha256_file(args.hash_receipt),
        "codec_parent_receipt_sha256": sha256_file(args.codec_receipt),
        "source": source,
        "inventory": observed_inventory,
        "ledger": ledger,
        "gate": gate,
        "claim_boundary": {
            "complete_checkpoint_inventory_counted": True,
            "all_ineligible_tensors_retained_at_source_dtype": True,
            "all_headers_alignment_scale_side_channels_metadata_and_auxiliary_files_counted": True,
            "active_projection_compute_metadata_reported_separately": True,
            "complete_artifact_exported": False,
            "whole_model_fixed_support_fit_run": False,
            "scale_quality_proven": False,
            "capability_preservation_proven": False,
            "end_to_end_speedup_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
