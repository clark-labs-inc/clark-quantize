#!/usr/bin/env python3
"""Emit the frozen complete Qwen3.8-27B half-support plus hash byte ledger."""
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
from clark_quantize.qwen38_half_support_ledger import (
    build_half_support_checkpoint_receipt,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-half-support-complete-ledger-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / (
        "qwen38_27b_half_support_complete_ledger_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    for label, descriptor in prereg["frozen_inputs"].items():
        source = REPO / descriptor["path"]
        if not source.is_file() or sha256_file(source) != descriptor["sha256"]:
            raise RuntimeError(f"frozen half-support ledger source changed: {label}")

    download_path = args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json"
    download = json.loads(download_path.read_text())
    pinned = prereg["model"]
    if (
        download.get("model_id") != pinned["id"]
        or download.get("revision") != pinned["revision"]
        or download.get("index_sha256") != pinned["index_sha256"]
        or download.get("shards") != pinned["shards"]
        or download.get("bytes") != pinned["source_shard_bytes"]
    ):
        raise RuntimeError("Qwen3.8-27B model identity changed")

    receipt = build_half_support_checkpoint_receipt(args.model_dir)
    expected = prereg["expected_inventory"]
    ledger = receipt["ledger"]
    observed = {
        "tensor_count": ledger["tensors"]["total"],
        "total_parameters": ledger["parameters"]["total"],
        "compressed_tensors": ledger["tensors"]["compressed"],
        "compressed_parameters": ledger["parameters"]["compressed"],
        "retained_tensors": ledger["tensors"]["retained_exact"],
        "retained_parameters": ledger["parameters"]["retained_exact"],
        "compressed_tail_tensors": ledger["tensors"]["compressed_with_even_tail"],
    }
    if observed != expected:
        raise RuntimeError(
            f"Qwen3.8-27B half-support inventory changed: {observed} != {expected}"
        )
    if receipt["source"]["index_sha256"] != pinned["index_sha256"]:
        raise RuntimeError("Qwen3.8-27B index changed during half-support inventory")
    rate = ledger["effective_bits_per_total_parameter"]
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_effective_bits_per_total_parameter": rate,
        "observed_complete_bundle_bytes": ledger["bytes"]["complete_bundle"],
        "observed_compressed_component_bits_per_parameter": ledger[
            "compressed_component_bits_per_compressed_parameter"
        ],
    }
    gate["passed"] = bool(
        rate <= float(gate_spec["maximum_effective_bits_per_total_parameter"])
        and observed == expected
        and ledger["policy"]["route_sharing_assumed"] is False
    )
    receipt.update(
        {
            "schema": "qwen38_27b_half_support_complete_ledger_receipt_v1",
            "created_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "complete": True,
            "terminal": True,
            "repository_commit": subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
            ).strip(),
            "preregistration_sha256": sha256_file(prereg_path),
            "model_identity": download,
            "observed_inventory": observed,
            "gate": gate,
        }
    )
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
