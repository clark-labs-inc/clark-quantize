#!/usr/bin/env python3
"""Build and gate the complete restartable Qwen3.8-27B aligned-2 cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen38_aligned2_cache import (
    CACHE_MANIFEST_SCHEMA,
    Qwen38Aligned2CacheBuilder,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for content in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(content)
    return digest.hexdigest()


def validate_frozen_inputs(preregistration: dict) -> None:
    for label, record in preregistration["frozen_inputs"].items():
        path = REPO / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen full-cache input changed: {label}")


def allocated_bytes(path: Path) -> int:
    return path.stat().st_blocks * 512 if path.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-cq3-row-normalized-container-v1/"
            "Qwen3.8-27B-row-normalized.cq3"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-aligned2-cache-v1/"
            "Qwen3.8-27B-row-normalized.a2"
        ),
    )
    parser.add_argument(
        "--parent-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-cache-v1.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-full-cache-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = (
        runtime / "qwen38_27b_aligned2_full_cache_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    parent_content = args.parent_receipt.read_bytes()
    parent = json.loads(parent_content)
    if (
        parent.get("schema") != "qwen38_27b_aligned2_cache_receipt_v1"
        or parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("aligned2 actual-tensor parent gate did not pass")
    parent_container = parent["container"]
    if (
        args.container.resolve() != Path(parent_container["path"]).resolve()
        or args.container.stat().st_size != int(parent_container["bytes"])
        or sha256_file(args.container) != parent_container["sha256"]
    ):
        raise RuntimeError("full-cache container differs from its passed parent")

    expected_cache_bytes = int(prereg["scope"]["cache_file_bytes"])
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    disk_before = shutil.disk_usage(args.cache.parent)
    remaining_allocation = max(
        0, expected_cache_bytes - allocated_bytes(args.cache)
    )
    required_after = int(prereg["resources"]["minimum_free_bytes_after_build"])
    if disk_before.free - remaining_allocation < required_after:
        raise SystemExit(
            "infrastructure-invalid: insufficient disk for the bounded "
            "Qwen3.8-27B aligned2 cache"
        )

    builder = Qwen38Aligned2CacheBuilder(args.container, args.cache)
    preexisting_completed = 0
    if builder.state_path.is_file():
        preexisting_completed = len(
            json.loads(builder.state_path.read_text()).get("completed", [])
        )
    started = time.perf_counter()
    manifest = builder.build(
        workers=int(prereg["measurement"]["physical_workers"])
    )
    verified = builder.verify_manifest(verify_full_cache=True)
    elapsed = time.perf_counter() - started
    disk_after = shutil.disk_usage(args.cache.parent)
    ledger = verified["identity"]["ledger"]
    gate_spec = prereg["gate"]
    finite = bool(
        math.isfinite(elapsed)
        and elapsed > 0
        and math.isfinite(ledger["active_runtime_bits_per_model_parameter"])
    )
    observations = {
        "observed_manifest_verified": manifest == verified,
        "observed_tensor_count": ledger["tensor_count"],
        "observed_ternary_parameters": ledger["ternary_parameters"],
        "observed_cache_file_bytes": ledger["cache_file_bytes"],
        "observed_active_runtime_bits_per_model_parameter": ledger[
            "active_runtime_bits_per_model_parameter"
        ],
        "observed_free_bytes_after_build": disk_after.free,
        "observed_elapsed_seconds": elapsed,
        "observed_no_full_int8_weight_matrix": (
            verified["cache"]["full_int8_weight_matrix_materialized"]
            is False
        ),
        "observed_all_measurements_finite": finite,
    }
    gate = {**gate_spec, **observations}
    gate["passed"] = (
        observations["observed_manifest_verified"]
        and observations["observed_tensor_count"]
        == gate_spec["required_tensor_count"]
        and observations["observed_ternary_parameters"]
        == gate_spec["required_ternary_parameters"]
        and observations["observed_cache_file_bytes"]
        == gate_spec["required_cache_file_bytes"]
        and observations[
            "observed_active_runtime_bits_per_model_parameter"
        ]
        <= gate_spec["maximum_active_runtime_bits_per_model_parameter"]
        and observations["observed_free_bytes_after_build"]
        >= gate_spec["minimum_free_bytes_after_build"]
        and observations["observed_elapsed_seconds"]
        <= gate_spec["maximum_elapsed_seconds"]
        and observations["observed_no_full_int8_weight_matrix"]
        and finite
    )
    receipt = {
        "schema": "qwen38_27b_aligned2_full_cache_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parent": {
            "path": str(args.parent_receipt),
            "sha256": hashlib.sha256(parent_content).hexdigest(),
        },
        "source_container": parent_container,
        "cache": {
            **verified["cache"],
            "manifest_path": str(builder.manifest_path),
            "manifest_sha256": sha256_file(builder.manifest_path),
            "checkpoint_path": str(builder.state_path),
            "checkpoint_sha256": sha256_file(builder.state_path),
        },
        "ledger": ledger,
        "restartability": {
            "preexisting_completed_tensors": preexisting_completed,
            "committed_tensors": len(verified["tensors"]),
            "per_tensor_sha256": True,
            "per_tensor_atomic_checkpoint": True,
        },
        "measurement": {
            "elapsed_seconds": elapsed,
            "physical_workers": prereg["measurement"]["physical_workers"],
            "disk_before": {
                "total": disk_before.total,
                "used": disk_before.used,
                "free": disk_before.free,
            },
            "disk_after": {
                "total": disk_after.total,
                "used": disk_after.used,
                "free": disk_after.free,
            },
        },
        "gate": gate,
        "claim_boundary": verified["claim_boundary"],
    }
    atomic_bytes(
        args.receipt,
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
