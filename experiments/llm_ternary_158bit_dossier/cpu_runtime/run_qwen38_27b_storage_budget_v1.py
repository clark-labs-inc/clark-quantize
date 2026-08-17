#!/usr/bin/env python3
"""Emit the preregistered whole-checkpoint Qwen3.8-27B storage projection."""
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
from clark_quantize.qwen38_storage_budget import build_receipt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/qwen38-27b-storage-budget-v1.json"))
    args = parser.parse_args()
    prereg_path = Path(__file__).with_name("qwen38_27b_storage_budget_preregistration_v1.json")
    prereg = json.loads(prereg_path.read_text())
    for item in prereg["frozen_inputs"].values():
        path = REPO / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {path}")

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
        raise RuntimeError("Qwen3.8-27B download identity does not match preregistration")

    receipt = build_receipt(args.model_dir)
    expected = {
        "tensor_count": 1199,
        "total_parameters": 27781427952,
        "ternary_rank_ge_2_tensors": 666,
        "ternary_rank_ge_2_parameters": 27780313088,
        "retained_rank_1_tensors": 533,
        "retained_rank_1_parameters": 1114864,
    }
    if receipt["inventory"] != expected:
        raise RuntimeError(
            f"Qwen3.8-27B inventory changed: expected {expected}, got {receipt['inventory']}"
        )
    if receipt["source"]["index_sha256"] != pinned["index_sha256"]:
        raise RuntimeError("checkpoint index SHA-256 changed during inventory")
    receipt["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt["repository_commit"] = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt["preregistration"] = {
        "path": str(prereg_path),
        "sha256": sha256(prereg_path),
    }
    receipt["model_identity"] = download
    candidate = receipt["strategies"][receipt["candidate"]["name"]]
    receipt["gate"] = {
        "maximum_projected_complete_bundle_bpw": 1.63,
        "observed_projected_complete_bundle_bpw": candidate["effective_bits_per_total_parameter"],
        "passed": candidate["effective_bits_per_total_parameter"] <= 1.63,
        "scope": "byte-accounting feasibility only",
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
