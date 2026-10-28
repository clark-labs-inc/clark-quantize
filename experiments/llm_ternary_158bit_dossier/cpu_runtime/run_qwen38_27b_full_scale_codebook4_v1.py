#!/usr/bin/env python3
"""Close the 264-tensor scale gap in the complete Qwen3.8-27B layout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_full_scale_codebook import (
    aggregate_scale_records,
    fit_scale_codebook_chunked,
    scale_error_metrics,
)
from clark_quantize.qwen38_storage_budget import inventory_from_safetensors
from clark_quantize.scale_codebook import pack_nibbles, unpack_nibbles
from clark_quantize.ternary import ternary_codes_scales


MODEL_ID = "Qwen/Qwen3.8-27B"
EXPECTED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
EXPECTED_INVENTORY_SHA256 = "8533a0edfba093faf2a865391405408461613b92ef83e8d511436f2ed96bc14d"
EXPECTED_CODE_MANIFEST_SHA256 = "c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4"
EXPECTED_TOTAL_TENSORS = 666
EXPECTED_TOTAL_PARAMETERS = 27_780_313_088
EXPECTED_TOTAL_SCALES = 217_521_520
EXPECTED_MAPPED_TENSORS = 402
EXPECTED_MAPPED_PARAMETERS = 26_869_760_000
EXPECTED_MAPPED_SCALES = 209_920_000
EXPECTED_SUPPLEMENTAL_TENSORS = 264
EXPECTED_SUPPLEMENTAL_PARAMETERS = 910_553_088
EXPECTED_SUPPLEMENTAL_SCALES = 7_601_520


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resource_guard(workspace: Path) -> None:
    competitors = (
        "quant-qwen38-27b-cpu-q4-baseline-v1.service",
        "quant-qwen38-27b-postbaseline-chain-v1.service",
        "quant-qwen38-27b-paired-throughput-watch-v1.service",
        "quant-qwen38-27b-cq3-x86-pair-watch-v1.service",
        "quant-qwen38-27b-scale-codebook4-watch-v1.service",
        "quant-qwen38-27b-block203-codec-watch-v2.service",
        "quant-qwen38-27b-block203-radix243-watch-v1.service",
    )
    for unit in competitors:
        if subprocess.run(
            ["systemctl", "is-active", "--quiet", unit], check=False
        ).returncode == 0:
            raise SystemExit(f"infrastructure-invalid: refusing to compete with {unit}")
    fields = {
        line.split(":", 1)[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line and len(line.split()) >= 2 and line.split()[1].isdigit()
    }
    if fields.get("MemAvailable", 0) < 8 * 1024 * 1024:
        raise SystemExit("infrastructure-invalid: less than 8 GiB available memory")
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < 4 * 1024**3:
        raise SystemExit("infrastructure-invalid: less than 4 GiB free disk")
    if len(Path("/proc/swaps").read_text().splitlines()) > 1:
        raise SystemExit("infrastructure-invalid: swap is active")


def validate_output(root: Path, descriptor: dict) -> None:
    path = root / descriptor["path"]
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise RuntimeError(f"authenticated scale-codebook output is missing or changed: {path}")


def validate_mapped_parent(root: Path, manifest: dict) -> list[dict]:
    if (
        manifest.get("schema") != "qwen38_27b_scale_codebook4_receipt_v1"
        or manifest.get("complete") is not True
        or manifest.get("gate", {}).get("passed") is not True
    ):
        raise SystemExit("valid-nonpromotion: mapped 402-tensor scale gate did not pass")
    records = manifest.get("records", [])
    if len(records) != EXPECTED_MAPPED_TENSORS:
        raise RuntimeError("mapped scale-codebook parent has incomplete tensor coverage")
    for record in records:
        validate_output(root, record["indices"])
        validate_output(root, record["codebook"])
    totals = aggregate_scale_records(records)
    if totals.scales != EXPECTED_MAPPED_SCALES:
        raise RuntimeError("mapped scale-codebook parent has an unexpected denominator")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--mapped-artifact",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"),
    )
    parser.add_argument(
        "--mapped-codebooks",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-scale-codebook4-v1"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-full-scale-codebook4-v1"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-full-scale-codebook4-v1.json"),
    )
    parser.add_argument("--row-batch", type=int, default=128)
    parser.add_argument("--fit-chunk", type=int, default=262_144)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.row_batch <= 0 or args.fit_chunk <= 0 or args.threads <= 0:
        raise ValueError("row batch, fit chunk, and thread count must be positive")
    resource_guard(Path("/workspace"))
    torch.set_num_threads(args.threads)

    runtime_dir = Path(__file__).resolve().parent
    prereg_path = runtime_dir / "qwen38_27b_full_scale_codebook4_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    for item in prereg["frozen_inputs"].values():
        path = REPO / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {path}")

    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if download != {
        "bytes": 55_563_006_776,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "model_id": MODEL_ID,
        "revision": EXPECTED_REVISION,
        "schema": "qwen38_model_download_v1",
        "shards": 18,
    }:
        raise RuntimeError("Qwen3.8-27B checkpoint identity mismatch")

    code_manifest_path = args.mapped_artifact / "codes.cq3d" / "manifest.json"
    if sha256_file(code_manifest_path) != EXPECTED_CODE_MANIFEST_SHA256:
        raise RuntimeError("mapped Qwen3.8-27B ternary source manifest changed")
    code_manifest = json.loads(code_manifest_path.read_text())
    mapped_sources = code_manifest.get("identity", {}).get("mapped_sources", [])
    mapped_names = {record["name"] for record in mapped_sources}
    if len(mapped_names) != EXPECTED_MAPPED_TENSORS:
        raise RuntimeError("mapped ternary source tensor coverage changed")

    mapped_manifest_path = args.mapped_codebooks / "manifest.json"
    mapped_manifest = json.loads(mapped_manifest_path.read_text())
    mapped_records = validate_mapped_parent(args.mapped_codebooks, mapped_manifest)
    if {record["tensor"] for record in mapped_records} != mapped_names:
        raise RuntimeError("mapped codebook and ternary tensor sets differ")

    tensors, source = inventory_from_safetensors(args.model_dir)
    if source["index_sha256"] != EXPECTED_INDEX_SHA256 or source["inventory_sha256"] != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError("complete Qwen3.8-27B tensor inventory changed")
    quantized = sorted((tensor for tensor in tensors if tensor.quantized), key=lambda item: item.name)
    if (
        len(quantized) != EXPECTED_TOTAL_TENSORS
        or sum(tensor.parameters for tensor in quantized) != EXPECTED_TOTAL_PARAMETERS
    ):
        raise RuntimeError("complete rank-2+ Qwen3.8-27B coverage changed")
    supplemental = [tensor for tensor in quantized if tensor.name not in mapped_names]
    if (
        len(supplemental) != EXPECTED_SUPPLEMENTAL_TENSORS
        or sum(tensor.parameters for tensor in supplemental) != EXPECTED_SUPPLEMENTAL_PARAMETERS
        or sum(tensor.rows * math.ceil(tensor.columns / 128) for tensor in supplemental)
        != EXPECTED_SUPPLEMENTAL_SCALES
    ):
        raise RuntimeError("supplemental Qwen3.8-27B tensor coverage changed")

    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    ordered = sorted(supplemental, key=lambda item: (weight_map[item.name], item.name))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.out_dir / "supplemental"
    output_root.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / "manifest.partial.json"
    final_path = args.out_dir / "manifest.json"
    run_identity = {
        "schema": "qwen38_27b_full_scale_codebook4_identity_v1",
        "model_id": MODEL_ID,
        "model_revision": EXPECTED_REVISION,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "mapped_code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "mapped_codebook_manifest_sha256": sha256_file(mapped_manifest_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "algorithm": {
            "group_size": 128,
            "ternary_method": "optimal",
            "entries": 16,
            "entry_zero_fixed": True,
            "fit": "deterministic unweighted chunked Lloyd k-means",
            "iterations": 50,
            "stored_codebook_dtype": "float16",
            "stored_indices": "two 4-bit indices per byte",
        },
    }
    if final_path.exists():
        result = json.loads(final_path.read_text())
        if result.get("identity") != run_identity or result.get("complete") is not True:
            raise RuntimeError("existing full scale-codebook manifest identity mismatch")
        for record in result["supplemental_records"]:
            validate_output(args.out_dir, record["indices"])
            validate_output(args.out_dir, record["codebook"])
        if not args.receipt.exists():
            atomic_bytes(args.receipt, final_path.read_bytes())
        elif sha256_file(args.receipt) != sha256_file(final_path):
            raise RuntimeError("full scale-codebook manifest and receipt diverged")
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if partial_path.exists():
        result = json.loads(partial_path.read_text())
        if result.get("identity") != run_identity or result.get("complete") is not False:
            raise RuntimeError("partial full scale-codebook manifest identity mismatch")
    else:
        result = {"identity": run_identity, "supplemental_records": [], "complete": False}
        atomic_bytes(partial_path, canonical_json(result))

    by_shard: dict[str, list] = {}
    for tensor in ordered:
        by_shard.setdefault(weight_map[tensor.name], []).append(tensor)
    supplemental_index = 0
    for shard, shard_tensors in sorted(by_shard.items()):
        with safe_open(args.model_dir / shard, framework="pt", device="cpu") as handle:
            for tensor in shard_tensors:
                name = tensor.name
                if supplemental_index < len(result["supplemental_records"]):
                    record = result["supplemental_records"][supplemental_index]
                    if (
                        record.get("tensor") != name
                        or record.get("shape") != list(tensor.shape)
                        or record.get("source_shard") != shard
                    ):
                        raise RuntimeError(f"resume tensor identity mismatch at {supplemental_index}")
                    validate_output(args.out_dir, record["indices"])
                    validate_output(args.out_dir, record["codebook"])
                    supplemental_index += 1
                    continue

                scale_parts: list[np.ndarray] = []
                code_digest = hashlib.sha256()
                scale_digest = hashlib.sha256()
                view = handle.get_slice(name)
                for row_start in range(0, tensor.rows, args.row_batch):
                    row_stop = min(tensor.rows, row_start + args.row_batch)
                    source_rows = view[row_start:row_stop].contiguous().reshape(
                        row_stop - row_start, tensor.columns
                    ).float()
                    codes, scales = ternary_codes_scales(
                        source_rows, group=128, method="optimal"
                    )
                    code_payload = codes.numpy().astype(np.int8, copy=False).tobytes()
                    scale_payload = scales.numpy().astype("<f2", copy=False).tobytes()
                    code_digest.update(code_payload)
                    scale_digest.update(scale_payload)
                    scale_parts.append(np.frombuffer(scale_payload, dtype="<f2").copy())
                    del source_rows, codes, scales
                source_scales = np.concatenate(scale_parts).astype(np.float32)
                expected_scales = tensor.rows * math.ceil(tensor.columns / 128)
                if source_scales.size != expected_scales:
                    raise RuntimeError(f"generated scale count mismatch for {name}")
                fitted = fit_scale_codebook_chunked(
                    source_scales,
                    entries=16,
                    iterations=50,
                    chunk_size=args.fit_chunk,
                )
                index_payload = pack_nibbles(fitted.indices)
                np.testing.assert_array_equal(
                    unpack_nibbles(index_payload, source_scales.size), fitted.indices
                )
                codebook_payload = fitted.values.astype("<f2", copy=False).tobytes()
                index_rel = f"supplemental/{supplemental_index:04d}.idx4"
                codebook_rel = f"supplemental/{supplemental_index:04d}.f16"
                atomic_bytes(args.out_dir / index_rel, index_payload)
                atomic_bytes(args.out_dir / codebook_rel, codebook_payload)
                record = {
                    "tensor": name,
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype,
                    "parameters": tensor.parameters,
                    "source_shard": shard,
                    "scale_count": int(source_scales.size),
                    "source_ternary_codes_sha256": code_digest.hexdigest(),
                    "source_scale_sha256": scale_digest.hexdigest(),
                    "indices": {
                        "path": index_rel,
                        "bytes": len(index_payload),
                        "sha256": sha256_bytes(index_payload),
                        "exact_nibble_roundtrip": True,
                    },
                    "codebook": {
                        "path": codebook_rel,
                        "bytes": len(codebook_payload),
                        "sha256": sha256_bytes(codebook_payload),
                        "values": fitted.values.astype(np.float32).tolist(),
                    },
                    "metrics": scale_error_metrics(source_scales, fitted),
                }
                result["supplemental_records"].append(record)
                atomic_bytes(partial_path, canonical_json(result))
                supplemental_index += 1

    supplemental_records = result["supplemental_records"]
    if len(supplemental_records) != EXPECTED_SUPPLEMENTAL_TENSORS:
        raise RuntimeError("supplemental scale-codebook coverage is incomplete")
    all_records = mapped_records + supplemental_records
    if {record["tensor"] for record in all_records} != {tensor.name for tensor in quantized}:
        raise RuntimeError("full scale-codebook tensor set differs from the checkpoint inventory")
    totals = aggregate_scale_records(all_records)
    if totals.tensors != EXPECTED_TOTAL_TENSORS or totals.scales != EXPECTED_TOTAL_SCALES:
        raise RuntimeError("full scale-codebook denominator is incomplete")
    gate = {
        "maximum_global_normalized_rmse": 0.02,
        "maximum_per_tensor_normalized_rmse": 0.05,
        "maximum_per_tensor_p99_relative_error_nonzero_scales": 0.10,
        "observed_global_normalized_rmse": totals.global_nrmse,
        "observed_maximum_per_tensor_normalized_rmse": totals.maximum_tensor_nrmse,
        "observed_maximum_per_tensor_p99_relative_error_nonzero_scales": (
            totals.maximum_tensor_p99_relative_error
        ),
    }
    gate["passed"] = (
        totals.global_nrmse <= gate["maximum_global_normalized_rmse"]
        and totals.maximum_tensor_nrmse <= gate["maximum_per_tensor_normalized_rmse"]
        and totals.maximum_tensor_p99_relative_error
        <= gate["maximum_per_tensor_p99_relative_error_nonzero_scales"]
    )
    result.update(
        {
            "schema": "qwen38_27b_full_scale_codebook4_receipt_v1",
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repository_commit": subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
            ).strip(),
            "complete": True,
            "mapped_parent": {
                "manifest": str(mapped_manifest_path),
                "sha256": sha256_file(mapped_manifest_path),
                "tensors": EXPECTED_MAPPED_TENSORS,
                "parameters": EXPECTED_MAPPED_PARAMETERS,
                "scales": EXPECTED_MAPPED_SCALES,
            },
            "summary": {
                "tensors": totals.tensors,
                "ternary_parameters": EXPECTED_TOTAL_PARAMETERS,
                "scales": totals.scales,
                "supplemental_tensors": EXPECTED_SUPPLEMENTAL_TENSORS,
                "supplemental_parameters": EXPECTED_SUPPLEMENTAL_PARAMETERS,
                "supplemental_scales": EXPECTED_SUPPLEMENTAL_SCALES,
                "global_normalized_rmse": totals.global_nrmse,
                "maximum_tensor_normalized_rmse": totals.maximum_tensor_nrmse,
                "maximum_tensor_p99_relative_error_nonzero_scales": (
                    totals.maximum_tensor_p99_relative_error
                ),
                "maximum_absolute_error": totals.maximum_absolute_error,
            },
            "gate": gate,
            "claim_boundary": {
                "all_666_rank_ge_2_qwen38_tensors_covered": True,
                "all_217521520_group_scales_covered": True,
                "model_quality_proven": False,
                "artifact_exported": False,
                "scientific_breakthrough_proven": False,
            },
        }
    )
    atomic_bytes(final_path, canonical_json(result))
    atomic_bytes(args.receipt, canonical_json(result))
    partial_path.unlink(missing_ok=True)
    marker = args.out_dir / ("PROMOTION_PASS" if gate["passed"] else "PROMOTION_REJECTED")
    atomic_bytes(marker, ("pass\n" if gate["passed"] else "rejected\n").encode())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
