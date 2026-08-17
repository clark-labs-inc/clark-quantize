#!/usr/bin/env python3
"""Fit restartable per-tensor 4-bit scale codebooks to the real 27B artifact."""
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.scale_codebook import fit_scale_codebook, pack_nibbles, unpack_nibbles


EXPECTED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
EXPECTED_CODE_MANIFEST_SHA256 = "c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4"
EXPECTED_TENSORS = 402
EXPECTED_SCALES = 209_920_000
EXPECTED_SCALE_BYTES = 419_840_000


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_authoritative_service() -> str | None:
    for unit in (
        "quant-qwen38-27b-cpu-q4-baseline-v1.service",
        "quant-qwen38-27b-postbaseline-chain-v1.service",
        "quant-qwen38-27b-paired-throughput-watch-v1.service",
        "quant-qwen38-27b-cq3-x86-pair-watch-v1.service",
    ):
        if subprocess.run(
            ["systemctl", "is-active", "--quiet", unit], check=False
        ).returncode == 0:
            return unit
    return None


def resource_guard(workspace: Path) -> None:
    active = active_authoritative_service()
    if active:
        raise SystemExit(f"infrastructure-invalid: refusing to compete with {active}")
    fields = {
        line.split(":", 1)[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line and len(line.split()) >= 2 and line.split()[1].isdigit()
    }
    if fields.get("MemAvailable", 0) < 8 * 1024 * 1024:
        raise SystemExit("infrastructure-invalid: less than 8 GiB available memory")
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < 2 * 1024**3:
        raise SystemExit("infrastructure-invalid: less than 2 GiB free disk")
    swaps = Path("/proc/swaps").read_text().splitlines()
    if len(swaps) > 1:
        raise SystemExit("infrastructure-invalid: swap is active")


def load_scale_bytes(artifact: Path, records: list[dict]) -> bytes:
    content = bytearray()
    for record in records:
        descriptor = record["descriptor"]
        path = artifact / "scales" / descriptor["scale_file"]
        raw = path.read_bytes()
        if len(raw) != descriptor["scale_bytes"] or sha256_bytes(raw) != descriptor["scale_sha256"]:
            raise ValueError(f"source scale block failed authentication: {path}")
        content.extend(raw)
    return bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-scale-codebook4-v1"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-scale-codebook4-v1.json"),
    )
    args = parser.parse_args()
    resource_guard(Path("/workspace"))

    runtime_dir = Path(__file__).resolve().parent
    prereg_path = runtime_dir / "qwen38_27b_scale_codebook4_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    for item in prereg["frozen_inputs"].values():
        path = REPO / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {path}")

    code_manifest_path = args.artifact / "codes.cq3d" / "manifest.json"
    if sha256_file(code_manifest_path) != EXPECTED_CODE_MANIFEST_SHA256:
        raise RuntimeError("source code manifest SHA-256 mismatch")
    source_manifest = json.loads(code_manifest_path.read_text())
    identity = source_manifest["identity"]
    if (
        identity.get("model_revision") != EXPECTED_REVISION
        or identity.get("index_sha256") != EXPECTED_INDEX_SHA256
        or identity.get("quantizer")
        != {"method": "optimal", "group_size": 128, "scale_dtype": "float16"}
        or len(identity.get("mapped_sources", [])) != EXPECTED_TENSORS
        or source_manifest.get("extra", {}).get("scale_count") != EXPECTED_SCALES
        or source_manifest.get("extra", {}).get("scale_bytes") != EXPECTED_SCALE_BYTES
    ):
        raise RuntimeError("source Qwen3.8-27B artifact identity mismatch")

    by_tensor: dict[str, list[dict]] = {source["name"]: [] for source in identity["mapped_sources"]}
    for record in source_manifest["chunks"]:
        by_tensor[record["descriptor"]["tensor"]].append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = args.out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / "manifest.partial.json"
    final_path = args.out_dir / "manifest.json"
    run_identity = {
        "schema": "qwen38_27b_scale_codebook4_identity_v1",
        "model_revision": EXPECTED_REVISION,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "source_code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "algorithm": {
            "entries": 16,
            "entry_zero_fixed": True,
            "fit": "deterministic unweighted linear-scale Lloyd k-means",
            "stored_codebook_dtype": "float16",
            "stored_indices": "two 4-bit indices per byte",
        },
        "preregistration_sha256": sha256_file(prereg_path),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    if final_path.exists():
        complete = json.loads(final_path.read_text())
        if complete.get("identity") != run_identity or not complete.get("complete"):
            raise RuntimeError("existing final scale-codebook manifest identity mismatch")
        if not args.receipt.exists():
            atomic_bytes(args.receipt, final_path.read_bytes())
        elif sha256_file(final_path) != sha256_file(args.receipt):
            raise RuntimeError("final manifest and receipt diverged")
        print(json.dumps(complete, indent=2, sort_keys=True))
        return
    if partial_path.exists():
        result = json.loads(partial_path.read_text())
        if result.get("identity") != run_identity or result.get("complete"):
            raise RuntimeError("partial scale-codebook manifest identity mismatch")
    else:
        result = {"identity": run_identity, "records": [], "complete": False}
        atomic_bytes(partial_path, canonical_json(result))

    total_error_squared = 0.0
    total_source_squared = 0.0
    total_scales = 0
    total_index_bytes = 0
    total_codebook_bytes = 0
    maximum_tensor_nrmse = 0.0
    maximum_tensor_p99_relative_error = 0.0
    maximum_absolute_error = 0.0
    reused = 0
    for tensor_index, source in enumerate(identity["mapped_sources"]):
        name = source["name"]
        records = by_tensor[name]
        raw = load_scale_bytes(args.artifact, records)
        source_sha256 = sha256_bytes(raw)
        scales = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        expected = source["shape"][0] * math.ceil(source["shape"][1] / 128)
        if scales.size != expected:
            raise RuntimeError(f"scale count mismatch for {name}: {scales.size} != {expected}")

        if tensor_index < len(result["records"]):
            record = result["records"][tensor_index]
            if record.get("tensor") != name or record.get("source_scale_sha256") != source_sha256:
                raise RuntimeError(f"resume source mismatch at tensor {tensor_index}: {name}")
            for key in ("indices", "codebook"):
                path = args.out_dir / record[key]["path"]
                if path.stat().st_size != record[key]["bytes"] or sha256_file(path) != record[key]["sha256"]:
                    raise RuntimeError(f"resume output mismatch: {path}")
            reused += 1
        else:
            fitted = fit_scale_codebook(scales, entries=16, iterations=50)
            index_payload = pack_nibbles(fitted.indices)
            np.testing.assert_array_equal(unpack_nibbles(index_payload, scales.size), fitted.indices)
            codebook_payload = fitted.values.astype("<f2", copy=False).tobytes()
            index_rel = f"tensors/{tensor_index:04d}.idx4"
            codebook_rel = f"tensors/{tensor_index:04d}.f16"
            atomic_bytes(args.out_dir / index_rel, index_payload)
            atomic_bytes(args.out_dir / codebook_rel, codebook_payload)
            error = scales.astype(np.float64) - fitted.reconstructed.astype(np.float64)
            source_squared = float(np.dot(scales.astype(np.float64), scales.astype(np.float64)))
            error_squared = float(np.dot(error, error))
            positive = scales > 0
            relative = np.abs(error[positive]) / scales[positive] if np.any(positive) else np.zeros(1)
            nrmse = math.sqrt(error_squared / source_squared) if source_squared else 0.0
            record = {
                "tensor": name,
                "shape": source["shape"],
                "scale_count": int(scales.size),
                "source_scale_sha256": source_sha256,
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
                "metrics": {
                    "sum_source_squared": source_squared,
                    "sum_error_squared": error_squared,
                    "normalized_rmse": nrmse,
                    "maximum_absolute_error": fitted.maximum_absolute_error,
                    "p99_relative_error_nonzero_scales": float(np.quantile(relative, 0.99)),
                },
            }
            result["records"].append(record)
            atomic_bytes(partial_path, canonical_json(result))

        metrics = record["metrics"]
        total_error_squared += metrics["sum_error_squared"]
        total_source_squared += metrics["sum_source_squared"]
        total_scales += record["scale_count"]
        total_index_bytes += record["indices"]["bytes"]
        total_codebook_bytes += record["codebook"]["bytes"]
        maximum_tensor_nrmse = max(maximum_tensor_nrmse, metrics["normalized_rmse"])
        maximum_tensor_p99_relative_error = max(
            maximum_tensor_p99_relative_error,
            metrics["p99_relative_error_nonzero_scales"],
        )
        maximum_absolute_error = max(maximum_absolute_error, metrics["maximum_absolute_error"])

    if total_scales != EXPECTED_SCALES or len(result["records"]) != EXPECTED_TENSORS:
        raise RuntimeError("full mapped scale coverage is incomplete")
    global_nrmse = math.sqrt(total_error_squared / total_source_squared)
    gate = {
        "maximum_global_normalized_rmse": 0.02,
        "maximum_per_tensor_normalized_rmse": 0.05,
        "maximum_per_tensor_p99_relative_error_nonzero_scales": 0.10,
        "observed_global_normalized_rmse": global_nrmse,
        "observed_maximum_per_tensor_normalized_rmse": maximum_tensor_nrmse,
        "observed_maximum_per_tensor_p99_relative_error_nonzero_scales": maximum_tensor_p99_relative_error,
    }
    gate["passed"] = (
        global_nrmse <= gate["maximum_global_normalized_rmse"]
        and maximum_tensor_nrmse <= gate["maximum_per_tensor_normalized_rmse"]
        and maximum_tensor_p99_relative_error <= gate["maximum_per_tensor_p99_relative_error_nonzero_scales"]
    )
    result.update({
        "schema": "qwen38_27b_scale_codebook4_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "summary": {
            "tensors": len(result["records"]),
            "scales": total_scales,
            "source_fp16_scale_bytes": EXPECTED_SCALE_BYTES,
            "packed_index_bytes": total_index_bytes,
            "fp16_codebook_bytes": total_codebook_bytes,
            "combined_bytes": total_index_bytes + total_codebook_bytes,
            "reused_tensors": reused,
            "global_normalized_rmse": global_nrmse,
            "maximum_tensor_normalized_rmse": maximum_tensor_nrmse,
            "maximum_tensor_p99_relative_error_nonzero_scales": maximum_tensor_p99_relative_error,
            "maximum_absolute_error": maximum_absolute_error,
        },
        "gate": gate,
        "claim_boundary": {
            "mapped_27b_scale_reconstruction_only": True,
            "unmapped_visual_mtp_and_convolution_scales_fitted": False,
            "model_quality_proven": False,
            "artifact_exported": False,
            "scientific_breakthrough_proven": False,
        },
    })
    atomic_bytes(final_path, canonical_json(result))
    atomic_bytes(args.receipt, canonical_json(result))
    partial_path.unlink(missing_ok=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
