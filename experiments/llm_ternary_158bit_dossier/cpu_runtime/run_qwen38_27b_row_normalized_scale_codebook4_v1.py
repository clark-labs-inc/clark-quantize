#!/usr/bin/env python3
"""Fit restartable row-normalized 4-bit scale codes to mapped Qwen3.8-27B."""
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
from clark_quantize.qwen38_full_scale_codebook import (
    aggregate_scale_records,
    scale_error_metrics,
)
from clark_quantize.qwen38_row_scale_codebook import (
    fit_row_normalized_log_codebook,
    reconstruct_row_normalized_scales,
)
from clark_quantize.scale_codebook import pack_nibbles, unpack_nibbles


EXPECTED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
EXPECTED_CODE_MANIFEST_SHA256 = "c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4"
EXPECTED_TENSORS = 402
EXPECTED_SCALES = 209_920_000
EXPECTED_SCALE_BYTES = 419_840_000

AUTHORITATIVE_UNITS = (
    "quant-qwen38-27b-infrastructure-retry-v1.service",
    "quant-qwen38-27b-layer0-mlp-activation-qat-watch-v1.service",
    "quant-qwen38-27b-layer0-mlp-activation-qat-paired-watch-v1.service",
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scale_bytes(artifact: Path, records: list[dict]) -> bytes:
    content = bytearray()
    for record in records:
        descriptor = record["descriptor"]
        path = artifact / "scales" / descriptor["scale_file"]
        raw = path.read_bytes()
        if (
            len(raw) != descriptor["scale_bytes"]
            or sha256_bytes(raw) != descriptor["scale_sha256"]
        ):
            raise ValueError(f"source scale block failed authentication: {path}")
        content.extend(raw)
    return bytes(content)


def validate_output(path: Path, *, size: int, digest: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
        raise ValueError(f"authenticated row-scale output is missing or changed: {path}")


def active_authoritative_service() -> str | None:
    for unit in AUTHORITATIVE_UNITS:
        if subprocess.run(
            ["systemctl", "is-active", "--quiet", unit], check=False
        ).returncode == 0:
            return unit
    return None


def resource_guard(workspace: Path) -> None:
    active = active_authoritative_service()
    if active:
        raise SystemExit(f"infrastructure-invalid: refusing to compete with {active}")
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < 2 * 1024**3:
        raise SystemExit("infrastructure-invalid: less than 2 GiB free disk")
    fields = {
        line.split(":", 1)[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line and len(line.split()) >= 2 and line.split()[1].isdigit()
    }
    if fields.get("MemAvailable", 0) < 8 * 1024 * 1024:
        raise SystemExit("infrastructure-invalid: less than 8 GiB available memory")
    if len(Path("/proc/swaps").read_text().splitlines()) > 1:
        raise SystemExit("infrastructure-invalid: swap is active")


def load_frozen_payload(
    paths: dict[str, Path], record: dict, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for kind, path in paths.items():
        validate_output(
            path,
            size=int(record[kind]["bytes"]),
            digest=str(record[kind]["sha256"]),
        )
    indices = unpack_nibbles(paths["indices"].read_bytes(), math.prod(shape)).reshape(shape)
    anchors = np.frombuffer(paths["anchors"].read_bytes(), dtype="<f2")
    codebook = np.frombuffer(paths["codebook"].read_bytes(), dtype="<f2")
    if anchors.size != shape[0] or codebook.size != 16:
        raise ValueError("row-scale payload dimensions do not match its record")
    if sha256_bytes(indices.tobytes()) != record["unpacked_indices_sha256"]:
        raise ValueError(f"resume nibble roundtrip mismatch: {record['tensor']}")
    return anchors, codebook, indices


def metrics_equal(observed: dict[str, float], expected: dict[str, float]) -> bool:
    return observed.keys() == expected.keys() and all(
        math.isclose(float(observed[key]), float(expected[key]), rel_tol=0.0, abs_tol=0.0)
        for key in observed
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("/workspace/quant-runs/qwen38-row-normalized-scale-codebook4-v1"),
    )
    parser.add_argument(
        "--receipt", type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-row-normalized-scale-codebook4-v1.json"),
    )
    args = parser.parse_args()

    resource_guard(Path("/workspace"))

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_row_normalized_scale_codebook4_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    source_manifest_path = args.artifact / "codes.cq3d" / "manifest.json"
    if sha256_file(source_manifest_path) != EXPECTED_CODE_MANIFEST_SHA256:
        raise ValueError("source Qwen3.8-27B code manifest SHA-256 mismatch")
    source_manifest = json.loads(source_manifest_path.read_text())
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
        raise ValueError("source Qwen3.8-27B artifact identity mismatch")

    by_tensor = {source["name"]: [] for source in identity["mapped_sources"]}
    for record in source_manifest["chunks"]:
        by_tensor[record["descriptor"]["tensor"]].append(record)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = args.out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / "manifest.partial.json"
    final_path = args.out_dir / "manifest.json"
    run_identity = {
        "schema": "qwen38_27b_row_normalized_scale_codebook4_identity_v1",
        "model_revision": EXPECTED_REVISION,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "source_code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "algorithm": prereg["candidate"],
        "preregistration_sha256": sha256_file(prereg_path),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    if final_path.exists():
        complete = json.loads(final_path.read_text())
        if complete.get("identity") != run_identity or complete.get("complete") is not True:
            raise ValueError("existing row-scale manifest identity mismatch")
        if not args.receipt.exists():
            atomic_bytes(args.receipt, final_path.read_bytes())
        elif sha256_file(args.receipt) != sha256_file(final_path):
            raise ValueError("row-scale manifest and receipt diverged")
        print(json.dumps(complete, indent=2, sort_keys=True))
        if not complete.get("gate", {}).get("passed"):
            raise SystemExit(2)
        return
    if partial_path.exists():
        result = json.loads(partial_path.read_text())
        if result.get("identity") != run_identity or result.get("complete"):
            raise ValueError("partial row-scale manifest identity mismatch")
    else:
        result = {"identity": run_identity, "records": [], "complete": False}
        atomic_bytes(partial_path, canonical_json(result))

    reused = 0
    for tensor_index, source in enumerate(identity["mapped_sources"]):
        name = source["name"]
        rows = int(source["shape"][0])
        groups = math.ceil(int(source["shape"][1]) / 128)
        raw = load_scale_bytes(args.artifact, by_tensor[name])
        source_sha256 = sha256_bytes(raw)
        scales = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(rows, groups)
        prefix = f"{tensor_index:03d}-{hashlib.sha256(name.encode()).hexdigest()[:16]}"
        paths = {
            "indices": tensor_dir / f"{prefix}.indices.bin",
            "anchors": tensor_dir / f"{prefix}.anchors.f16",
            "codebook": tensor_dir / f"{prefix}.codebook.f16",
        }
        if tensor_index < len(result["records"]):
            record = result["records"][tensor_index]
            if (
                record.get("tensor") != name
                or record.get("source_scale_sha256") != source_sha256
            ):
                raise ValueError(f"resume source mismatch at tensor {tensor_index}: {name}")
            anchors, codebook, reconstructed_indices = load_frozen_payload(
                paths, record, scales.shape
            )
            rebuilt = reconstruct_row_normalized_scales(
                anchors, codebook, reconstructed_indices
            )
            resumed_metrics = scale_error_metrics(
                scales,
                type("FrozenScalePayload", (), {"reconstructed": rebuilt})(),
            )
            if not metrics_equal(resumed_metrics, record["metrics"]):
                raise ValueError(f"resume reconstruction metrics mismatch: {name}")
            reused += 1
            continue
        if tensor_index != len(result["records"]):
            raise ValueError("partial record sequence is not contiguous")

        fitted = fit_row_normalized_log_codebook(
            scales,
            entries=16,
            iterations=int(prereg["candidate"]["iterations"]),
            chunk_rows=int(prereg["candidate"]["chunk_rows"]),
        )
        packed = pack_nibbles(fitted.indices)
        restored = unpack_nibbles(packed, fitted.indices.size).reshape(fitted.indices.shape)
        if not np.array_equal(restored, fitted.indices):
            raise AssertionError(f"nibble pack roundtrip failed: {name}")
        contents = {
            "indices": packed,
            "anchors": fitted.anchors.astype("<f2", copy=False).tobytes(),
            "codebook": fitted.values.astype("<f2", copy=False).tobytes(),
        }
        for kind, content in contents.items():
            atomic_bytes(paths[kind], content)
        metrics = scale_error_metrics(scales, fitted)
        record = {
            "tensor": name,
            "shape": source["shape"],
            "rows": rows,
            "groups_per_row": groups,
            "scale_count": scales.size,
            "source_scale_sha256": source_sha256,
            "unpacked_indices_sha256": sha256_bytes(fitted.indices.tobytes()),
            "metrics": metrics,
        }
        for kind, path in paths.items():
            record[kind] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        result["records"].append(record)
        atomic_bytes(partial_path, canonical_json(result))
        print(
            json.dumps(
                {
                    "tensor_index": tensor_index,
                    "tensor": name,
                    "normalized_rmse": metrics["normalized_rmse"],
                    "p99_relative_error": metrics[
                        "p99_relative_error_nonzero_scales"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    totals = aggregate_scale_records(result["records"])
    if totals.tensors != EXPECTED_TENSORS or totals.scales != EXPECTED_SCALES:
        raise ValueError("row-normalized mapped scale coverage is incomplete")
    index_bytes = sum(item["indices"]["bytes"] for item in result["records"])
    anchor_bytes = sum(item["anchors"]["bytes"] for item in result["records"])
    codebook_bytes = sum(item["codebook"]["bytes"] for item in result["records"])
    combined_bytes = index_bytes + anchor_bytes + codebook_bytes
    reconstruction = prereg["reconstruction_gate"]
    storage = prereg["storage_gate"]
    gate = {
        **reconstruction,
        "maximum_combined_bytes_for_mapped_402_tensors": storage[
            "maximum_combined_bytes_for_mapped_402_tensors"
        ],
        "observed_global_normalized_rmse": totals.global_nrmse,
        "observed_maximum_per_tensor_normalized_rmse": totals.maximum_tensor_nrmse,
        "observed_maximum_per_tensor_p99_relative_error_nonzero_scales": (
            totals.maximum_tensor_p99_relative_error
        ),
        "observed_combined_bytes": combined_bytes,
    }
    gate["passed"] = (
        totals.global_nrmse <= reconstruction["maximum_global_normalized_rmse"]
        and totals.maximum_tensor_nrmse
        <= reconstruction["maximum_per_tensor_normalized_rmse"]
        and totals.maximum_tensor_p99_relative_error
        <= reconstruction["maximum_per_tensor_p99_relative_error_nonzero_scales"]
        and combined_bytes <= storage["maximum_combined_bytes_for_mapped_402_tensors"]
    )
    result.update(
        {
            "schema": "qwen38_27b_row_normalized_scale_codebook4_receipt_v1",
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "complete": True,
            "summary": {
                "tensors": totals.tensors,
                "scales": totals.scales,
                "source_fp16_scale_bytes": EXPECTED_SCALE_BYTES,
                "packed_index_bytes": index_bytes,
                "fp16_row_anchor_bytes": anchor_bytes,
                "fp16_codebook_bytes": codebook_bytes,
                "combined_bytes": combined_bytes,
                "reused_tensors": reused,
                "global_normalized_rmse": totals.global_nrmse,
                "maximum_tensor_normalized_rmse": totals.maximum_tensor_nrmse,
                "maximum_tensor_p99_relative_error_nonzero_scales": (
                    totals.maximum_tensor_p99_relative_error
                ),
            },
            "gate": gate,
            "claim_boundary": {
                "mapped_27b_scale_reconstruction_only": True,
                "remaining_264_tensors_fitted": False,
                "model_quality_proven": False,
                "artifact_exported": False,
                "scientific_breakthrough_proven": False,
            },
        }
    )
    payload = canonical_json(result)
    atomic_bytes(final_path, payload)
    atomic_bytes(args.receipt, payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
