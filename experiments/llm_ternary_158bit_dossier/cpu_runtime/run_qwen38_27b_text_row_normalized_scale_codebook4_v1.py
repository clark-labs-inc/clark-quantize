#!/usr/bin/env python3
"""Fit every remaining row-normalized scale in the Qwen3.8-27B text graph."""
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
    scale_error_metrics,
)
from clark_quantize.qwen38_row_scale_codebook import (
    fit_row_normalized_log_codebook,
    reconstruct_row_normalized_scales,
)
from clark_quantize.qwen38_row_scale_storage import row_normalized_scale_ledger
from clark_quantize.qwen38_storage_budget import (
    StorageStrategy,
    canonical_inventory_digest,
    inventory_from_safetensors,
    strategy_ledger,
)
from clark_quantize.qwen38_text_scope import qwen38_text_inference_tensors
from clark_quantize.scale_codebook import pack_nibbles, unpack_nibbles
from clark_quantize.ternary import ternary_codes_scales


MODEL_ID = "Qwen/Qwen3.8-27B"
EXPECTED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
EXPECTED_CHECKPOINT_INVENTORY_SHA256 = "8533a0edfba093faf2a865391405408461613b92ef83e8d511436f2ed96bc14d"
EXPECTED_TEXT_INVENTORY_SHA256 = "23a5c5137b7893b6d551fcc0085c4a994bf7ae158938d57974d32c7ed856361f"
EXPECTED_CODE_MANIFEST_SHA256 = "c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4"
EXPECTED_STORAGE_RECEIPT_SHA256 = "1e37e00ff6736e6614d77f7dd1a1f69aecae96503e1659b0d027df9a37caec92"
EXPECTED_TEXT_TENSORS = 851
EXPECTED_TEXT_PARAMETERS = 26_895_998_464
EXPECTED_TOTAL_TENSORS = 546
EXPECTED_TOTAL_PARAMETERS = 26_895_319_040
EXPECTED_TOTAL_SCALES = 210_595_840
EXPECTED_TOTAL_ROWS = 4_892_160
EXPECTED_RETAINED_TENSORS = 305
EXPECTED_RETAINED_PARAMETERS = 679_424
EXPECTED_MAPPED_TENSORS = 402
EXPECTED_MAPPED_PARAMETERS = 26_869_760_000
EXPECTED_MAPPED_SCALES = 209_920_000
EXPECTED_MAPPED_ROWS = 4_396_032
EXPECTED_SUPPLEMENTAL_TENSORS = 144
EXPECTED_SUPPLEMENTAL_PARAMETERS = 25_559_040
EXPECTED_SUPPLEMENTAL_SCALES = 675_840
EXPECTED_SUPPLEMENTAL_ROWS = 496_128
EXPECTED_COMPLETE_BUNDLE_BYTES = 5_471_531_964

AUTHORITATIVE_UNITS = (
    "quant-qwen38-27b-infrastructure-retry-v1.service",
    "quant-qwen38-27b-layer0-mlp-activation-qat-watch-v1.service",
    "quant-qwen38-27b-layer0-mlp-activation-qat-paired-watch-v1.service",
    "quant-qwen38-27b-row-normalized-scale-codebook4-watch-v1.service",
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resource_guard(workspace: Path) -> None:
    for unit in AUTHORITATIVE_UNITS:
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


def output_path(root: Path, descriptor: dict) -> Path:
    path = Path(descriptor["path"])
    return path if path.is_absolute() else root / path


def validate_output(root: Path, descriptor: dict) -> Path:
    path = output_path(root, descriptor)
    if (
        not path.is_file()
        or path.stat().st_size != int(descriptor["bytes"])
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise RuntimeError(f"authenticated row-scale output is missing or changed: {path}")
    return path


def load_payload(
    root: Path, record: dict, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_path = validate_output(root, record["indices"])
    anchor_path = validate_output(root, record["anchors"])
    codebook_path = validate_output(root, record["codebook"])
    indices = unpack_nibbles(index_path.read_bytes(), math.prod(shape)).reshape(shape)
    anchors = np.frombuffer(anchor_path.read_bytes(), dtype="<f2")
    codebook = np.frombuffer(codebook_path.read_bytes(), dtype="<f2")
    if anchors.size != shape[0] or codebook.size != 16:
        raise RuntimeError("row-normalized payload dimensions changed")
    if sha256_bytes(indices.tobytes()) != record["unpacked_indices_sha256"]:
        raise RuntimeError("row-normalized nibble roundtrip digest changed")
    return anchors, codebook, indices


def metrics_equal(observed: dict, expected: dict) -> bool:
    return observed.keys() == expected.keys() and all(
        math.isclose(float(observed[key]), float(expected[key]), rel_tol=0.0, abs_tol=0.0)
        for key in observed
    )


def quantize_source_scales(
    handle, tensor, *, row_batch: int
) -> tuple[np.ndarray, str, str]:
    scale_parts: list[np.ndarray] = []
    code_digest = hashlib.sha256()
    scale_digest = hashlib.sha256()
    groups = math.ceil(tensor.columns / 128)
    view = handle.get_slice(tensor.name)
    with torch.no_grad():
        for row_start in range(0, tensor.rows, row_batch):
            row_stop = min(tensor.rows, row_start + row_batch)
            source_rows = view[row_start:row_stop].contiguous().reshape(
                row_stop - row_start, tensor.columns
            ).float()
            codes, scales = ternary_codes_scales(
                source_rows, group=128, method="optimal"
            )
            code_payload = codes.numpy().astype(np.int8, copy=False).tobytes()
            scale_array = scales.numpy().astype("<f2", copy=False).reshape(
                row_stop - row_start, groups
            )
            scale_payload = scale_array.tobytes()
            code_digest.update(code_payload)
            scale_digest.update(scale_payload)
            scale_parts.append(scale_array.copy())
            del source_rows, codes, scales
    source_scales = np.concatenate(scale_parts, axis=0).astype(np.float32)
    if source_scales.shape != (tensor.rows, groups):
        raise RuntimeError(f"generated scale shape mismatch for {tensor.name}")
    return source_scales, code_digest.hexdigest(), scale_digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--mapped-artifact",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"
        ),
    )
    parser.add_argument(
        "--mapped-codebooks",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-row-normalized-scale-codebook4-v1"
        ),
    )
    parser.add_argument(
        "--mapped-watch-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-row-normalized-scale-codebook4-watch-v1/receipt.json"
        ),
    )
    parser.add_argument(
        "--scope-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-text-graph-scope-v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-text-row-normalized-scale-codebook4-v1"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-text-row-normalized-scale-codebook4-v1.json"
        ),
    )
    parser.add_argument("--row-batch", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.row_batch <= 0 or args.threads <= 0:
        raise ValueError("row batch and thread count must be positive")
    resource_guard(Path("/workspace"))
    torch.set_num_threads(args.threads)

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / (
        "qwen38_27b_text_row_normalized_scale_codebook4_preregistration_v1.json"
    )
    scope_path = runtime / "qwen38_27b_text_graph_scope_amendment_v1.json"
    prereg = json.loads(prereg_path.read_text())
    scope_amendment = json.loads(scope_path.read_text())
    if (
        scope_amendment.get("selected_text_graph_scope", {}).get("parameters")
        != EXPECTED_TEXT_PARAMETERS
    ):
        raise RuntimeError("Qwen3.8 text-graph scope amendment changed")
    scope_receipt = json.loads(args.scope_receipt.read_text())
    if (
        scope_receipt.get("schema") != "qwen38_27b_text_graph_scope_receipt_v1"
        or scope_receipt.get("model") != MODEL_ID
        or scope_receipt.get("revision") != EXPECTED_REVISION
        or scope_receipt.get("text_inventory_sha256") != EXPECTED_TEXT_INVENTORY_SHA256
        or scope_receipt.get("scopes", {}).get("text", {}).get("parameters")
        != EXPECTED_TEXT_PARAMETERS
        or scope_receipt.get("valid") is not True
        or scope_receipt.get("complete") is not True
        or scope_receipt.get("terminal") is not True
    ):
        raise RuntimeError("Qwen3.8 text-graph scope receipt is invalid")
    storage_path = runtime / "receipts/qwen38-27b-storage-budget-v1.json"
    if sha256_file(storage_path) != EXPECTED_STORAGE_RECEIPT_SHA256:
        raise RuntimeError("frozen complete-model storage receipt changed")
    storage_receipt = json.loads(storage_path.read_text())

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

    code_manifest_path = args.mapped_artifact / "codes.cq3d/manifest.json"
    if sha256_file(code_manifest_path) != EXPECTED_CODE_MANIFEST_SHA256:
        raise RuntimeError("mapped Qwen3.8-27B ternary source manifest changed")
    code_manifest = json.loads(code_manifest_path.read_text())
    mapped_sources = code_manifest.get("identity", {}).get("mapped_sources", [])
    mapped_names = {record["name"] for record in mapped_sources}
    if (
        len(mapped_names) != EXPECTED_MAPPED_TENSORS
        or sum(int(record["shape"][0]) for record in mapped_sources)
        != EXPECTED_MAPPED_ROWS
    ):
        raise RuntimeError("mapped ternary tensor coverage changed")

    mapped_manifest_path = args.mapped_codebooks / "manifest.json"
    mapped_manifest = json.loads(mapped_manifest_path.read_text())
    mapped_manifest_sha256 = sha256_file(mapped_manifest_path)
    mapped_watch = json.loads(args.mapped_watch_receipt.read_text())
    if (
        mapped_watch.get("schema")
        != "qwen38_27b_row_normalized_scale_codebook4_watch_receipt_v1"
        or mapped_watch.get("terminal") is not True
        or mapped_watch.get("static_reconstruction_pass") is not True
        or mapped_watch.get("result_receipt_sha256") != mapped_manifest_sha256
    ):
        raise RuntimeError("mapped row-scale watch receipt does not authenticate the parent")
    if (
        mapped_manifest.get("schema")
        != "qwen38_27b_row_normalized_scale_codebook4_receipt_v1"
        or mapped_manifest.get("complete") is not True
        or mapped_manifest.get("gate", {}).get("passed") is not True
    ):
        raise SystemExit("valid-nonpromotion: mapped row-scale gate did not pass")
    mapped_records = mapped_manifest.get("records", [])
    if (
        len(mapped_records) != EXPECTED_MAPPED_TENSORS
        or {record["tensor"] for record in mapped_records} != mapped_names
    ):
        raise RuntimeError("mapped row-scale record coverage changed")
    mapped_totals = aggregate_scale_records(mapped_records)
    if mapped_totals.scales != EXPECTED_MAPPED_SCALES:
        raise RuntimeError("mapped row-scale denominator changed")
    for record in mapped_records:
        rows = int(record["rows"])
        groups = int(record["groups_per_row"])
        load_payload(args.mapped_codebooks, record, (rows, groups))

    checkpoint_tensors, source = inventory_from_safetensors(args.model_dir)
    if (
        source["index_sha256"] != EXPECTED_INDEX_SHA256
        or source["inventory_sha256"] != EXPECTED_CHECKPOINT_INVENTORY_SHA256
    ):
        raise RuntimeError("complete Qwen3.8-27B tensor inventory changed")
    if (
        source["auxiliary_bytes"] != storage_receipt["source"]["auxiliary_bytes"]
        or source["auxiliary_files"] != storage_receipt["source"]["auxiliary_files"]
    ):
        raise RuntimeError("Qwen3.8 auxiliary file inventory changed")
    tensors = qwen38_text_inference_tensors(checkpoint_tensors)
    if (
        len(tensors) != EXPECTED_TEXT_TENSORS
        or sum(tensor.parameters for tensor in tensors) != EXPECTED_TEXT_PARAMETERS
        or canonical_inventory_digest(tensors) != EXPECTED_TEXT_INVENTORY_SHA256
    ):
        raise RuntimeError("Qwen3.8 deployable text-graph inventory changed")
    quantized = sorted(
        (tensor for tensor in tensors if tensor.quantized), key=lambda item: item.name
    )
    retained = [tensor for tensor in tensors if not tensor.quantized]
    if (
        len(quantized) != EXPECTED_TOTAL_TENSORS
        or sum(tensor.parameters for tensor in quantized) != EXPECTED_TOTAL_PARAMETERS
        or sum(tensor.rows for tensor in quantized) != EXPECTED_TOTAL_ROWS
        or len(retained) != EXPECTED_RETAINED_TENSORS
        or sum(tensor.parameters for tensor in retained) != EXPECTED_RETAINED_PARAMETERS
    ):
        raise RuntimeError("complete rank-2+ Qwen3.8-27B text coverage changed")
    if not mapped_names <= {tensor.name for tensor in quantized}:
        raise RuntimeError("mapped ternary parent includes a non-text tensor")
    supplemental = [tensor for tensor in quantized if tensor.name not in mapped_names]
    if (
        len(supplemental) != EXPECTED_SUPPLEMENTAL_TENSORS
        or sum(tensor.parameters for tensor in supplemental)
        != EXPECTED_SUPPLEMENTAL_PARAMETERS
        or sum(tensor.rows for tensor in supplemental) != EXPECTED_SUPPLEMENTAL_ROWS
        or sum(
            tensor.rows * math.ceil(tensor.columns / 128) for tensor in supplemental
        )
        != EXPECTED_SUPPLEMENTAL_SCALES
    ):
        raise RuntimeError("supplemental Qwen3.8-27B tensor coverage changed")

    base_storage = strategy_ledger(
        tensors,
        StorageStrategy(
            "base3_group128_codebook4", "base3", 128, 4, True
        ),
        source["auxiliary_bytes"],
    )
    storage = row_normalized_scale_ledger(tensors, base_storage)
    if (
        storage["scales"]["count"] != EXPECTED_TOTAL_SCALES
        or storage["scales"]["fp16_row_anchor_count"] != EXPECTED_TOTAL_ROWS
        or storage["bytes"]["complete_bundle"] != EXPECTED_COMPLETE_BUNDLE_BYTES
    ):
        raise RuntimeError("text row-normalized storage denominator changed")

    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    ordered = sorted(supplemental, key=lambda item: (weight_map[item.name], item.name))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.out_dir / "supplemental"
    output_root.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / "manifest.partial.json"
    final_path = args.out_dir / "manifest.json"
    run_identity = {
        "schema": "qwen38_27b_text_row_normalized_scale_codebook4_identity_v1",
        "model_id": MODEL_ID,
        "model_revision": EXPECTED_REVISION,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "checkpoint_inventory_sha256": EXPECTED_CHECKPOINT_INVENTORY_SHA256,
        "text_graph_inventory_sha256": EXPECTED_TEXT_INVENTORY_SHA256,
        "text_graph_parameters": EXPECTED_TEXT_PARAMETERS,
        "scope_amendment_sha256": sha256_file(scope_path),
        "scope_receipt_sha256": sha256_file(args.scope_receipt),
        "mapped_code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "mapped_row_scale_manifest_sha256": mapped_manifest_sha256,
        "mapped_row_scale_watch_receipt_sha256": sha256_file(
            args.mapped_watch_receipt
        ),
        "storage_receipt_sha256": EXPECTED_STORAGE_RECEIPT_SHA256,
        "preregistration_sha256": sha256_file(prereg_path),
        "algorithm": prereg["supplemental_fit"],
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    if final_path.exists():
        result = json.loads(final_path.read_text())
        if result.get("identity") != run_identity or result.get("complete") is not True:
            raise RuntimeError("existing full row-scale manifest identity mismatch")
        for record in result["supplemental_records"]:
            load_payload(
                args.out_dir,
                record,
                (int(record["rows"]), int(record["groups_per_row"])),
            )
        if not args.receipt.exists():
            atomic_bytes(args.receipt, final_path.read_bytes())
        elif sha256_file(args.receipt) != sha256_file(final_path):
            raise RuntimeError("full row-scale manifest and receipt diverged")
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result.get("gate", {}).get("passed"):
            raise SystemExit(2)
        return
    if partial_path.exists():
        result = json.loads(partial_path.read_text())
        if result.get("identity") != run_identity or result.get("complete") is not False:
            raise RuntimeError("partial full row-scale manifest identity mismatch")
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
                source_scales, code_sha256, scale_sha256 = quantize_source_scales(
                    handle, tensor, row_batch=args.row_batch
                )
                shape = source_scales.shape
                prefix = f"{supplemental_index:04d}-{hashlib.sha256(name.encode()).hexdigest()[:16]}"
                paths = {
                    "indices": output_root / f"{prefix}.indices.bin",
                    "anchors": output_root / f"{prefix}.anchors.f16",
                    "codebook": output_root / f"{prefix}.codebook.f16",
                }
                if supplemental_index < len(result["supplemental_records"]):
                    record = result["supplemental_records"][supplemental_index]
                    if (
                        record.get("tensor") != name
                        or record.get("shape") != list(tensor.shape)
                        or record.get("source_shard") != shard
                        or record.get("source_ternary_codes_sha256") != code_sha256
                        or record.get("source_scale_sha256") != scale_sha256
                    ):
                        raise RuntimeError(
                            f"resume source identity mismatch at {supplemental_index}"
                        )
                    anchors, codebook, indices = load_payload(
                        args.out_dir, record, shape
                    )
                    rebuilt = reconstruct_row_normalized_scales(
                        anchors, codebook, indices
                    )
                    observed = scale_error_metrics(
                        source_scales,
                        type("FrozenScalePayload", (), {"reconstructed": rebuilt})(),
                    )
                    if not metrics_equal(observed, record["metrics"]):
                        raise RuntimeError(f"resume metrics changed for {name}")
                    supplemental_index += 1
                    continue
                if supplemental_index != len(result["supplemental_records"]):
                    raise RuntimeError("supplemental record prefix is not contiguous")

                fitted = fit_row_normalized_log_codebook(
                    source_scales,
                    entries=16,
                    iterations=int(prereg["supplemental_fit"]["iterations"]),
                    chunk_rows=256,
                )
                packed = pack_nibbles(fitted.indices)
                restored = unpack_nibbles(packed, fitted.indices.size).reshape(shape)
                if not np.array_equal(restored, fitted.indices):
                    raise AssertionError(f"nibble pack roundtrip failed: {name}")
                contents = {
                    "indices": packed,
                    "anchors": fitted.anchors.astype("<f2", copy=False).tobytes(),
                    "codebook": fitted.values.astype("<f2", copy=False).tobytes(),
                }
                for kind, content in contents.items():
                    atomic_bytes(paths[kind], content)
                record = {
                    "tensor": name,
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype,
                    "parameters": tensor.parameters,
                    "source_shard": shard,
                    "rows": tensor.rows,
                    "groups_per_row": shape[1],
                    "scale_count": int(source_scales.size),
                    "source_ternary_codes_sha256": code_sha256,
                    "source_scale_sha256": scale_sha256,
                    "unpacked_indices_sha256": sha256_bytes(fitted.indices.tobytes()),
                    "metrics": scale_error_metrics(source_scales, fitted),
                }
                for kind, path in paths.items():
                    record[kind] = {
                        "path": str(path.relative_to(args.out_dir)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                result["supplemental_records"].append(record)
                atomic_bytes(partial_path, canonical_json(result))
                supplemental_index += 1
                print(
                    json.dumps(
                        {
                            "supplemental_index": supplemental_index - 1,
                            "tensor": name,
                            "normalized_rmse": record["metrics"]["normalized_rmse"],
                            "p99_relative_error": record["metrics"][
                                "p99_relative_error_nonzero_scales"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    supplemental_records = result["supplemental_records"]
    if len(supplemental_records) != EXPECTED_SUPPLEMENTAL_TENSORS:
        raise RuntimeError("supplemental row-scale coverage is incomplete")
    all_records = mapped_records + supplemental_records
    if {record["tensor"] for record in all_records} != {
        tensor.name for tensor in quantized
    }:
        raise RuntimeError("full row-scale tensor set differs from text-graph inventory")
    totals = aggregate_scale_records(all_records)
    if totals.tensors != EXPECTED_TOTAL_TENSORS or totals.scales != EXPECTED_TOTAL_SCALES:
        raise RuntimeError("full row-scale denominator is incomplete")

    actual_index_bytes = sum(record["indices"]["bytes"] for record in all_records)
    actual_anchor_bytes = sum(record["anchors"]["bytes"] for record in all_records)
    actual_codebook_bytes = sum(record["codebook"]["bytes"] for record in all_records)
    if (
        actual_index_bytes != storage["scales"]["index_bytes"]
        or actual_anchor_bytes != storage["scales"]["fp16_row_anchor_bytes"]
        or actual_codebook_bytes != storage["scales"]["fp16_codebook_bytes"]
    ):
        raise RuntimeError("actual row-scale payload bytes differ from complete ledger")

    reconstruction = prereg["reconstruction_gate"]
    storage_gate = prereg["storage_gate"]
    gate = {
        **reconstruction,
        "maximum_complete_bundle_bpw": storage_gate["maximum_complete_bundle_bpw"],
        "observed_global_normalized_rmse": totals.global_nrmse,
        "observed_maximum_per_tensor_normalized_rmse": totals.maximum_tensor_nrmse,
        "observed_maximum_per_tensor_p99_relative_error_nonzero_scales": (
            totals.maximum_tensor_p99_relative_error
        ),
        "observed_complete_bundle_bpw": storage[
            "effective_bits_per_total_parameter"
        ],
    }
    gate["passed"] = (
        totals.global_nrmse <= reconstruction["maximum_global_normalized_rmse"]
        and totals.maximum_tensor_nrmse
        <= reconstruction["maximum_per_tensor_normalized_rmse"]
        and totals.maximum_tensor_p99_relative_error
        <= reconstruction["maximum_per_tensor_p99_relative_error_nonzero_scales"]
        and storage["effective_bits_per_total_parameter"]
        <= storage_gate["maximum_complete_bundle_bpw"]
    )
    result.update(
        {
            "schema": "qwen38_27b_text_row_normalized_scale_codebook4_receipt_v1",
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "complete": True,
            "scope_parent": {
                "receipt": str(args.scope_receipt),
                "sha256": sha256_file(args.scope_receipt),
                "text_graph_parameters": EXPECTED_TEXT_PARAMETERS,
            },
            "mapped_parent": {
                "manifest": str(mapped_manifest_path),
                "sha256": mapped_manifest_sha256,
                "watch_receipt": str(args.mapped_watch_receipt),
                "watch_receipt_sha256": sha256_file(args.mapped_watch_receipt),
                "tensors": EXPECTED_MAPPED_TENSORS,
                "parameters": EXPECTED_MAPPED_PARAMETERS,
                "scales": EXPECTED_MAPPED_SCALES,
                "rows": EXPECTED_MAPPED_ROWS,
            },
            "summary": {
                "text_graph_tensors": EXPECTED_TEXT_TENSORS,
                "text_graph_parameters": EXPECTED_TEXT_PARAMETERS,
                "tensors": totals.tensors,
                "ternary_parameters": EXPECTED_TOTAL_PARAMETERS,
                "scales": totals.scales,
                "rows": EXPECTED_TOTAL_ROWS,
                "supplemental_tensors": EXPECTED_SUPPLEMENTAL_TENSORS,
                "supplemental_parameters": EXPECTED_SUPPLEMENTAL_PARAMETERS,
                "supplemental_scales": EXPECTED_SUPPLEMENTAL_SCALES,
                "supplemental_rows": EXPECTED_SUPPLEMENTAL_ROWS,
                "packed_index_bytes": actual_index_bytes,
                "fp16_row_anchor_bytes": actual_anchor_bytes,
                "fp16_codebook_bytes": actual_codebook_bytes,
                "global_normalized_rmse": totals.global_nrmse,
                "maximum_tensor_normalized_rmse": totals.maximum_tensor_nrmse,
                "maximum_tensor_p99_relative_error_nonzero_scales": (
                    totals.maximum_tensor_p99_relative_error
                ),
            },
            "storage_ledger": storage,
            "gate": gate,
            "claim_boundary": {
                "all_546_rank_ge_2_text_tensors_covered": True,
                "all_210595840_text_group_scales_covered": True,
                "all_4892160_text_row_anchors_counted": True,
                "vision_graph_included": False,
                "mtp_nextn_graph_included": False,
                "scope_matches_llamacpp_no_nextn_controls": True,
                "model_quality_proven": False,
                "artifact_exported": False,
                "scientific_breakthrough_proven": False,
            },
        }
    )
    payload = canonical_json(result)
    atomic_bytes(final_path, payload)
    atomic_bytes(args.receipt, payload)
    partial_path.unlink(missing_ok=True)
    marker = args.out_dir / ("PROMOTION_PASS" if gate["passed"] else "PROMOTION_REJECTED")
    atomic_bytes(marker, ("pass\n" if gate["passed"] else "rejected\n").encode())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
