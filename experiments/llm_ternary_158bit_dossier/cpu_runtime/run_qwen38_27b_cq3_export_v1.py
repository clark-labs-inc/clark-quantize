#!/usr/bin/env python3
"""Export and independently verify the complete Qwen3.8-27B CQ3 container."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from itertools import repeat

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_cq3_container import (
    Qwen38CQ3Reader,
    Qwen38CQ3Writer,
)
from clark_quantize.qwen38_cq3_export import (
    atomic_code_spool,
    exact_tensor_bytes,
    pack_verified_rows,
    sha256_file,
    transcode_mapped_record,
)
from clark_quantize.qwen38_storage_budget import (
    DTYPE_BITS,
    inventory_from_safetensors,
)
from clark_quantize.ternary import ternary_codes_scales
from clark_quantize.ternary_cpu import cpu_features, native_available


MODEL_ID = "Qwen/Qwen3.8-27B"
EXPECTED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
EXPECTED_INVENTORY_SHA256 = "8533a0edfba093faf2a865391405408461613b92ef83e8d511436f2ed96bc14d"
EXPECTED_CODE_MANIFEST_SHA256 = "c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4"
EXPECTED_STORAGE_RECEIPT_SHA256 = "1e37e00ff6736e6614d77f7dd1a1f69aecae96503e1659b0d027df9a37caec92"
EXPECTED_TENSORS = 1_199
EXPECTED_PARAMETERS = 27_781_427_952
EXPECTED_QUANTIZED_TENSORS = 666
EXPECTED_QUANTIZED_PARAMETERS = 27_780_313_088
EXPECTED_RETAINED_TENSORS = 533
EXPECTED_RETAINED_PARAMETERS = 1_114_864
EXPECTED_SCALES = 217_521_520
EXPECTED_CODE_BYTES = 5_507_270_842
EXPECTED_SCALE_BYTES = 108_782_072
EXPECTED_ROW_SCALE_INDEX_BYTES = 108_760_760
EXPECTED_ROW_SCALE_ANCHOR_BYTES = 10_483_168
EXPECTED_ROW_SCALE_CODEBOOK_BYTES = 21_312
EXPECTED_ROW_SCALE_BYTES = 119_265_240
EXPECTED_RETAINED_BYTES = 2_229_728
EXPECTED_AUXILIARY_BYTES = 23_108_348
EXPECTED_PROJECTED_BYTES = 5_641_581_820
MAXIMUM_COMPLETE_BPW = 1.63
MAXIMUM_ROW_NORMALIZED_COMPLETE_BPW = 1.64
ROW_BATCH = 128


def resource_guard(workspace: Path, *, assembly: bool = False) -> None:
    competitors = (
        "quant-qwen38-27b-cpu-q4-baseline-v1.service",
        "quant-qwen38-27b-postbaseline-chain-v1.service",
        "quant-qwen38-27b-paired-throughput-watch-v1.service",
        "quant-qwen38-27b-cq3-x86-pair-watch-v1.service",
        "quant-qwen38-27b-scale-codebook4-watch-v1.service",
        "quant-qwen38-27b-block203-codec-watch-v2.service",
        "quant-qwen38-27b-block203-radix243-watch-v1.service",
        "quant-qwen38-27b-full-scale-codebook4-watch-v1.service",
        "quant-qwen38-27b-infrastructure-retry-v1.service",
        "quant-qwen38-27b-layer0-mlp-activation-qat-watch-v1.service",
        "quant-qwen38-27b-layer0-mlp-activation-qat-paired-watch-v1.service",
        "quant-qwen38-27b-row-normalized-scale-codebook4-watch-v1.service",
        "quant-qwen38-27b-full-row-normalized-scale-codebook4-watch-v1.service",
        "quant-qwen38-27b-row-scale4-kernel-watch-v1.service",
        "quant-qwen38-27b-block203-row-scale4-direct-gemv-watch-v1.service",
        "quant-qwen38-27b-block203-row-scale4-constdiv-watch-v1.service",
        "quant-qwen38-27b-block203-row-scale4-chunk16-watch-v1.service",
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
    free = stat.f_bavail * stat.f_frsize
    minimum = EXPECTED_PROJECTED_BYTES + 1024**3 if assembly else 14 * 1024**3
    if free < minimum:
        raise SystemExit(
            f"infrastructure-invalid: {free} free bytes is below the {minimum}-byte export guard"
        )
    if len(Path("/proc/swaps").read_text().splitlines()) > 1:
        raise SystemExit("infrastructure-invalid: swap is active")


def validate_descriptor(root: Path, descriptor: dict) -> Path:
    path = root / descriptor["path"]
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise RuntimeError(f"authenticated export source is missing or changed: {path}")
    return path


def validate_spool_record(root: Path, record: dict) -> None:
    if record["encoding"] in {
        "base3_group128_codebook4",
        "base3_group128_row_normalized_codebook4",
    }:
        validate_descriptor(root, record["codes"])
    elif record["encoding"].startswith("retained_exact_"):
        validate_descriptor(root, record["retained"])
    else:
        raise RuntimeError(f"unknown spool encoding for {record.get('name')}")


def scale_sources(
    name: str,
    mapped: dict[str, dict],
    supplemental: dict[str, dict],
    mapped_root: Path,
    full_root: Path,
    *,
    row_normalized: bool,
) -> tuple[dict, tuple[Path, ...]]:
    if name in mapped:
        record = mapped[name]
        root = mapped_root
        source = "mapped_402_parent"
    elif name in supplemental:
        record = supplemental[name]
        root = full_root
        source = "supplemental_264"
    else:
        raise RuntimeError(f"no scale-codebook record for {name}")
    indices = validate_descriptor(root, record["indices"])
    codebook = validate_descriptor(root, record["codebook"])
    if record["codebook"]["bytes"] != 32:
        raise RuntimeError(f"Qwen3.8 scale codebook for {name} is not 16 FP16 values")
    result = {
        "source": source,
        "scale_count": record["scale_count"],
        "source_scale_sha256": record["source_scale_sha256"],
        "indices": record["indices"],
        "codebook": record["codebook"],
    }
    paths = [indices]
    if row_normalized:
        anchors = validate_descriptor(root, record["anchors"])
        if record["anchors"]["bytes"] != int(record["rows"]) * 2:
            raise RuntimeError(f"Qwen3.8 row anchors for {name} have the wrong size")
        result["anchors"] = record["anchors"]
        paths.append(anchors)
    paths.append(codebook)
    return result, tuple(paths)


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
        "--full-codebooks",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-full-scale-codebook4-v1"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-cq3-container-v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-cq3-container-v1/Qwen3.8-27B.cq3"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-cq3-export-v1.json"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--row-normalized",
        action="store_true",
        help="export one FP16 row anchor plus per-tensor FP16 ratio codebook",
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.threads <= 0:
        raise ValueError("worker and thread counts must be positive")
    resource_guard(Path("/workspace"))
    if not native_available():
        raise SystemExit("infrastructure-invalid: exact native block-203 codec is unavailable")
    torch.set_num_threads(args.threads)

    runtime_dir = Path(__file__).resolve().parent
    row_normalized = bool(args.row_normalized)
    prereg_path = runtime_dir / (
        "qwen38_27b_cq3_row_normalized_export_preregistration_v1.json"
        if row_normalized
        else "qwen38_27b_cq3_export_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    for item in prereg["frozen_inputs"].values():
        path = REPO / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"frozen export input hash mismatch: {path}")

    storage_path = runtime_dir / "receipts" / "qwen38-27b-storage-budget-v1.json"
    if sha256_file(storage_path) != EXPECTED_STORAGE_RECEIPT_SHA256:
        raise RuntimeError("Qwen3.8-27B storage projection receipt changed")
    storage = json.loads(storage_path.read_text())
    candidate = storage["strategies"]["base3_group128_codebook4"]
    if (
        candidate["bytes"]["ternary_code_payload"] != EXPECTED_CODE_BYTES
        or candidate["bytes"]["scale_payload_and_codebooks"] != EXPECTED_SCALE_BYTES
        or candidate["bytes"]["retained_exact_payload"] != EXPECTED_RETAINED_BYTES
        or candidate["bytes"]["auxiliary_model_files"] != EXPECTED_AUXILIARY_BYTES
        or candidate["bytes"]["complete_bundle"] != EXPECTED_PROJECTED_BYTES
    ):
        raise RuntimeError("Qwen3.8-27B projected byte ledger changed")

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
        raise RuntimeError("mapped Qwen3.8-27B code manifest changed")
    code_manifest = json.loads(code_manifest_path.read_text())
    mapped_sources = code_manifest["identity"]["mapped_sources"]
    mapped_names = {record["name"] for record in mapped_sources}
    records_by_tensor = {name: [] for name in mapped_names}
    for record in code_manifest["chunks"]:
        records_by_tensor[record["descriptor"]["tensor"]].append(record)

    mapped_manifest_path = args.mapped_codebooks / "manifest.json"
    mapped_manifest = json.loads(mapped_manifest_path.read_text())
    expected_mapped_schema = (
        "qwen38_27b_row_normalized_scale_codebook4_receipt_v1"
        if row_normalized
        else "qwen38_27b_scale_codebook4_receipt_v1"
    )
    if (
        mapped_manifest.get("schema") != expected_mapped_schema
        or mapped_manifest.get("complete") is not True
        or mapped_manifest.get("gate", {}).get("passed") is not True
    ):
        raise SystemExit("valid-nonpromotion: mapped scale-codebook gate did not pass")
    mapped_scale = {record["tensor"]: record for record in mapped_manifest["records"]}

    full_manifest_path = args.full_codebooks / "manifest.json"
    full_manifest = json.loads(full_manifest_path.read_text())
    expected_full_schema = (
        "qwen38_27b_full_row_normalized_scale_codebook4_receipt_v1"
        if row_normalized
        else "qwen38_27b_full_scale_codebook4_receipt_v1"
    )
    if (
        full_manifest.get("schema") != expected_full_schema
        or full_manifest.get("complete") is not True
        or full_manifest.get("gate", {}).get("passed") is not True
        or full_manifest.get("summary", {}).get("tensors") != EXPECTED_QUANTIZED_TENSORS
        or full_manifest.get("summary", {}).get("scales") != EXPECTED_SCALES
        or full_manifest.get("mapped_parent", {}).get("sha256")
        != sha256_file(mapped_manifest_path)
    ):
        raise SystemExit("valid-nonpromotion: complete scale-codebook gate did not pass")
    supplemental_scale = {
        record["tensor"]: record for record in full_manifest["supplemental_records"]
    }
    if row_normalized:
        row_storage = full_manifest.get("storage_ledger", {})
        if (
            row_storage.get("strategy", {}).get("name")
            != "base3_group128_row_normalized_codebook4"
            or row_storage.get("scales", {}).get("index_bytes")
            != EXPECTED_ROW_SCALE_INDEX_BYTES
            or row_storage.get("scales", {}).get("fp16_row_anchor_bytes")
            != EXPECTED_ROW_SCALE_ANCHOR_BYTES
            or row_storage.get("scales", {}).get("fp16_codebook_bytes")
            != EXPECTED_ROW_SCALE_CODEBOOK_BYTES
            or row_storage.get("bytes", {}).get("scale_payload_anchors_and_codebooks")
            != EXPECTED_ROW_SCALE_BYTES
            or float(row_storage.get("effective_bits_per_total_parameter", math.inf))
            > MAXIMUM_ROW_NORMALIZED_COMPLETE_BPW
        ):
            raise RuntimeError("complete row-normalized storage ledger changed")
        projected_bytes = int(row_storage["bytes"]["complete_bundle"])
        expected_scale_bytes = EXPECTED_ROW_SCALE_BYTES
        maximum_complete_bpw = MAXIMUM_ROW_NORMALIZED_COMPLETE_BPW
        quantized_encoding = "base3_group128_row_normalized_codebook4"
    else:
        projected_bytes = EXPECTED_PROJECTED_BYTES
        expected_scale_bytes = EXPECTED_SCALE_BYTES
        maximum_complete_bpw = MAXIMUM_COMPLETE_BPW
        quantized_encoding = "base3_group128_codebook4"

    tensors, source = inventory_from_safetensors(args.model_dir)
    if source["index_sha256"] != EXPECTED_INDEX_SHA256 or source["inventory_sha256"] != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError("complete Qwen3.8-27B tensor inventory changed")
    tensors = sorted(tensors, key=lambda item: item.name)
    if (
        len(tensors) != EXPECTED_TENSORS
        or sum(tensor.parameters for tensor in tensors) != EXPECTED_PARAMETERS
        or sum(tensor.quantized for tensor in tensors) != EXPECTED_QUANTIZED_TENSORS
        or sum(tensor.parameters for tensor in tensors if tensor.quantized)
        != EXPECTED_QUANTIZED_PARAMETERS
        or sum(not tensor.quantized for tensor in tensors) != EXPECTED_RETAINED_TENSORS
        or sum(tensor.parameters for tensor in tensors if not tensor.quantized)
        != EXPECTED_RETAINED_PARAMETERS
    ):
        raise RuntimeError("complete Qwen3.8-27B export inventory changed")
    quantized_names = {tensor.name for tensor in tensors if tensor.quantized}
    if set(mapped_scale) | set(supplemental_scale) != quantized_names or set(mapped_scale) & set(supplemental_scale):
        raise RuntimeError("scale-codebook coverage is not an exact partition of rank-2+ tensors")

    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spool_root = args.out_dir / "spool"
    spool_root.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / "spool.manifest.partial.json"
    complete_path = args.out_dir / "spool.manifest.json"
    identity = {
        "schema": "qwen38_27b_cq3_export_identity_v1",
        "model_id": MODEL_ID,
        "model_revision": EXPECTED_REVISION,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "mapped_code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "mapped_codebook_manifest_sha256": sha256_file(mapped_manifest_path),
        "full_codebook_manifest_sha256": sha256_file(full_manifest_path),
        "storage_projection_receipt_sha256": EXPECTED_STORAGE_RECEIPT_SHA256,
        "preregistration_sha256": sha256_file(prereg_path),
        "codec": {
            "name": (
                "base3_group128_block203_row_normalized_codebook4"
                if row_normalized
                else "base3_group128_block203_codebook4"
            ),
            "group_size": 128,
            "full_group_bits": 203,
            "group_boundaries": "reset at every flattened first-dimension row",
            "tensor_stream": "bit-contiguous with canonical zero padding only at tensor end",
            "scale_indices": "u4 low nibble first",
            "row_anchor": (
                "one little-endian FP16 value per flattened output row"
                if row_normalized
                else None
            ),
            "scale_codebook": "16 little-endian FP16 values per tensor",
        },
    }
    if complete_path.exists():
        spool = json.loads(complete_path.read_text())
        if spool.get("identity") != identity or spool.get("complete") is not True:
            raise RuntimeError("existing complete export spool identity mismatch")
        for record in spool["tensors"]:
            validate_spool_record(args.out_dir, record)
    elif partial_path.exists():
        spool = json.loads(partial_path.read_text())
        if spool.get("identity") != identity or spool.get("complete") is not False:
            raise RuntimeError("partial export spool identity mismatch")
    else:
        spool = {"identity": identity, "tensors": [], "complete": False}
        atomic_bytes(partial_path, canonical_json(spool))

    with ExitStack() as stack, ProcessPoolExecutor(max_workers=args.workers) as executor:
        handles = {
            shard: stack.enter_context(
                safe_open(args.model_dir / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(weight_map.values()))
        }
        for tensor_index, tensor in enumerate(tensors):
            name = tensor.name
            if tensor_index < len(spool["tensors"]):
                record = spool["tensors"][tensor_index]
                if (
                    record.get("index") != tensor_index
                    or record.get("name") != name
                    or record.get("shape") != list(tensor.shape)
                    or record.get("dtype") != tensor.dtype
                ):
                    raise RuntimeError(f"export spool resume identity mismatch at tensor {tensor_index}")
                validate_spool_record(args.out_dir, record)
                continue

            base = {
                "index": tensor_index,
                "name": name,
                "shape": list(tensor.shape),
                "parameters": tensor.parameters,
                "dtype": tensor.dtype,
                "source_shard": weight_map[name],
            }
            if tensor.quantized:
                code_path = spool_root / f"{tensor_index:04d}.codes"
                if name in mapped_names:
                    source_records = records_by_tensor[name]
                    batches = executor.map(
                        transcode_mapped_record,
                        repeat(str(args.mapped_artifact)),
                        source_records,
                        chunksize=1,
                    )
                    code_receipt = atomic_code_spool(
                        code_path,
                        batches,
                        expected_rows=tensor.rows,
                        expected_columns=tensor.columns,
                    )
                    scale_digest = hashlib.sha256()
                    for source_record in source_records:
                        descriptor = source_record["descriptor"]
                        raw = (args.mapped_artifact / "scales" / descriptor["scale_file"]).read_bytes()
                        if (
                            len(raw) != descriptor["scale_bytes"]
                            or hashlib.sha256(raw).hexdigest() != descriptor["scale_sha256"]
                        ):
                            raise RuntimeError(f"mapped Qwen3.8 scale block changed for {name}")
                        scale_digest.update(raw)
                    if scale_digest.hexdigest() != mapped_scale[name]["source_scale_sha256"]:
                        raise RuntimeError(f"mapped Qwen3.8 tensor-scale stream changed for {name}")
                    code_source = "authenticated_402_tensor_arithmetic_artifact"
                else:
                    supplemental = supplemental_scale[name]
                    view = handles[weight_map[name]].get_slice(name)
                    code_digest = hashlib.sha256()
                    scale_digest = hashlib.sha256()

                    def generated_batches():
                        for row_start in range(0, tensor.rows, ROW_BATCH):
                            row_stop = min(tensor.rows, row_start + ROW_BATCH)
                            rows = view[row_start:row_stop].contiguous().reshape(
                                row_stop - row_start, tensor.columns
                            ).float()
                            codes, scales = ternary_codes_scales(
                                rows, group=128, method="optimal"
                            )
                            code_content = codes.numpy().astype(np.int8, copy=False).tobytes()
                            scale_content = scales.numpy().astype("<f2", copy=False).tobytes()
                            code_digest.update(code_content)
                            scale_digest.update(scale_content)
                            yield pack_verified_rows(codes.numpy())
                            del rows, codes, scales

                    code_receipt = atomic_code_spool(
                        code_path,
                        generated_batches(),
                        expected_rows=tensor.rows,
                        expected_columns=tensor.columns,
                    )
                    if (
                        code_digest.hexdigest() != supplemental["source_ternary_codes_sha256"]
                        or scale_digest.hexdigest() != supplemental["source_scale_sha256"]
                    ):
                        raise RuntimeError(f"supplemental Qwen3.8 quantizer replay changed for {name}")
                    code_source = "replayed_pinned_safetensors_optimal_group128"
                code_receipt["path"] = str(code_path.relative_to(args.out_dir))
                scale, _ = scale_sources(
                    name,
                    mapped_scale,
                    supplemental_scale,
                    args.mapped_codebooks,
                    args.full_codebooks,
                    row_normalized=row_normalized,
                )
                expected_scale_count = tensor.rows * math.ceil(tensor.columns / 128)
                if scale["scale_count"] != expected_scale_count:
                    raise RuntimeError(f"Qwen3.8 scale count changed for {name}")
                record = {
                    **base,
                    "encoding": quantized_encoding,
                    "code_source": code_source,
                    "codes": code_receipt,
                    "scales": scale,
                }
            else:
                content = exact_tensor_bytes(handles[weight_map[name]].get_tensor(name))
                expected_bytes = tensor.parameters * DTYPE_BITS[tensor.dtype] // 8
                if len(content) != expected_bytes:
                    raise RuntimeError(f"retained Qwen3.8 tensor byte count changed for {name}")
                retained_path = spool_root / f"{tensor_index:04d}.retained"
                atomic_bytes(retained_path, content)
                record = {
                    **base,
                    "encoding": f"retained_exact_{tensor.dtype.lower()}",
                    "retained": {
                        "path": str(retained_path.relative_to(args.out_dir)),
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    },
                }
            spool["tensors"].append(record)
            atomic_bytes(partial_path, canonical_json(spool))

    if (
        len(spool["tensors"]) != EXPECTED_TENSORS
        or sum(record["parameters"] for record in spool["tensors"]) != EXPECTED_PARAMETERS
    ):
        raise RuntimeError("Qwen3.8 export spool coverage is incomplete")
    code_bytes = sum(
        record["codes"]["bytes"]
        for record in spool["tensors"]
        if record["encoding"] == quantized_encoding
    )
    retained_bytes = sum(
        record["retained"]["bytes"]
        for record in spool["tensors"]
        if record["encoding"].startswith("retained_exact_")
    )
    if code_bytes != EXPECTED_CODE_BYTES or retained_bytes != EXPECTED_RETAINED_BYTES:
        raise RuntimeError("Qwen3.8 export spool byte ledger differs from the projection")
    if not complete_path.exists():
        spool["complete"] = True
        spool["completed_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        spool["ledger"] = {
            "tensors": len(spool["tensors"]),
            "parameters": EXPECTED_PARAMETERS,
            "ternary_code_bytes": code_bytes,
            "retained_exact_bytes": retained_bytes,
        }
        atomic_bytes(complete_path, canonical_json(spool))
        partial_path.unlink(missing_ok=True)

    resource_guard(Path("/workspace"), assembly=True)
    tensor_records = []
    auxiliary_records = []
    with Qwen38CQ3Writer(
        args.output,
        model={
            "id": MODEL_ID,
            "revision": EXPECTED_REVISION,
            "index_sha256": EXPECTED_INDEX_SHA256,
            "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        },
        strategy={
            "name": quantized_encoding,
            "codec": "block203",
            "group_size": 128,
            "scale_bits": 4,
            "row_anchor_dtype": "F16" if row_normalized else None,
            "scale_codebook_entries": 16,
            "scale_codebook_dtype": "F16",
        },
        parameter_count=EXPECTED_PARAMETERS,
    ) as writer:
        for record in spool["tensors"]:
            prefix = f"tensor.{record['index']:04d}"
            sections = []
            if record["encoding"] == quantized_encoding:
                code = writer.add_section(
                    prefix + ".codes",
                    "ternary_block203",
                    args.out_dir / record["codes"]["path"],
                    metadata={
                        "rows": record["codes"]["rows"],
                        "columns": record["codes"]["columns"],
                        "trits": record["codes"]["trits"],
                        "payload_bits": record["codes"]["payload_bits"],
                        "group_trits": 128,
                        "full_group_bits": 203,
                    },
                )
                if code["bytes"] != record["codes"]["bytes"] or code["sha256"] != record["codes"]["sha256"]:
                    raise RuntimeError(f"container code section changed for {record['name']}")
                scale, scale_paths = scale_sources(
                    record["name"],
                    mapped_scale,
                    supplemental_scale,
                    args.mapped_codebooks,
                    args.full_codebooks,
                    row_normalized=row_normalized,
                )
                scale_kind = (
                    "scale_indices_u4_anchors_f16_and_codebook_f16"
                    if row_normalized
                    else "scale_indices_u4_and_codebook_f16"
                )
                scale_metadata = {
                    "scale_count": scale["scale_count"],
                    "index_bytes": scale["indices"]["bytes"],
                    "codebook_bytes": scale["codebook"]["bytes"],
                    "codebook_entries": 16,
                    "index_order": "low_nibble_first",
                }
                if row_normalized:
                    scale_metadata.update(
                        {
                            "row_count": int(record["shape"][0]),
                            "anchor_bytes": scale["anchors"]["bytes"],
                            "factor_order": "indices_then_row_anchors_then_codebook",
                            "decode": "float32(fp16_anchor) * float32(fp16_ratio)",
                        }
                    )
                scales = writer.add_section_parts(
                    prefix + ".scales",
                    scale_kind,
                    scale_paths,
                    metadata=scale_metadata,
                )
                expected_tensor_scale_bytes = (
                    scale["indices"]["bytes"]
                    + scale["codebook"]["bytes"]
                    + (scale["anchors"]["bytes"] if row_normalized else 0)
                )
                if scales["bytes"] != expected_tensor_scale_bytes:
                    raise RuntimeError(f"container scale section changed for {record['name']}")
                sections.extend((code["id"], scales["id"]))
            else:
                retained = writer.add_section(
                    prefix + ".retained",
                    record["encoding"],
                    args.out_dir / record["retained"]["path"],
                    metadata={"dtype": record["dtype"]},
                )
                if (
                    retained["bytes"] != record["retained"]["bytes"]
                    or retained["sha256"] != record["retained"]["sha256"]
                ):
                    raise RuntimeError(f"container retained section changed for {record['name']}")
                sections.append(retained["id"])
            tensor_records.append(
                {
                    "name": record["name"],
                    "shape": record["shape"],
                    "parameters": record["parameters"],
                    "source_dtype": record["dtype"],
                    "encoding": record["encoding"],
                    "sections": sections,
                }
            )
        for auxiliary_index, expected in enumerate(source["auxiliary_files"]):
            path = args.model_dir / expected["name"]
            if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                raise RuntimeError(f"Qwen3.8 auxiliary file changed: {path}")
            section = writer.add_section(
                f"aux.{auxiliary_index:04d}",
                "auxiliary_file",
                path,
                metadata={"name": expected["name"]},
            )
            auxiliary_records.append(
                {
                    "name": expected["name"],
                    "section": section["id"],
                    "bytes": section["bytes"],
                    "sha256": section["sha256"],
                }
            )
        write_receipt = writer.finalize(
            tensors=tensor_records,
            auxiliary_files=auxiliary_records,
        )

    with Qwen38CQ3Reader(args.output) as reader:
        verification = reader.verify_all()
        directory = reader.directory
    section_totals: dict[str, int] = {}
    for section in directory["sections"]:
        section_totals[section["kind"]] = section_totals.get(section["kind"], 0) + section["bytes"]
    scale_kind = (
        "scale_indices_u4_anchors_f16_and_codebook_f16"
        if row_normalized
        else "scale_indices_u4_and_codebook_f16"
    )
    if (
        section_totals.get("ternary_block203") != EXPECTED_CODE_BYTES
        or section_totals.get(scale_kind) != expected_scale_bytes
        or sum(
            size for kind, size in section_totals.items() if kind.startswith("retained_exact_")
        )
        != EXPECTED_RETAINED_BYTES
        or section_totals.get("auxiliary_file") != EXPECTED_AUXILIARY_BYTES
    ):
        raise RuntimeError("measured Qwen3.8 container section ledger changed")
    output_bytes = args.output.stat().st_size
    complete_bpw = output_bytes * 8 / EXPECTED_PARAMETERS
    gate = {
        "maximum_complete_container_bpw": maximum_complete_bpw,
        "observed_complete_container_bpw": complete_bpw,
        "projected_bytes": projected_bytes,
        "observed_bytes": output_bytes,
        "observed_minus_projected_bytes": output_bytes - projected_bytes,
        "passed": complete_bpw <= maximum_complete_bpw,
    }
    receipt = {
        "schema": (
            "qwen38_27b_cq3_row_normalized_export_receipt_v1"
            if row_normalized
            else "qwen38_27b_cq3_export_receipt_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "id": MODEL_ID,
            "revision": EXPECTED_REVISION,
            "parameters": EXPECTED_PARAMETERS,
            "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        },
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "native_cpu": {"available": True, "features": cpu_features()},
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "parents": {
            "mapped_code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
            "mapped_codebook_manifest_sha256": sha256_file(mapped_manifest_path),
            "full_codebook_manifest_sha256": sha256_file(full_manifest_path),
            "storage_projection_receipt_sha256": EXPECTED_STORAGE_RECEIPT_SHA256,
            "row_normalized_storage_ledger": (
                full_manifest["storage_ledger"] if row_normalized else None
            ),
            "spool_manifest_sha256": sha256_file(complete_path),
        },
        "artifact": {
            "path": str(args.output),
            "bytes": output_bytes,
            "sha256": sha256_file(args.output),
            "write_receipt": write_receipt,
            "independent_verification": verification,
            "section_payload_bytes_by_kind": section_totals,
        },
        "gate": gate,
        "claim_boundary": {
            "complete_27b_container_exported": True,
            "all_sections_independently_authenticated": True,
            "model_quality_proven": False,
            "end_to_end_inference_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.receipt, canonical_json(receipt))
    atomic_bytes(args.out_dir / "ARTIFACT_COMPLETE", b"complete\n")
    atomic_bytes(
        args.out_dir / ("STORAGE_PASS" if gate["passed"] else "STORAGE_REJECTED"),
        ("pass\n" if gate["passed"] else "rejected\n").encode(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
