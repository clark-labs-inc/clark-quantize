#!/usr/bin/env python3
"""Benchmark the exact native 203-bit codec on a real Qwen3.8-27B tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_block203 import pack_row
from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader
from clark_quantize.ternary_cpu import (
    block203_size,
    memory_read_sum,
    native_available,
    pack_block203,
    unpack_block203,
)
from benchmark_ternary_gemv import file_sha256, load_tensor_codes


def validate_preregistration(value: dict, args: argparse.Namespace) -> None:
    sample = value.get("sample", {})
    performance = value.get("performance", {})
    if (
        value.get("schema") != "clark_qwen38_27b_block203_codec_preregistration_v1"
        or value.get("model", {}).get("id") != "Qwen/Qwen3.8-27B"
        or sample.get("tensor") != args.tensor
        or sample.get("rows") != args.rows
        or sample.get("columns") != args.columns
        or sample.get("group_trits") != 128
        or sample.get("group_bits") != 203
        or performance.get("warmups") != args.warmups
        or performance.get("repetitions") != args.repetitions
        or performance.get("threads") != 1
        or performance.get("cache_eviction_bytes") != args.cache_eviction_bytes
    ):
        raise ValueError("block-203 benchmark arguments differ from preregistration")


def summarize(samples: list[float], trits: int) -> dict:
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "median_mtrits_per_second": trits / median / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"
        ),
    )
    parser.add_argument(
        "--pack-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-base3-full.json"),
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(__file__).with_name(
            "qwen38_27b_block203_codec_preregistration_v1.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-block203-codec-v1.json"),
    )
    parser.add_argument(
        "--tensor", default="model.language_model.layers.0.mlp.gate_proj.weight"
    )
    parser.add_argument("--rows", type=int, default=17408)
    parser.add_argument("--columns", type=int, default=5120)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--cache-eviction-bytes", type=int, default=268435456)
    args = parser.parse_args()

    preregistration_content = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_content)
    validate_preregistration(preregistration, args)
    if not native_available():
        raise RuntimeError("the preregistered benchmark requires the native CPU library")

    pack_receipt = json.loads(args.pack_receipt.read_text())
    expected_revision = preregistration["model"]["revision"]
    if (
        pack_receipt.get("promotion", {}).get("passed") is not True
        or pack_receipt.get("identity", {}).get("model_revision") != expected_revision
    ):
        raise ValueError("full-pack receipt is not the promoted Qwen3.8-27B parent")
    reader = QwenBase3ArtifactReader(args.artifact)
    if reader.manifest.get("schema") != preregistration["parent"]["artifact_schema"]:
        raise ValueError("artifact schema differs from preregistration")

    codes, _, record_indices = load_tensor_codes(
        reader, args.tensor, args.rows, args.columns
    )
    codes = np.ascontiguousarray(codes.reshape(-1), dtype=np.int8)
    payload_bits, payload_bytes = block203_size(codes.size)
    row_bits, row_bytes = block203_size(args.columns)
    if row_bits % 8 != 0 or row_bytes * args.rows != payload_bytes:
        raise AssertionError("the preregistered Qwen row layout is not byte aligned")

    reference_rows = preregistration["sample"]["reference_row_indices"]
    reference_parity = {}
    initial_payload = pack_block203(codes)
    for row in reference_rows:
        start = row * args.columns
        stop = start + args.columns
        native = initial_payload[row * row_bytes : (row + 1) * row_bytes].tobytes()
        reference = pack_row(codes[start:stop])
        reference_parity[str(row)] = {
            "equal": native == reference,
            "sha256": hashlib.sha256(native).hexdigest(),
        }

    initial_decoded = unpack_block203(initial_payload, codes.size)
    exact_roundtrip = bool(np.array_equal(initial_decoded, codes))
    del initial_decoded

    eviction = np.arange(args.cache_eviction_bytes, dtype=np.uint8)
    expected_eviction_sum = int(eviction.sum(dtype=np.uint64))

    def evict() -> None:
        if memory_read_sum(eviction, implementation="scalar") != expected_eviction_sum:
            raise AssertionError("cache-eviction checksum changed")

    for _ in range(args.warmups):
        warm_payload = pack_block203(codes)
        warm_decoded = unpack_block203(warm_payload, codes.size)
        if not np.array_equal(warm_decoded, codes):
            raise AssertionError("warmup roundtrip changed codes")

    samples = {"pack": [], "unpack": []}
    pack_hashes: list[str] = []
    unpack_hashes: list[str] = []
    for repetition in range(args.repetitions):
        order = ("pack", "unpack") if repetition % 2 == 0 else ("unpack", "pack")
        for operation in order:
            evict()
            started = time.perf_counter()
            if operation == "pack":
                result = pack_block203(codes)
            else:
                result = unpack_block203(initial_payload, codes.size)
            samples[operation].append(time.perf_counter() - started)
            digest = hashlib.sha256(result.tobytes()).hexdigest()
            if operation == "pack":
                pack_hashes.append(digest)
            else:
                unpack_hashes.append(digest)

    pack_summary = summarize(samples["pack"], codes.size)
    unpack_summary = summarize(samples["unpack"], codes.size)
    ternary_parameters = preregistration["model"]["ternary_parameters"]
    projected_pack_seconds = (
        ternary_parameters / (pack_summary["median_mtrits_per_second"] * 1e6)
    )
    projected_unpack_seconds = (
        ternary_parameters / (unpack_summary["median_mtrits_per_second"] * 1e6)
    )
    payload_bits_per_trit = payload_bits / codes.size
    promotion = preregistration["promotion"]
    all_pack_hashes_identical = len(set(pack_hashes)) == 1
    all_unpack_hashes_identical = len(set(unpack_hashes)) == 1
    selected_rows_match = all(item["equal"] for item in reference_parity.values())
    passed = (
        exact_roundtrip
        and all_pack_hashes_identical
        and all_unpack_hashes_identical
        and selected_rows_match
        and projected_pack_seconds
        <= promotion["maximum_projected_whole_model_pack_seconds"]
        and projected_unpack_seconds
        <= promotion["maximum_projected_whole_model_unpack_seconds"]
        and payload_bits_per_trit
        <= promotion["maximum_payload_bits_per_ternary_parameter"]
    )

    native_library = next(
        path
        for path in (
            REPO / "clark_quantize/_ternary_cpu.so",
            REPO / "clark_quantize/_ternary_cpu.dylib",
        )
        if path.exists()
    )
    report = {
        "schema": "clark_qwen38_27b_block203_codec_receipt_v1",
        "created_at": time.time(),
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(preregistration_content).hexdigest(),
            "rules": preregistration,
        },
        "parent": {
            "artifact": str(args.artifact),
            "artifact_manifest_sha256": file_sha256(reader.manifest_path),
            "pack_receipt": str(args.pack_receipt),
            "pack_receipt_sha256": file_sha256(args.pack_receipt),
            "tensor": args.tensor,
            "record_indices": record_indices,
            "codes": int(codes.size),
            "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
        },
        "build": {
            "repo_commit": subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
            ).strip(),
            "native_library": str(native_library),
            "native_library_sha256": file_sha256(native_library),
            "c_source_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.c"),
            "python_wrapper_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.py"),
            "benchmark_sha256": file_sha256(Path(__file__)),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        },
        "format": {
            "group_trits": 128,
            "group_bits": 203,
            "payload_bits": payload_bits,
            "payload_bytes": payload_bytes,
            "payload_bits_per_trit": payload_bits_per_trit,
            "row_bits": row_bits,
            "row_bytes": row_bytes,
        },
        "correctness": {
            "full_tensor_exact_roundtrip": exact_roundtrip,
            "selected_row_reference_parity": reference_parity,
            "all_pack_payload_hashes_identical": all_pack_hashes_identical,
            "all_unpack_output_hashes_identical": all_unpack_hashes_identical,
            "pack_payload_sha256": pack_hashes[0],
            "unpack_output_sha256": unpack_hashes[0],
        },
        "performance": {
            "pack": pack_summary,
            "unpack": unpack_summary,
            "projected_whole_model_pack_seconds": projected_pack_seconds,
            "projected_whole_model_unpack_seconds": projected_unpack_seconds,
            "ternary_parameters": ternary_parameters,
            "cache_eviction_bytes": args.cache_eviction_bytes,
        },
        "promotion": {
            "passed": passed,
            "rules": promotion,
            "claim_boundary": preregistration["claim_boundary"],
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(args.receipt, canonical_json(report))
    print(json.dumps(report["promotion"], sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
