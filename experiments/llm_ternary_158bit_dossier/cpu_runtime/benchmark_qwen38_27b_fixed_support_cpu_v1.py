#!/usr/bin/env python3
"""Benchmark exact native fixed-support coding on Qwen3.8 dimensions."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_fixed_support_cpu import load_fixed_support_cpu_library


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
            raise RuntimeError(f"frozen fixed-support CPU source changed: {label}")


def timed(function, *, warmups: int, repetitions: int) -> dict:
    for _ in range(warmups):
        if function() != 0:
            raise RuntimeError("native fixed-support warmup failed")
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        status = function()
        elapsed = time.perf_counter_ns() - started
        if status != 0:
            raise RuntimeError(f"native fixed-support timed call failed: {status}")
        samples.append(elapsed)
    ordered = sorted(samples)
    return {
        "samples_ns": samples,
        "median_ns": float(statistics.median(ordered)),
        "minimum_ns": float(ordered[0]),
        "p90_ns": float(ordered[math.ceil(0.9 * len(ordered)) - 1]),
    }


def make_codes(
    *, rows: int, columns: int, support: int, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    codes = np.zeros((rows, columns), dtype=np.int8)
    for group in codes.reshape(-1, 128):
        positions = generator.choice(128, size=support, replace=False)
        group[positions] = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=support
        )
    return codes


def cpu_metadata() -> dict:
    if shutil.which("lscpu"):
        return json.loads(subprocess.check_output(["lscpu", "-J"], text=True))
    return {
        "fallback": True,
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-fixed-support-cpu-v1.json"),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 9 or args.repetitions % 2 == 0:
        raise ValueError("fixed-support benchmark requires odd repetitions >= 9")

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_fixed_support_cpu_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    native = load_fixed_support_cpu_library(args.library)
    if native is None:
        raise RuntimeError("native fixed-support library did not load")
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)

    records = []
    for index, case in enumerate(prereg["cases"]):
        rows = int(case["rows"])
        columns = int(case["columns"])
        support = int(case["nonzero_per_group"])
        codes = make_codes(
            rows=rows,
            columns=columns,
            support=support,
            seed=int(prereg["measurement"]["seed"]) + index,
        )
        bits = ctypes.c_uint64()
        size = ctypes.c_uint64()
        status = native.library.cq3_hfs_matrix_size(
            rows, columns, support, ctypes.byref(bits), ctypes.byref(size)
        )
        if status != 0:
            raise RuntimeError(f"native fixed-support size failed: {status}")
        payload = np.zeros(size.value, dtype=np.uint8)
        decoded = np.empty_like(codes)
        observed_bits = ctypes.c_uint64()
        pack_args = (
            codes.ctypes.data_as(i8p), rows, columns, support,
            payload.ctypes.data_as(u8p), payload.size,
            ctypes.byref(observed_bits),
        )
        unpack_args = (
            payload.ctypes.data_as(u8p), payload.size, bits.value,
            rows, columns, support, decoded.ctypes.data_as(i8p),
        )
        if native.library.cq3_hfs_pack_matrix(*pack_args) != 0:
            raise RuntimeError("native fixed-support initial pack failed")
        if observed_bits.value != bits.value:
            raise RuntimeError("native fixed-support bit ledger changed")
        if native.library.cq3_hfs_unpack_matrix(*unpack_args) != 0:
            raise RuntimeError("native fixed-support initial unpack failed")
        exact = bool(np.array_equal(decoded, codes))
        if not exact:
            raise RuntimeError("native fixed-support round trip changed codes")
        pack_timing = timed(
            lambda: native.library.cq3_hfs_pack_matrix(*pack_args),
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        unpack_timing = timed(
            lambda: native.library.cq3_hfs_unpack_matrix(*unpack_args),
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        weights = rows * columns
        records.append(
            {
                "name": case["name"],
                "rows": rows,
                "columns": columns,
                "nonzero_per_group": support,
                "weights": weights,
                "payload_bits": bits.value,
                "payload_bytes": size.value,
                "code_bits_per_weight": bits.value / weights,
                "exact_roundtrip": exact,
                "payload_sha256": hashlib.sha256(payload.tobytes()).hexdigest(),
                "pack_timing": pack_timing,
                "unpack_timing": unpack_timing,
                "pack_mweights_per_second": weights / pack_timing["median_ns"] * 1e3,
                "unpack_mweights_per_second": weights / unpack_timing["median_ns"] * 1e3,
            }
        )

    minimum_pack = min(record["pack_mweights_per_second"] for record in records)
    minimum_unpack = min(record["unpack_mweights_per_second"] for record in records)
    maximum_rate = max(record["code_bits_per_weight"] for record in records)
    all_exact = all(record["exact_roundtrip"] for record in records)
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_cases": len(records),
        "observed_all_exact_roundtrips": all_exact,
        "observed_minimum_pack_mweights_per_second": minimum_pack,
        "observed_minimum_unpack_mweights_per_second": minimum_unpack,
        "observed_maximum_code_bits_per_weight": maximum_rate,
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_cases"])
        and all_exact
        and minimum_pack >= float(gate_spec["minimum_pack_mweights_per_second"])
        and minimum_unpack >= float(gate_spec["minimum_unpack_mweights_per_second"])
        and maximum_rate <= float(gate_spec["maximum_code_bits_per_weight"])
    )
    receipt = {
        "schema": "qwen38_27b_fixed_support_cpu_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "library_sha256": sha256_file(args.library),
        "cpu": cpu_metadata(),
        "process_metadata": {
            "binomial_table_entries": 129 * 65,
            "binomial_table_bytes": 129 * 65 * 16,
            "counted_as_model_storage": False,
        },
        "records": records,
        "gate": gate,
        "claim_boundary": {
            "exact_codec_arithmetic_proven": gate["passed"],
            "Qwen_input_dimensions_benchmarked": True,
            "synthetic_codes_used": True,
            "one_core_load_time_throughput_measured": True,
            "fused_compressed_GEMV_measured": False,
            "token_throughput_or_capability_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
