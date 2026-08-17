#!/usr/bin/env python3
"""Benchmark the native Clark-Hash ``A(Rx)`` correction on Qwen3.8 shapes."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_hash_residual import sparse_jl_projection
from clark_quantize.qwen38_hash_residual_cpu import (
    load_hash_residual_cpu_library,
    native_hash_int4_gemv_f32,
    native_hash_projection_f32,
    native_hash_residual_f32,
    pack_signed_int4_rows,
)


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
            raise RuntimeError(f"frozen native hash-residual source changed: {label}")


def timed(function, *, warmups: int, repetitions: int) -> dict:
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - started)
    ordered = sorted(samples)
    median = float(statistics.median(ordered))
    return {
        "samples_ns": samples,
        "median_ns": median,
        "minimum_ns": float(ordered[0]),
        "p90_ns": float(ordered[math.ceil(0.9 * len(ordered)) - 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--parent-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-hash-residual-screen-v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-hash-residual-cpu-v1.json"),
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 9 or args.repetitions % 2 == 0:
        raise ValueError("hash-residual benchmark requires warmups and odd repetitions >= 9")

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_hash_residual_cpu_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    parent = json.loads(args.parent_receipt.read_text())
    if parent.get("schema") != "qwen38_27b_hash_residual_screen_receipt_v1":
        raise RuntimeError("native benchmark parent receipt schema changed")
    if not parent.get("gate", {}).get("passed"):
        raise RuntimeError("native hash benchmark requires a passing data-free parent")
    parent_prereg_path = runtime / "qwen38_27b_hash_residual_screen_preregistration_v1.json"
    if parent.get("preregistration_sha256") != sha256_file(parent_prereg_path):
        raise RuntimeError("native hash benchmark parent preregistration changed")
    selected = parent["gate"]["selected_candidate"]
    candidates = {
        item["name"]: item
        for item in json.loads(parent_prereg_path.read_text())["candidate_family"]["candidates"]
    }
    if selected not in candidates:
        raise RuntimeError("native hash benchmark parent candidate is unknown")
    candidate = candidates[selected]
    ratio = int(candidate["input_to_sketch_ratio"])
    seed = int(prereg["representation"]["projection_seed"])
    hashes_per_input = int(prereg["representation"]["hashes_per_input"])

    native = load_hash_residual_cpu_library(args.library)
    if native is None:
        raise RuntimeError("native hash-residual library did not load")
    features = native.features()
    if not features["avx2"] or not features["fma"]:
        raise RuntimeError("authoritative native hash-residual benchmark requires AVX2/FMA")
    f32p = ctypes.POINTER(ctypes.c_float)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u16p = ctypes.POINTER(ctypes.c_uint16)

    records = []
    arrays_by_name = {}
    for case_index, case in enumerate(prereg["cases"]):
        name = case["name"]
        input_dim = int(case["input_dim"])
        rows = int(case["output_rows"])
        if input_dim % ratio != 0:
            raise RuntimeError(f"selected candidate does not divide {name}")
        sketch_dim = input_dim // ratio
        projection = sparse_jl_projection(
            input_dim, sketch_dim, hashes_per_input=hashes_per_input, seed=seed
        )
        generator = np.random.default_rng(int(prereg["measurement"]["seed"]) + case_index)
        values = generator.normal(size=input_dim).astype(np.float32)
        codes = generator.integers(-7, 8, size=(rows, sketch_dim), dtype=np.int8)
        scales = generator.uniform(1e-4, 0.02, size=rows).astype(np.float16)
        packed = pack_signed_int4_rows(codes)
        buckets = np.ascontiguousarray(projection.buckets.numpy(), dtype=np.uint16)
        signs = np.ascontiguousarray(projection.signs.numpy(), dtype=np.int8)
        scale_bits = np.ascontiguousarray(scales.view(np.uint16))
        scratch = np.empty(sketch_dim, dtype=np.float32)
        scalar_output = np.empty(rows, dtype=np.float32)
        avx2_output = np.empty(rows, dtype=np.float32)
        normalization = np.float32(1.0 / math.sqrt(hashes_per_input))

        projected = native_hash_projection_f32(
            values, buckets, signs, sketch_dim=sketch_dim, library=native
        )
        scalar = native_hash_int4_gemv_f32(
            packed, scales, projected, sketch_dim=sketch_dim, library=native,
            implementation="scalar",
        )
        avx2 = native_hash_int4_gemv_f32(
            packed, scales, projected, sketch_dim=sketch_dim, library=native,
            implementation="avx2",
        )
        fused = native_hash_residual_f32(
            values, buckets, signs, packed, scales, sketch_dim=sketch_dim,
            library=native, implementation="avx2",
        )
        base = generator.normal(size=rows).astype(np.float32)
        fused_add = native_hash_residual_f32(
            values, buckets, signs, packed, scales, sketch_dim=sketch_dim,
            library=native, implementation="avx2", base_output=base,
        )
        scalar_error = float(np.max(np.abs(avx2 - scalar)))
        fused_error = float(np.max(np.abs(fused - avx2)))
        add_error = float(np.max(np.abs(fused_add - (base + avx2))))

        projection_call = native.library.cq3_hash_projection_f32_scalar_trusted
        scalar_call = native.library.cq3_hash_int4_gemv_f32_scalar_trusted
        avx2_call = native.library.cq3_hash_int4_gemv_f32_avx2_trusted
        fused_call = native.library.cq3_hash_residual_f32_avx2_trusted

        projection_args = (
            values.ctypes.data_as(f32p), buckets.ctypes.data_as(u16p),
            signs.ctypes.data_as(i8p), input_dim, sketch_dim, hashes_per_input,
            normalization, scratch.ctypes.data_as(f32p),
        )
        scalar_args = (
            packed.ctypes.data_as(u8p), scale_bits.ctypes.data_as(u16p),
            projected.ctypes.data_as(f32p), rows, sketch_dim,
            scalar_output.ctypes.data_as(f32p),
        )
        avx2_args = (
            packed.ctypes.data_as(u8p), scale_bits.ctypes.data_as(u16p),
            projected.ctypes.data_as(f32p), rows, sketch_dim,
            avx2_output.ctypes.data_as(f32p),
        )
        fused_args = (
            values.ctypes.data_as(f32p), buckets.ctypes.data_as(u16p),
            signs.ctypes.data_as(i8p), input_dim, sketch_dim, hashes_per_input,
            normalization, packed.ctypes.data_as(u8p),
            scale_bits.ctypes.data_as(u16p), rows, scratch.ctypes.data_as(f32p),
            avx2_output.ctypes.data_as(f32p), 0,
        )
        projection_timing = timed(
            lambda: projection_call(*projection_args),
            warmups=args.warmups, repetitions=args.repetitions,
        )
        scalar_timing = timed(
            lambda: scalar_call(*scalar_args),
            warmups=args.warmups, repetitions=args.repetitions,
        )
        avx2_timing = timed(
            lambda: avx2_call(*avx2_args),
            warmups=args.warmups, repetitions=args.repetitions,
        )
        fused_timing = timed(
            lambda: fused_call(*fused_args),
            warmups=args.warmups, repetitions=args.repetitions,
        )
        coefficient_bytes = int(packed.nbytes + scales.nbytes)
        record = {
            "name": name,
            "input_dim": input_dim,
            "output_rows": rows,
            "sketch_dim": sketch_dim,
            "coefficient_bytes": coefficient_bytes,
            "coefficient_bits_per_original_weight": coefficient_bytes * 8 / (rows * input_dim),
            "load_time_projection_metadata_bytes": int(buckets.nbytes + signs.nbytes),
            "scratch_bytes": int(scratch.nbytes),
            "maximum_absolute_error_avx2_vs_scalar": scalar_error,
            "maximum_absolute_error_fused_vs_separate": fused_error,
            "maximum_absolute_error_fused_add": add_error,
            "projection_timing": projection_timing,
            "scalar_gemv_timing": scalar_timing,
            "avx2_gemv_timing": avx2_timing,
            "fused_avx2_timing": fused_timing,
            "avx2_speedup_vs_scalar": scalar_timing["median_ns"] / avx2_timing["median_ns"],
            "avx2_coefficient_read_gib_per_second": (
                coefficient_bytes / (1024**3) / (avx2_timing["median_ns"] * 1e-9)
            ),
        }
        records.append(record)
        arrays_by_name[name] = {
            "values": values,
            "buckets": buckets,
            "signs": signs,
            "normalization": normalization,
            "sketch": scratch,
            "packed": packed,
            "scale_bits": scale_bits,
            "output": avx2_output,
            "sketch_dim": sketch_dim,
            "input_dim": input_dim,
            "rows": rows,
        }

    # Gate and up have identical input/projection geometry, so the production
    # schedule computes R x once and runs two coefficient GEMVs.
    gate = arrays_by_name["gate"]
    up = arrays_by_name["up"]
    if not np.array_equal(gate["buckets"], up["buckets"]) or not np.array_equal(gate["signs"], up["signs"]):
        raise RuntimeError("gate/up Clark Hash projections cannot be shared")
    projection_call = native.library.cq3_hash_projection_f32_scalar_trusted
    avx2_call = native.library.cq3_hash_int4_gemv_f32_avx2_trusted
    shared_scratch = np.empty(gate["sketch_dim"], dtype=np.float32)
    shared_gate_output = np.empty(gate["rows"], dtype=np.float32)
    shared_up_output = np.empty(up["rows"], dtype=np.float32)

    def shared_gate_up() -> None:
        projection_call(
            gate["values"].ctypes.data_as(f32p), gate["buckets"].ctypes.data_as(u16p),
            gate["signs"].ctypes.data_as(i8p), gate["input_dim"], gate["sketch_dim"],
            hashes_per_input, gate["normalization"], shared_scratch.ctypes.data_as(f32p),
        )
        for current, output in ((gate, shared_gate_output), (up, shared_up_output)):
            avx2_call(
                current["packed"].ctypes.data_as(u8p),
                current["scale_bits"].ctypes.data_as(u16p),
                shared_scratch.ctypes.data_as(f32p), current["rows"],
                current["sketch_dim"], output.ctypes.data_as(f32p),
            )

    shared_timing = timed(
        shared_gate_up, warmups=args.warmups, repetitions=args.repetitions
    )
    maximum_error = max(
        max(
            record["maximum_absolute_error_avx2_vs_scalar"],
            record["maximum_absolute_error_fused_vs_separate"],
            record["maximum_absolute_error_fused_add"],
        )
        for record in records
    )
    minimum_speedup = min(record["avx2_speedup_vs_scalar"] for record in records)
    gate_spec = prereg["gate"]
    gate_result = {
        **gate_spec,
        "selected_candidate": selected,
        "observed_cases": len(records),
        "observed_maximum_absolute_error": maximum_error,
        "observed_minimum_avx2_speedup_vs_scalar": minimum_speedup,
        "observed_projected_component_bits_per_weight": parent["gate"][
            "observed_projected_component_bits_per_weight"
        ],
    }
    gate_result["passed"] = bool(
        len(records) == int(gate_spec["required_qwen_projection_cases"])
        and maximum_error <= float(gate_spec["maximum_absolute_error"])
        and minimum_speedup >= float(gate_spec["minimum_avx2_gemv_speedup_vs_scalar"])
        and gate_result["observed_projected_component_bits_per_weight"]
        <= float(gate_spec["maximum_projected_component_bits_per_weight"])
    )
    receipt = {
        "schema": "qwen38_27b_hash_residual_cpu_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "library_sha256": sha256_file(args.library),
        "parent_receipt_sha256": sha256_file(args.parent_receipt),
        "cpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
        "features": features,
        "selected_candidate": selected,
        "records": records,
        "shared_gate_up_timing": shared_timing,
        "gate": gate_result,
        "claim_boundary": {
            "real_Qwen38_selected_representation_used": True,
            "exact_Qwen38_projection_shapes_benchmarked": True,
            "synthetic_coefficients_used_for_full_shape_timing": True,
            "native_AVX2_correction_arithmetic_proven": gate_result["passed"],
            "ternary_base_kernel_fused_in_same_call": False,
            "complete_layer_or_model_speedup_proven": False,
            "capability_preservation_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate_result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
