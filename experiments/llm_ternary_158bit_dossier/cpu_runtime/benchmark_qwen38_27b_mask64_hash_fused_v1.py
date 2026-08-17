#!/usr/bin/env python3
"""Measure one-call mask64 ternary GEMV plus learned-route INT4 correction."""
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
from clark_quantize.qwen38_fixed_support_mask64_cpu import (
    mask64_hash_component_storage,
    native_pack_mask64_matrix,
    pack_scale_indices4,
)
from clark_quantize.qwen38_hash_residual_cpu import pack_signed_int4_rows
from clark_quantize.qwen38_mask64_hash_fused_cpu import (
    load_mask64_hash_fused_cpu_library,
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
            raise RuntimeError(f"frozen fused mask64/hash source changed: {label}")


def timed(function, *, warmups: int, repetitions: int) -> dict:
    for _ in range(warmups):
        if function() != 0:
            raise RuntimeError("fused mask64/hash GEMV warmup failed")
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        status = function()
        elapsed = time.perf_counter_ns() - started
        if status != 0:
            raise RuntimeError(f"fused mask64/hash timed call failed: {status}")
        samples.append(elapsed)
    ordered = sorted(samples)
    return {
        "samples_ns": samples,
        "median_ns": float(statistics.median(ordered)),
        "minimum_ns": float(ordered[0]),
        "p90_ns": float(ordered[math.ceil(0.9 * len(ordered)) - 1]),
    }


def make_codes(
    generator: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    codes = np.zeros((rows, columns), dtype=np.int8)
    groups = codes.reshape(-1, 128)
    for start in range(0, groups.shape[0], 2048):
        stop = min(start + 2048, groups.shape[0])
        count = stop - start
        scores = generator.random((count, 128), dtype=np.float32)
        positions = np.argpartition(scores, 64, axis=1)[:, :64]
        signs = generator.integers(0, 2, size=(count, 64), dtype=np.int8)
        signs = signs * np.int8(2) - np.int8(1)
        groups[start:stop][np.arange(count)[:, None], positions] = signs
    return codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-mask64-hash-fused-v1.json"),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 9 or args.repetitions % 2 == 0:
        raise ValueError("fused mask64/hash benchmark requires odd repetitions >= 9")

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_mask64_hash_fused_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    native = load_mask64_hash_fused_cpu_library(args.library)
    if native is None:
        raise RuntimeError("fused mask64/hash CPU library did not load")
    features = native.features()
    if not all(features.values()):
        raise RuntimeError("authoritative fused mask64/hash run requires AVX2/FMA/F16C/BMI2")

    u8p = ctypes.POINTER(ctypes.c_uint8)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    records = []
    for case_index, case in enumerate(prereg["cases"]):
        rows = int(case["rows"])
        columns = int(case["columns"])
        sketch_dim = int(case["sketch_dim"])
        groups = columns // 128
        generator = np.random.default_rng(
            int(prereg["measurement"]["seed"]) + case_index
        )
        codes = make_codes(generator, rows, columns)
        encoded = native_pack_mask64_matrix(codes, library=native.mask)
        weights = np.frombuffer(encoded.payload, dtype=np.uint8).copy()
        indices = generator.integers(1, 16, size=(rows, groups), dtype=np.uint8)
        packed_indices = pack_scale_indices4(indices)
        anchors = generator.uniform(0.001, 0.02, size=rows).astype(np.float16)
        ratios = generator.uniform(0.5, 1.5, size=16).astype(np.float16)
        ratios[0] = np.float16(0.0)
        activation = generator.integers(-127, 128, size=columns, dtype=np.int8)
        activation_scales = generator.uniform(0.002, 0.03, size=groups).astype(
            np.float32
        )
        activation_sums = activation.reshape(groups, 128).sum(
            axis=1, dtype=np.int32
        )
        buckets = generator.integers(
            0, sketch_dim, size=columns, dtype=np.uint16
        )
        route_signs = generator.integers(0, 2, size=columns, dtype=np.int8)
        route_signs = route_signs * np.int8(2) - np.int8(1)
        correction_codes = generator.integers(
            -7, 8, size=(rows, sketch_dim), dtype=np.int8
        )
        packed_correction = pack_signed_int4_rows(correction_codes)
        correction_scales = generator.uniform(1e-4, 0.01, size=rows).astype(
            np.float16
        )
        scratch = np.empty(sketch_dim, dtype=np.float32)
        base_output = np.empty(rows, dtype=np.float32)
        fused_output = np.empty(rows, dtype=np.float32)
        anchors_u16 = anchors.view(np.uint16)
        ratios_u16 = ratios.view(np.uint16)
        correction_scales_u16 = correction_scales.view(np.uint16)
        base_args = (
            weights.ctypes.data_as(u8p),
            packed_indices.ctypes.data_as(u8p),
            anchors_u16.ctypes.data_as(u16p),
            ratios_u16.ctypes.data_as(u16p),
            activation.ctypes.data_as(i8p),
            activation_scales.ctypes.data_as(f32p),
            activation_sums.ctypes.data_as(i32p),
            rows, columns,
            base_output.ctypes.data_as(f32p),
        )
        fused_args = (
            weights.ctypes.data_as(u8p),
            packed_indices.ctypes.data_as(u8p),
            anchors_u16.ctypes.data_as(u16p),
            ratios_u16.ctypes.data_as(u16p),
            activation.ctypes.data_as(i8p),
            activation_scales.ctypes.data_as(f32p),
            activation_sums.ctypes.data_as(i32p),
            rows, columns,
            buckets.ctypes.data_as(u16p),
            route_signs.ctypes.data_as(i8p),
            sketch_dim,
            packed_correction.ctypes.data_as(u8p),
            correction_scales_u16.ctypes.data_as(u16p),
            scratch.ctypes.data_as(f32p),
            fused_output.ctypes.data_as(f32p),
        )
        base_scalar = native.library.cq3_hsm64_gemv_q8_row_scale4_scalar_trusted
        base_avx2 = native.library.cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_trusted
        fused_scalar = native.library.cq3_mhf_gemv_q8_row_scale4_scalar_trusted
        fused_avx2 = native.library.cq3_mhf_gemv_q8_row_scale4_avx2_bmi2_trusted
        if base_scalar(*base_args) != 0:
            raise RuntimeError("mask64 scalar reference failed")
        base_scalar_output = base_output.copy()
        if fused_scalar(*fused_args) != 0:
            raise RuntimeError("fused mask64/hash scalar reference failed")
        fused_scalar_output = fused_output.copy()
        if fused_avx2(*fused_args) != 0:
            raise RuntimeError("fused mask64/hash AVX2 correctness call failed")
        fused_avx2_output = fused_output.copy()
        if base_avx2(*base_args) != 0:
            raise RuntimeError("mask64 AVX2 correctness call failed")
        base_avx2_output = base_output.copy()

        dequantized = activation.astype(np.float32) * np.repeat(
            activation_scales, 128
        )
        expected_sketch = np.zeros(sketch_dim, dtype=np.float32)
        np.add.at(expected_sketch, buckets, dequantized * route_signs)
        explicit_output = base_scalar_output + (
            correction_codes.astype(np.float32) @ expected_sketch
        ) * correction_scales.astype(np.float32)
        errors = {
            "fused_scalar_vs_explicit": float(
                np.max(np.abs(fused_scalar_output - explicit_output))
            ),
            "fused_avx2_vs_fused_scalar": float(
                np.max(np.abs(fused_avx2_output - fused_scalar_output))
            ),
            "base_avx2_vs_base_scalar": float(
                np.max(np.abs(base_avx2_output - base_scalar_output))
            ),
        }
        if case_index % 2 == 0:
            fused_timing = timed(
                lambda: fused_avx2(*fused_args),
                warmups=args.warmups, repetitions=args.repetitions,
            )
            base_timing = timed(
                lambda: base_avx2(*base_args),
                warmups=args.warmups, repetitions=args.repetitions,
            )
        else:
            base_timing = timed(
                lambda: base_avx2(*base_args),
                warmups=args.warmups, repetitions=args.repetitions,
            )
            fused_timing = timed(
                lambda: fused_avx2(*fused_args),
                warmups=args.warmups, repetitions=args.repetitions,
            )
        original_weights = rows * columns
        fused_throughput = original_weights / fused_timing["median_ns"]
        base_throughput = original_weights / base_timing["median_ns"]
        storage = mask64_hash_component_storage(
            input_dim=columns, output_rows=rows, sketch_dim=sketch_dim
        )
        records.append(
            {
                "name": case["name"],
                "rows": rows,
                "columns": columns,
                "sketch_dim": sketch_dim,
                "original_weights": original_weights,
                "errors": errors,
                "fused_avx2_timing": fused_timing,
                "mask64_only_avx2_timing": base_timing,
                "fused_gweights_per_second": fused_throughput,
                "mask64_only_gweights_per_second": base_throughput,
                "fused_throughput_fraction_of_mask64_only": (
                    fused_throughput / base_throughput
                ),
                "complete_hash_component_storage": storage,
            }
        )

    maximum_error = max(max(record["errors"].values()) for record in records)
    minimum_fraction = min(
        record["fused_throughput_fraction_of_mask64_only"] for record in records
    )
    maximum_rate = max(
        record["complete_hash_component_storage"]["bits_per_weight"]
        for record in records
    )
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_cases": len(records),
        "observed_maximum_absolute_error": maximum_error,
        "observed_minimum_fused_throughput_fraction_of_mask64_only": minimum_fraction,
        "observed_maximum_complete_component_bits_per_weight": maximum_rate,
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_cases"])
        and maximum_error <= float(gate_spec["maximum_absolute_error"])
        and minimum_fraction
        >= float(gate_spec["minimum_fused_throughput_fraction_of_mask64_only"])
        and maximum_rate
        <= float(gate_spec["maximum_complete_component_bits_per_weight"])
    )
    receipt = {
        "schema": "qwen38_27b_mask64_hash_fused_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "library_sha256": sha256_file(args.library),
        "features": features,
        "cpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
        "records": records,
        "gate": gate,
        "claim_boundary": {
            "synthetic_exact_support_codes_used": True,
            "synthetic_learned_route_geometry_used": True,
            "full_Qwen_layer0_MLP_shapes_benchmarked": True,
            "ternary_and_hash_correction_executed_in_one_call": True,
            "real_learned_coefficients_or_model_weights_used": False,
            "model_token_throughput_or_capability_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
