#!/usr/bin/env python3
"""Compare the 1.5-bit mask64 GEMV with the aligned two-bit AVX2 control."""
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
    load_mask64_cpu_library,
    mask64_hash_component_storage,
    native_pack_mask64_matrix,
    pack_scale_indices4,
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
            raise RuntimeError(f"frozen mask64 GEMV source changed: {label}")


def timed(function, *, warmups: int, repetitions: int) -> dict:
    for _ in range(warmups):
        if function() != 0:
            raise RuntimeError("mask64 GEMV warmup failed")
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        status = function()
        elapsed = time.perf_counter_ns() - started
        if status != 0:
            raise RuntimeError(f"mask64 GEMV timed call failed: {status}")
        samples.append(elapsed)
    ordered = sorted(samples)
    return {
        "samples_ns": samples,
        "median_ns": float(statistics.median(ordered)),
        "minimum_ns": float(ordered[0]),
        "p90_ns": float(ordered[math.ceil(0.9 * len(ordered)) - 1]),
    }


def configure_ternary_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    library.cq3_cpu_features.argtypes = ()
    library.cq3_cpu_features.restype = ctypes.c_uint32
    library.cq3_pack_group128_2bit.argtypes = (
        i8p, ctypes.c_uint64, u8p,
    )
    library.cq3_pack_group128_2bit.restype = ctypes.c_int
    gemv = (
        u8p, u8p, u16p, u16p, i8p, f32p, i32p,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, f32p,
    )
    for implementation in ("scalar", "avx2"):
        function = getattr(
            library,
            f"cq3_gemv_q8_group128_2bit_row_scale4_{implementation}",
        )
        function.argtypes = gemv
        function.restype = ctypes.c_int
    return library


def make_codes(
    generator: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    codes = np.zeros((rows, columns), dtype=np.int8)
    groups = codes.reshape(-1, 128)
    # Build exact-k synthetic groups in bounded-memory batches. Calling
    # Generator.choice once per full Qwen matrix group would make input
    # construction dominate the actual frozen microkernel measurement.
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
    parser.add_argument("--mask-library", type=Path, required=True)
    parser.add_argument("--ternary-library", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-mask64-gemv-v1.json"),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 9 or args.repetitions % 2 == 0:
        raise ValueError("mask64 GEMV benchmark requires odd repetitions >= 9")

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_mask64_gemv_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    mask = load_mask64_cpu_library(args.mask_library)
    if mask is None:
        raise RuntimeError("mask64 CPU library did not load")
    features = mask.features()
    if not all(features.values()):
        raise RuntimeError("authoritative mask64 GEMV requires AVX2/FMA/F16C/BMI2")
    ternary = configure_ternary_library(args.ternary_library)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    records = []

    for case_index, case in enumerate(prereg["cases"]):
        rows = int(case["rows"])
        columns = int(case["columns"])
        groups = columns // 128
        generator = np.random.default_rng(
            int(prereg["measurement"]["seed"]) + case_index
        )
        codes = make_codes(generator, rows, columns)
        mask_payload = native_pack_mask64_matrix(codes, library=mask)
        aligned = np.empty(rows * columns // 4, dtype=np.uint8)
        pack_status = ternary.cq3_pack_group128_2bit(
            codes.ctypes.data_as(i8p), codes.size, aligned.ctypes.data_as(u8p)
        )
        if pack_status != 0:
            raise RuntimeError(f"aligned-two-bit pack failed: {pack_status}")
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
        mask_bytes = np.frombuffer(mask_payload.payload, dtype=np.uint8).copy()
        mask_output = np.empty(rows, dtype=np.float32)
        control_output = np.empty(rows, dtype=np.float32)
        anchors_u16 = anchors.view(np.uint16)
        ratios_u16 = ratios.view(np.uint16)
        common = (
            packed_indices.ctypes.data_as(u8p),
            anchors_u16.ctypes.data_as(u16p),
            ratios_u16.ctypes.data_as(u16p),
            activation.ctypes.data_as(i8p),
            activation_scales.ctypes.data_as(f32p),
            activation_sums.ctypes.data_as(i32p),
            rows, columns,
        )
        mask_args = (
            mask_bytes.ctypes.data_as(u8p), *common,
            mask_output.ctypes.data_as(f32p),
        )
        control_args = (
            aligned.ctypes.data_as(u8p), *common, 128,
            control_output.ctypes.data_as(f32p),
        )
        mask_scalar = mask.library.cq3_hsm64_gemv_q8_row_scale4_scalar_trusted
        mask_avx2 = mask.library.cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_trusted
        control_scalar = ternary.cq3_gemv_q8_group128_2bit_row_scale4_scalar
        control_avx2 = ternary.cq3_gemv_q8_group128_2bit_row_scale4_avx2
        if mask_scalar(*mask_args) != 0:
            raise RuntimeError("mask64 scalar correctness call failed")
        scalar_reference = mask_output.copy()
        if mask_avx2(*mask_args) != 0:
            raise RuntimeError("mask64 AVX2 correctness call failed")
        mask_avx_output = mask_output.copy()
        if control_scalar(*control_args) != 0:
            raise RuntimeError("aligned-two-bit scalar correctness call failed")
        control_scalar_output = control_output.copy()
        if control_avx2(*control_args) != 0:
            raise RuntimeError("aligned-two-bit AVX2 correctness call failed")
        mask_error = float(np.max(np.abs(mask_avx_output - scalar_reference)))
        control_error = float(
            np.max(np.abs(mask_avx_output - control_output))
        )
        scalar_control_error = float(
            np.max(np.abs(scalar_reference - control_scalar_output))
        )

        scalar_rows = min(rows, int(prereg["measurement"]["scalar_rows"]))
        scalar_mask_args = (
            mask_bytes.ctypes.data_as(u8p),
            packed_indices.ctypes.data_as(u8p),
            anchors_u16.ctypes.data_as(u16p),
            ratios_u16.ctypes.data_as(u16p),
            activation.ctypes.data_as(i8p),
            activation_scales.ctypes.data_as(f32p),
            activation_sums.ctypes.data_as(i32p),
            scalar_rows, columns,
            mask_output.ctypes.data_as(f32p),
        )
        scalar_timing = timed(
            lambda: mask_scalar(*scalar_mask_args),
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        if case_index % 2 == 0:
            mask_timing = timed(
                lambda: mask_avx2(*mask_args),
                warmups=args.warmups,
                repetitions=args.repetitions,
            )
            control_timing = timed(
                lambda: control_avx2(*control_args),
                warmups=args.warmups,
                repetitions=args.repetitions,
            )
        else:
            control_timing = timed(
                lambda: control_avx2(*control_args),
                warmups=args.warmups,
                repetitions=args.repetitions,
            )
            mask_timing = timed(
                lambda: mask_avx2(*mask_args),
                warmups=args.warmups,
                repetitions=args.repetitions,
            )
        weights = rows * columns
        mask_throughput = weights / mask_timing["median_ns"]
        control_throughput = weights / control_timing["median_ns"]
        scalar_throughput = scalar_rows * columns / scalar_timing["median_ns"]
        storage = mask64_hash_component_storage(
            input_dim=columns,
            output_rows=rows,
            sketch_dim=columns // 128,
        )
        records.append(
            {
                "name": case["name"],
                "rows": rows,
                "columns": columns,
                "weights": weights,
                "mask64_payload_bytes": len(mask_payload.payload),
                "aligned2_payload_bytes": aligned.size,
                "mask64_code_bits_per_weight": len(mask_payload.payload) * 8 / weights,
                "complete_hash_component_storage": storage,
                "maximum_absolute_error_mask_avx2_vs_mask_scalar": mask_error,
                "maximum_absolute_error_mask_avx2_vs_aligned2_avx2": control_error,
                "maximum_absolute_error_mask_scalar_vs_aligned2_scalar": scalar_control_error,
                "mask64_scalar_subset_timing": scalar_timing,
                "mask64_avx2_timing": mask_timing,
                "aligned2_avx2_timing": control_timing,
                "mask64_gweights_per_second": mask_throughput,
                "aligned2_gweights_per_second": control_throughput,
                "mask64_throughput_fraction_of_aligned2": (
                    mask_throughput / control_throughput
                ),
                "mask64_avx2_speedup_vs_scalar_subset": (
                    mask_throughput / scalar_throughput
                ),
            }
        )

    maximum_error = max(
        max(
            record["maximum_absolute_error_mask_avx2_vs_mask_scalar"],
            record["maximum_absolute_error_mask_avx2_vs_aligned2_avx2"],
            record["maximum_absolute_error_mask_scalar_vs_aligned2_scalar"],
        )
        for record in records
    )
    minimum_fraction = min(
        record["mask64_throughput_fraction_of_aligned2"] for record in records
    )
    minimum_speedup = min(
        record["mask64_avx2_speedup_vs_scalar_subset"] for record in records
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
        "observed_minimum_throughput_fraction_of_aligned2": minimum_fraction,
        "observed_minimum_avx2_speedup_vs_scalar": minimum_speedup,
        "observed_maximum_complete_component_bits_per_weight": maximum_rate,
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_cases"])
        and maximum_error <= float(gate_spec["maximum_absolute_error"])
        and minimum_fraction
        >= float(gate_spec["minimum_throughput_fraction_of_aligned2"])
        and minimum_speedup >= float(gate_spec["minimum_avx2_speedup_vs_scalar"])
        and maximum_rate
        <= float(gate_spec["maximum_complete_component_bits_per_weight"])
    )
    receipt = {
        "schema": "qwen38_27b_mask64_gemv_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "mask_library_sha256": sha256_file(args.mask_library),
        "ternary_library_sha256": sha256_file(args.ternary_library),
        "features": features,
        "cpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
        "records": records,
        "gate": gate,
        "claim_boundary": {
            "synthetic_exact_support_codes_used": True,
            "full_Qwen_layer0_MLP_shapes_benchmarked": True,
            "mask64_direct_GEMV_arithmetic_proven": gate["passed"],
            "learned_hash_correction_fused_in_same_call": False,
            "real_model_token_throughput_or_capability_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
