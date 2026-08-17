#!/usr/bin/env python3
"""Benchmark exact half-support coding and direct Qwen vision-tail GEMV."""
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
from clark_quantize.qwen38_fixed_support_mask64_cpu import pack_scale_indices4
from clark_quantize.qwen38_half_support_tail_cpu import (
    load_half_support_tail_cpu_library,
    native_pack_half_support_tail_matrix,
    native_unpack_half_support_tail_matrix,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources(preregistration: dict) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        source = REPO / descriptor["path"]
        if not source.is_file() or sha256_file(source) != descriptor["sha256"]:
            raise RuntimeError(f"frozen half-support tail source changed: {label}")


def timed(function, *, warmups: int, repetitions: int) -> dict:
    for _ in range(warmups):
        if function() != 0:
            raise RuntimeError("half-support tail warmup failed")
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        status = function()
        elapsed = time.perf_counter_ns() - started
        if status != 0:
            raise RuntimeError(f"half-support tail timed call failed: {status}")
        samples.append(elapsed)
    ordered = sorted(samples)
    return {
        "samples_ns": samples,
        "median_ns": float(statistics.median(ordered)),
        "minimum_ns": float(ordered[0]),
        "p90_ns": float(ordered[math.ceil(0.9 * len(ordered)) - 1]),
    }


def configure_control(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
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
    for start in range(0, columns, 128):
        width = min(128, columns - start)
        block = codes[:, start : start + width]
        for batch_start in range(0, rows, 256):
            batch_stop = min(batch_start + 256, rows)
            count = batch_stop - batch_start
            scores = generator.random((count, width), dtype=np.float32)
            positions = np.argpartition(scores, width // 2, axis=1)[:, : width // 2]
            signs = generator.integers(
                0, 2, size=(count, width // 2), dtype=np.int8
            )
            signs = signs * np.int8(2) - np.int8(1)
            block[batch_start:batch_stop][
                np.arange(count)[:, None], positions
            ] = signs
    return codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-library", type=Path, required=True)
    parser.add_argument("--control-library", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-half-support-tail-v1.json"),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 9 or args.repetitions % 2 == 0:
        raise ValueError("half-support tail benchmark requires odd repetitions >= 9")

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_half_support_tail_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    native = load_half_support_tail_cpu_library(args.tail_library)
    if native is None:
        raise RuntimeError("half-support tail CPU library did not load")
    features = native.features()
    if not all(features.values()):
        raise RuntimeError("authoritative tail run requires AVX2/FMA/F16C/BMI2")
    control = configure_control(args.control_library)
    case = prereg["case"]
    rows = int(case["rows"])
    columns = int(case["columns"])
    padded_columns = int(case["padded_control_columns"])
    groups = (columns + 127) // 128
    generator = np.random.default_rng(int(prereg["measurement"]["seed"]))
    codes = make_codes(generator, rows, columns)

    started = time.perf_counter_ns()
    encoded = native_pack_half_support_tail_matrix(codes, library=native)
    pack_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    decoded = native_unpack_half_support_tail_matrix(encoded, library=native)
    unpack_ns = time.perf_counter_ns() - started
    exact_roundtrip = bool(np.array_equal(decoded, codes))
    tail_weights = np.frombuffer(encoded.payload, dtype=np.uint8).copy()

    padded_codes = np.zeros((rows, padded_columns), dtype=np.int8)
    padded_codes[:, :columns] = codes
    aligned = np.empty(rows * padded_columns // 4, dtype=np.uint8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    if control.cq3_pack_group128_2bit(
        padded_codes.ctypes.data_as(i8p), padded_codes.size,
        aligned.ctypes.data_as(u8p),
    ) != 0:
        raise RuntimeError("padded aligned-two-bit control pack failed")
    indices = generator.integers(1, 16, size=(rows, groups), dtype=np.uint8)
    packed_indices = pack_scale_indices4(indices)
    anchors = generator.uniform(0.001, 0.02, size=rows).astype(np.float16)
    ratios = generator.uniform(0.5, 1.5, size=16).astype(np.float16)
    ratios[0] = np.float16(0.0)
    activation = generator.integers(-127, 128, size=columns, dtype=np.int8)
    padded_activation = np.zeros(padded_columns, dtype=np.int8)
    padded_activation[:columns] = activation
    activation_scales = generator.uniform(0.002, 0.03, size=groups).astype(
        np.float32
    )
    activation_sums = np.empty(groups, dtype=np.int32)
    for group in range(groups):
        start = group * 128
        activation_sums[group] = activation[start : start + 128].sum(dtype=np.int32)
    tail_output = np.empty(rows, dtype=np.float32)
    control_output = np.empty(rows, dtype=np.float32)
    anchors_u16 = anchors.view(np.uint16)
    ratios_u16 = ratios.view(np.uint16)
    tail_args = (
        tail_weights.ctypes.data_as(u8p),
        packed_indices.ctypes.data_as(u8p),
        anchors_u16.ctypes.data_as(u16p),
        ratios_u16.ctypes.data_as(u16p),
        activation.ctypes.data_as(i8p),
        activation_scales.ctypes.data_as(f32p),
        activation_sums.ctypes.data_as(i32p),
        rows, columns,
        tail_output.ctypes.data_as(f32p),
    )
    control_args = (
        aligned.ctypes.data_as(u8p),
        packed_indices.ctypes.data_as(u8p),
        anchors_u16.ctypes.data_as(u16p),
        ratios_u16.ctypes.data_as(u16p),
        padded_activation.ctypes.data_as(i8p),
        activation_scales.ctypes.data_as(f32p),
        activation_sums.ctypes.data_as(i32p),
        rows, padded_columns, 128,
        control_output.ctypes.data_as(f32p),
    )
    tail_scalar = native.library.cq3_hst_gemv_q8_row_scale4_scalar_trusted
    tail_avx2 = native.library.cq3_hst_gemv_q8_row_scale4_avx2_bmi2_trusted
    control_avx2 = control.cq3_gemv_q8_group128_2bit_row_scale4_avx2
    if tail_scalar(*tail_args) != 0:
        raise RuntimeError("half-support tail scalar correctness call failed")
    scalar_output = tail_output.copy()
    if tail_avx2(*tail_args) != 0:
        raise RuntimeError("half-support tail AVX2 correctness call failed")
    avx2_output = tail_output.copy()
    if control_avx2(*control_args) != 0:
        raise RuntimeError("padded aligned-two-bit AVX2 correctness call failed")
    maximum_error = max(
        float(np.max(np.abs(avx2_output - scalar_output))),
        float(np.max(np.abs(avx2_output - control_output))),
    )

    scalar_rows = int(prereg["measurement"]["scalar_rows"])
    scalar_tail_args = (
        tail_weights.ctypes.data_as(u8p),
        packed_indices.ctypes.data_as(u8p),
        anchors_u16.ctypes.data_as(u16p),
        ratios_u16.ctypes.data_as(u16p),
        activation.ctypes.data_as(i8p),
        activation_scales.ctypes.data_as(f32p),
        activation_sums.ctypes.data_as(i32p),
        scalar_rows, columns,
        tail_output.ctypes.data_as(f32p),
    )
    scalar_timing = timed(
        lambda: tail_scalar(*scalar_tail_args),
        warmups=args.warmups, repetitions=args.repetitions,
    )
    tail_timing = timed(
        lambda: tail_avx2(*tail_args),
        warmups=args.warmups, repetitions=args.repetitions,
    )
    control_timing = timed(
        lambda: control_avx2(*control_args),
        warmups=args.warmups, repetitions=args.repetitions,
    )
    logical_weights = rows * columns
    tail_throughput = logical_weights / tail_timing["median_ns"]
    control_logical_throughput = logical_weights / control_timing["median_ns"]
    scalar_throughput = scalar_rows * columns / scalar_timing["median_ns"]
    measurements = {
        "exact_roundtrip": exact_roundtrip,
        "payload_bytes": len(encoded.payload),
        "raw_code_bits_per_weight": len(encoded.payload) * 8 / logical_weights,
        "pack_mweights_per_second": logical_weights / pack_ns / 1e-3,
        "unpack_mweights_per_second": logical_weights / unpack_ns / 1e-3,
        "maximum_absolute_error": maximum_error,
        "tail_avx2_timing": tail_timing,
        "padded_aligned2_avx2_timing": control_timing,
        "tail_scalar_subset_timing": scalar_timing,
        "tail_gweights_per_second": tail_throughput,
        "padded_control_logical_gweights_per_second": control_logical_throughput,
        "tail_throughput_fraction_of_padded_aligned2": (
            tail_throughput / control_logical_throughput
        ),
        "tail_avx2_speedup_vs_scalar": tail_throughput / scalar_throughput,
    }
    gate_spec = prereg["gate"]
    gate = {**gate_spec, **{f"observed_{key}": value for key, value in measurements.items() if isinstance(value, (bool, int, float))}}
    gate["passed"] = bool(
        exact_roundtrip
        and measurements["raw_code_bits_per_weight"]
        <= float(gate_spec["maximum_raw_code_bits_per_weight"])
        and measurements["pack_mweights_per_second"]
        >= float(gate_spec["minimum_pack_mweights_per_second"])
        and measurements["unpack_mweights_per_second"]
        >= float(gate_spec["minimum_unpack_mweights_per_second"])
        and maximum_error <= float(gate_spec["maximum_absolute_error"])
        and measurements["tail_throughput_fraction_of_padded_aligned2"]
        >= float(gate_spec["minimum_throughput_fraction_of_padded_aligned2"])
        and measurements["tail_avx2_speedup_vs_scalar"]
        >= float(gate_spec["minimum_avx2_speedup_vs_scalar"])
    )
    receipt = {
        "schema": "qwen38_27b_half_support_tail_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "tail_library_sha256": sha256_file(args.tail_library),
        "control_library_sha256": sha256_file(args.control_library),
        "features": features,
        "cpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
        "case": case,
        "measurements": measurements,
        "gate": gate,
        "claim_boundary": {
            "synthetic_exact_support_codes_used": True,
            "real_Qwen_vision_fc2_shape_used": True,
            "exact_tail_codec_and_arithmetic_proven": gate["passed"],
            "learned_hash_correction_fused_for_tail": False,
            "real_model_weights_or_capability_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
