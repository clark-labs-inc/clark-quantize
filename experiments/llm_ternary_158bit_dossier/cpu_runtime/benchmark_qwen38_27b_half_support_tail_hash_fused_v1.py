#!/usr/bin/env python3
"""Benchmark one-call vision-tail ternary GEMV plus learned-hash correction."""
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
    native_pack_half_support_tail_matrix,
)
from clark_quantize.qwen38_half_support_tail_hash_fused_cpu import (
    load_half_support_tail_hash_fused_cpu_library,
)
from clark_quantize.qwen38_hash_residual_cpu import pack_signed_int4_rows
from clark_quantize.qwen38_storage_budget import align_up


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
            raise RuntimeError(f"frozen tail/hash source changed: {label}")


def interleaved_timed(
    base, fused, *, warmups: int, repetitions: int
) -> tuple[dict, dict]:
    for _ in range(warmups):
        if base() != 0 or fused() != 0:
            raise RuntimeError("fused tail/hash warmup failed")
    base_samples: list[int] = []
    fused_samples: list[int] = []
    for repetition in range(repetitions):
        functions = ((base, base_samples), (fused, fused_samples))
        if repetition % 2:
            functions = tuple(reversed(functions))
        for function, samples in functions:
            started = time.perf_counter_ns()
            status = function()
            elapsed = time.perf_counter_ns() - started
            if status != 0:
                raise RuntimeError(f"fused tail/hash timed call failed: {status}")
            samples.append(elapsed)

    def summarize(samples: list[int]) -> dict:
        ordered = sorted(samples)
        return {
            "samples_ns": samples,
            "median_ns": float(statistics.median(ordered)),
            "minimum_ns": float(ordered[0]),
            "p90_ns": float(ordered[math.ceil(0.9 * len(ordered)) - 1]),
        }

    return summarize(base_samples), summarize(fused_samples)


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


def complete_component_storage(
    *, rows: int, columns: int, sketch_dim: int, payload_bytes: int
) -> dict:
    groups = (columns + 127) // 128
    bucket_bits = max(1, math.ceil(math.log2(sketch_dim)))
    raw = {
        "half_support_mask_and_sign_codes": payload_bytes,
        "scale4_indices_row_anchors_and_codebook": (
            (rows * groups * 4 + 7) // 8 + rows * 2 + 16 * 2
        ),
        "signed_int4_hash_coefficients_and_row_scales": (
            rows * ((sketch_dim + 1) // 2) + rows * 2
        ),
        "bitpacked_learned_route": (
            columns * (bucket_bits + 1) + 7
        ) // 8,
        "tensor_header": 64,
    }
    aligned = {name: align_up(value) for name, value in raw.items()}
    total = sum(aligned.values())
    return {
        "raw_bytes": raw,
        "aligned_bytes": aligned,
        "complete_bytes": total,
        "bits_per_weight": total * 8 / (rows * columns),
        "route_sharing_assumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-half-support-tail-hash-fused-v1.json"
        ),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 9 or args.repetitions % 2 == 0:
        raise ValueError("fused tail/hash benchmark requires odd repetitions >= 9")

    runtime = Path(__file__).resolve().parent
    prereg_path = (
        runtime / "qwen38_27b_half_support_tail_hash_fused_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    native = load_half_support_tail_hash_fused_cpu_library(args.library)
    if native is None:
        raise RuntimeError("fused half-support tail/hash CPU library did not load")
    features = native.features()
    if not all(features.values()):
        raise RuntimeError("authoritative fused tail/hash run requires AVX2/FMA/F16C/BMI2")

    case = prereg["case"]
    rows = int(case["rows"])
    columns = int(case["columns"])
    sketch_dim = int(case["sketch_dim"])
    groups = (columns + 127) // 128
    generator = np.random.default_rng(int(prereg["measurement"]["seed"]))
    codes = make_codes(generator, rows, columns)
    encoded = native_pack_half_support_tail_matrix(codes, library=native.tail)
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
    activation_sums = np.empty(groups, dtype=np.int32)
    for group in range(groups):
        start = group * 128
        activation_sums[group] = activation[start : start + 128].sum(dtype=np.int32)
    buckets = generator.integers(0, sketch_dim, size=columns, dtype=np.uint16)
    route_signs = generator.integers(0, 2, size=columns, dtype=np.int8)
    route_signs = route_signs * np.int8(2) - np.int8(1)
    correction_codes = generator.integers(
        -7, 8, size=(rows, sketch_dim), dtype=np.int8
    )
    packed_correction = pack_signed_int4_rows(correction_codes)
    correction_scales = generator.uniform(1e-4, 0.01, size=rows).astype(
        np.float16
    )
    sketch_scratch = np.empty(sketch_dim, dtype=np.float32)
    correction_scratch = np.empty(rows, dtype=np.float32)
    base_output = np.empty(rows, dtype=np.float32)
    fused_output = np.empty(rows, dtype=np.float32)

    u8p = ctypes.POINTER(ctypes.c_uint8)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
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
        sketch_scratch.ctypes.data_as(f32p),
        correction_scratch.ctypes.data_as(f32p),
        fused_output.ctypes.data_as(f32p),
    )
    base_scalar = native.library.cq3_hst_gemv_q8_row_scale4_scalar_trusted
    base_avx2 = native.library.cq3_hst_gemv_q8_row_scale4_avx2_bmi2_trusted
    fused_scalar = native.library.cq3_htf_gemv_q8_row_scale4_scalar_trusted
    fused_avx2 = native.library.cq3_htf_gemv_q8_row_scale4_avx2_bmi2_trusted
    if base_scalar(*base_args) != 0:
        raise RuntimeError("half-support tail scalar reference failed")
    base_scalar_output = base_output.copy()
    if fused_scalar(*fused_args) != 0:
        raise RuntimeError("fused tail/hash scalar reference failed")
    fused_scalar_output = fused_output.copy()
    if fused_avx2(*fused_args) != 0:
        raise RuntimeError("fused tail/hash AVX2 correctness call failed")
    fused_avx2_output = fused_output.copy()
    if base_avx2(*base_args) != 0:
        raise RuntimeError("half-support tail AVX2 correctness call failed")
    base_avx2_output = base_output.copy()

    dequantized = activation.astype(np.float32) * np.repeat(
        activation_scales, 128
    )[:columns]
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
    base_timing, fused_timing = interleaved_timed(
        lambda: base_avx2(*base_args),
        lambda: fused_avx2(*fused_args),
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    logical_weights = rows * columns
    base_throughput = logical_weights / base_timing["median_ns"]
    fused_throughput = logical_weights / fused_timing["median_ns"]
    storage = complete_component_storage(
        rows=rows,
        columns=columns,
        sketch_dim=sketch_dim,
        payload_bytes=len(encoded.payload),
    )
    measurements = {
        "payload_bytes": len(encoded.payload),
        "raw_code_bits_per_weight": len(encoded.payload) * 8 / logical_weights,
        "errors": errors,
        "base_tail_avx2_timing": base_timing,
        "fused_tail_hash_avx2_timing": fused_timing,
        "base_tail_gweights_per_second": base_throughput,
        "fused_tail_hash_gweights_per_second": fused_throughput,
        "fused_throughput_fraction_of_tail_only": (
            fused_throughput / base_throughput
        ),
        "complete_component_storage": storage,
    }
    maximum_error = max(errors.values())
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_maximum_absolute_error": maximum_error,
        "observed_fused_throughput_fraction_of_tail_only": (
            measurements["fused_throughput_fraction_of_tail_only"]
        ),
        "observed_complete_component_bits_per_weight": storage["bits_per_weight"],
    }
    gate["passed"] = bool(
        maximum_error <= float(gate_spec["maximum_absolute_error"])
        and measurements["fused_throughput_fraction_of_tail_only"]
        >= float(gate_spec["minimum_fused_throughput_fraction_of_tail_only"])
        and storage["bits_per_weight"]
        <= float(gate_spec["maximum_complete_component_bits_per_weight"])
    )
    receipt = {
        "schema": "qwen38_27b_half_support_tail_hash_fused_receipt_v1",
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
        "case": case,
        "measurements": measurements,
        "gate": gate,
        "claim_boundary": {
            "synthetic_exact_support_codes_used": True,
            "synthetic_learned_route_geometry_used": True,
            "real_Qwen_vision_fc2_shape_used": True,
            "ternary_tail_and_hash_correction_executed_in_one_call": True,
            "real_learned_coefficients_or_model_weights_used": False,
            "model_token_throughput_or_capability_proven": False,
            "semi_lossless_compression_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
