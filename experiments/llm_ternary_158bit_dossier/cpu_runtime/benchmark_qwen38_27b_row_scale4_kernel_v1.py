#!/usr/bin/env python3
"""Benchmark the Qwen3.8-27B row-scale4 fused AVX2 microkernel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_row_scale_codebook import (
    reconstruct_row_normalized_scales,
)
from clark_quantize.scale_codebook import pack_nibbles
from clark_quantize.ternary_cpu import (
    cpu_features,
    decode_row_scale4,
    gemv_q8_group128_2bit,
    gemv_q8_group128_2bit_row_scale4,
    native_available,
    quantize_q8,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(function, repetitions: int) -> tuple[list[float], np.ndarray]:
    samples = []
    output = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = function()
        elapsed = (time.perf_counter_ns() - started) / 1e9
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("non-finite kernel timing")
        samples.append(elapsed)
    assert output is not None
    return samples, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-row-scale4-kernel-v1.json"
        ),
    )
    parser.add_argument("--rows", type=int, default=17_408)
    parser.add_argument("--columns", type=int, default=5_120)
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_row_scale4_kernel_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    frozen = prereg["scope"]
    if args.rows != frozen["rows"] or args.columns != frozen["columns"]:
        raise ValueError("authoritative benchmark dimensions are frozen to Qwen3.8-27B")
    if not native_available():
        raise SystemExit("infrastructure-invalid: native ternary CPU library is absent")
    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if not native_library or not Path(native_library).is_file():
        raise SystemExit("infrastructure-invalid: native library path is not pinned")
    features = cpu_features()
    if not all(features[name] for name in ("avx2", "f16c", "fma")):
        raise SystemExit("infrastructure-invalid: AVX2/F16C/FMA are required")
    if len(os.sched_getaffinity(0)) != 1:
        raise SystemExit("infrastructure-invalid: benchmark must be pinned to one CPU")

    rows = args.rows
    columns = args.columns
    groups = columns // 128
    generator = np.random.default_rng(int(prereg["candidate"]["seed"]))
    codes = generator.integers(-1, 2, size=(rows, columns), dtype=np.int8)
    shaped = codes.reshape(rows, groups, 4, 32).astype(np.uint8) + np.uint8(1)
    shifts = np.asarray([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    packed_weights = np.bitwise_or.reduce(shaped << shifts, axis=2).reshape(
        rows, columns // 4
    )
    anchors = generator.lognormal(-2.0, 0.55, size=rows).astype(np.float16)
    codebook = np.concatenate(
        (
            np.zeros(1, dtype=np.float16),
            np.geomspace(0.45, 2.2, 15, dtype=np.float32).astype(np.float16),
        )
    )
    scale_codes = generator.integers(0, 16, size=(rows, groups), dtype=np.uint8)
    packed_scale_codes = np.frombuffer(
        pack_nibbles(scale_codes), dtype=np.uint8
    ).copy()
    candidate_scales = reconstruct_row_normalized_scales(
        anchors, codebook, scale_codes
    )
    materialized_control_scales = candidate_scales.astype(np.float16)
    activation = generator.normal(size=columns).astype(np.float32)
    activation_q8, activation_scales, activation_sums = quantize_q8(activation)

    candidate = lambda: gemv_q8_group128_2bit_row_scale4(
        packed_weights,
        packed_scale_codes,
        anchors,
        codebook,
        activation_q8,
        activation_scales,
        activation_sums,
        implementation="avx2",
        validated=True,
    )
    control = lambda: gemv_q8_group128_2bit(
        packed_weights,
        materialized_control_scales,
        activation_q8,
        activation_scales,
        activation_sums,
        implementation="avx2",
        validated=True,
    )
    for _ in range(int(prereg["measurement"]["warmups"])):
        candidate()
        control()

    candidate_samples: list[float] = []
    control_samples: list[float] = []
    candidate_output = control_output = None
    for repetition in range(int(prereg["measurement"]["paired_repetitions"])):
        order = ((candidate, candidate_samples), (control, control_samples))
        if repetition & 1:
            order = tuple(reversed(order))
        for function, samples in order:
            started = time.perf_counter_ns()
            output = function()
            elapsed = (time.perf_counter_ns() - started) / 1e9
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("non-finite paired timing")
            samples.append(elapsed)
            if function is candidate:
                candidate_output = output
            else:
                control_output = output
    assert candidate_output is not None and control_output is not None

    subset_rows = int(prereg["measurement"]["scalar_correctness_rows"])
    subset_count = subset_rows * groups
    subset_indices = packed_scale_codes[: (subset_count + 1) // 2].copy()
    scalar_subset = lambda: gemv_q8_group128_2bit_row_scale4(
        packed_weights[:subset_rows],
        subset_indices,
        anchors[:subset_rows],
        codebook,
        activation_q8,
        activation_scales,
        activation_sums,
        implementation="scalar",
        validated=True,
    )
    avx_subset = lambda: gemv_q8_group128_2bit_row_scale4(
        packed_weights[:subset_rows],
        subset_indices,
        anchors[:subset_rows],
        codebook,
        activation_q8,
        activation_scales,
        activation_sums,
        implementation="avx2",
        validated=True,
    )
    scalar_samples, scalar_output = measure(scalar_subset, 5)
    avx_subset_samples, avx_subset_output = measure(avx_subset, 5)
    maximum_absolute_error = float(
        np.max(np.abs(scalar_output - avx_subset_output), initial=0.0)
    )

    decode_repetitions = int(prereg["measurement"]["standalone_decode_repetitions"])
    scalar_decode = lambda: decode_row_scale4(
        packed_scale_codes, anchors, codebook, groups,
        implementation="scalar", validated=True,
    )
    avx_decode = lambda: decode_row_scale4(
        packed_scale_codes, anchors, codebook, groups,
        implementation="avx2", validated=True,
    )
    scalar_decode_samples, scalar_decoded = measure(scalar_decode, decode_repetitions)
    avx_decode_samples, avx_decoded = measure(avx_decode, decode_repetitions)
    exact_scalar_decode = bool(np.array_equal(scalar_decoded, candidate_scales))
    exact_avx_decode = bool(np.array_equal(avx_decoded, candidate_scales))

    candidate_median = statistics.median(candidate_samples)
    control_median = statistics.median(control_samples)
    throughput_fraction = control_median / candidate_median
    scalar_speedup = statistics.median(scalar_samples) / statistics.median(
        avx_subset_samples
    )
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_candidate_avx2_vs_candidate_scalar_absolute_error": (
            maximum_absolute_error
        ),
        "observed_exact_scalar_scale_decode": exact_scalar_decode,
        "observed_exact_avx2_scale_decode": exact_avx_decode,
        "observed_candidate_throughput_fraction_of_materialized_fp16_control": (
            throughput_fraction
        ),
        "observed_candidate_avx2_speedup_over_candidate_scalar_subset": (
            scalar_speedup
        ),
    }
    gate["passed"] = (
        maximum_absolute_error
        <= gate_spec["maximum_candidate_avx2_vs_candidate_scalar_absolute_error"]
        and exact_scalar_decode
        and exact_avx_decode
        and throughput_fraction
        >= gate_spec[
            "minimum_candidate_throughput_fraction_of_materialized_fp16_control"
        ]
        and scalar_speedup
        >= gate_spec["minimum_candidate_avx2_speedup_over_candidate_scalar_subset"]
    )
    receipt = {
        "schema": "qwen38_27b_row_scale4_kernel_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "native_library": native_library,
        "native_library_sha256": sha256_file(Path(native_library)),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_features": features,
        "shape": {"rows": rows, "columns": columns, "groups": groups},
        "storage_bytes": {
            "materialized_fp16_control_scales": materialized_control_scales.nbytes,
            "candidate_packed_scale_indices": packed_scale_codes.nbytes,
            "candidate_fp16_row_anchors": anchors.nbytes,
            "candidate_fp16_ratio_codebook": codebook.nbytes,
        },
        "paired_gemv": {
            "candidate_samples_seconds": candidate_samples,
            "control_samples_seconds": control_samples,
            "candidate_median_seconds": candidate_median,
            "control_median_seconds": control_median,
            "candidate_rows_per_second": rows / candidate_median,
            "control_rows_per_second": rows / control_median,
            "candidate_throughput_fraction_of_control": throughput_fraction,
            "candidate_output_finite": bool(np.all(np.isfinite(candidate_output))),
            "control_output_finite": bool(np.all(np.isfinite(control_output))),
        },
        "candidate_correctness_subset": {
            "rows": subset_rows,
            "scalar_samples_seconds": scalar_samples,
            "avx2_samples_seconds": avx_subset_samples,
            "maximum_absolute_error": maximum_absolute_error,
            "avx2_speedup_over_scalar": scalar_speedup,
        },
        "standalone_scale_decode": {
            "scales": int(candidate_scales.size),
            "scalar_samples_seconds": scalar_decode_samples,
            "avx2_samples_seconds": avx_decode_samples,
            "scalar_median_scales_per_second": candidate_scales.size
            / statistics.median(scalar_decode_samples),
            "avx2_median_scales_per_second": candidate_scales.size
            / statistics.median(avx_decode_samples),
            "exact_scalar": exact_scalar_decode,
            "exact_avx2": exact_avx_decode,
        },
        "gate": gate,
        "claim_boundary": {
            "qwen38_exact_shape_scale_layout_microkernel_only": True,
            "base3_direct_decode_proven": False,
            "model_quality_proven": False,
            "end_to_end_tokens_per_second_proven": False,
            "scientific_breakthrough_proven": False
        }
    }
    if not (
        receipt["paired_gemv"]["candidate_output_finite"]
        and receipt["paired_gemv"]["control_output_finite"]
    ):
        raise RuntimeError("kernel produced non-finite output")
    atomic_bytes(args.receipt, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
