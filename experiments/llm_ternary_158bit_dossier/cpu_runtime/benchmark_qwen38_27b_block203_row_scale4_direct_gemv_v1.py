#!/usr/bin/env python3
"""Benchmark direct block203 plus row-scale4 GEMV for Qwen3.8-27B."""
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
from clark_quantize.scale_codebook import pack_nibbles
from clark_quantize.ternary_cpu import (
    cpu_features,
    gemv_q8_group128_2bit_row_scale4,
    gemv_q8_group128_block203_row_scale4,
    native_available,
    pack_block203_rows,
    quantize_q8,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(function, repetitions: int) -> tuple[list[float], np.ndarray]:
    samples: list[float] = []
    output = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = function()
        elapsed = (time.perf_counter_ns() - started) / 1e9
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("non-finite direct-kernel timing")
        samples.append(elapsed)
    assert output is not None
    return samples, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-block203-row-scale4-direct-gemv-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / (
        "qwen38_27b_block203_row_scale4_direct_gemv_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if (
        not native_available()
        or not native_library
        or not Path(native_library).is_file()
    ):
        raise SystemExit("infrastructure-invalid: pinned native library is absent")
    features = cpu_features()
    if not all(features[name] for name in ("avx2", "f16c", "fma")):
        raise SystemExit("infrastructure-invalid: AVX2/F16C/FMA are required")
    if len(os.sched_getaffinity(0)) != 1:
        raise SystemExit("infrastructure-invalid: benchmark must be pinned to one CPU")

    scope = prereg["scope"]
    rows = int(scope["rows"])
    columns = int(scope["columns"])
    groups = int(scope["groups_per_row"])
    generator = np.random.default_rng(int(prereg["candidate"]["seed"]))
    codes = generator.integers(-1, 2, size=(rows, columns), dtype=np.int8)
    block203_weights = pack_block203_rows(codes)
    if block203_weights.nbytes != int(scope["base3_weight_payload_bytes"]):
        raise RuntimeError("block203 payload byte count differs from preregistration")
    shaped = codes.reshape(rows, groups, 4, 32).astype(np.uint8) + np.uint8(1)
    shifts = np.asarray([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    aligned_weights = np.bitwise_or.reduce(shaped << shifts, axis=2).reshape(
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
    activation = generator.normal(size=columns).astype(np.float32)
    activation_q8, activation_scales, activation_sums = quantize_q8(activation)

    direct = lambda: gemv_q8_group128_block203_row_scale4(
        block203_weights,
        packed_scale_codes,
        anchors,
        codebook,
        activation_q8,
        activation_scales,
        activation_sums,
        implementation="avx2",
        validated=True,
    )
    control = lambda: gemv_q8_group128_2bit_row_scale4(
        aligned_weights,
        packed_scale_codes,
        anchors,
        codebook,
        activation_q8,
        activation_scales,
        activation_sums,
        implementation="avx2",
        validated=True,
    )
    for _ in range(int(prereg["measurement"]["warmups"])):
        direct()
        control()

    direct_samples: list[float] = []
    control_samples: list[float] = []
    direct_output = control_output = None
    for repetition in range(int(prereg["measurement"]["paired_repetitions"])):
        order = ((direct, direct_samples), (control, control_samples))
        if repetition & 1:
            order = tuple(reversed(order))
        for function, samples in order:
            started = time.perf_counter_ns()
            output = function()
            elapsed = (time.perf_counter_ns() - started) / 1e9
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("non-finite paired direct timing")
            samples.append(elapsed)
            if function is direct:
                direct_output = output
            else:
                control_output = output
    assert direct_output is not None and control_output is not None

    subset_rows = int(prereg["measurement"]["scalar_correctness_rows"])
    subset_groups = subset_rows * groups
    subset_indices = packed_scale_codes[: (subset_groups + 1) // 2].copy()
    subset_block = pack_block203_rows(codes[:subset_rows])
    direct_scalar = lambda: gemv_q8_group128_block203_row_scale4(
        subset_block,
        subset_indices,
        anchors[:subset_rows],
        codebook,
        activation_q8,
        activation_scales,
        activation_sums,
        implementation="scalar",
        validated=True,
    )
    direct_avx2 = lambda: gemv_q8_group128_block203_row_scale4(
        subset_block,
        subset_indices,
        anchors[:subset_rows],
        codebook,
        activation_q8,
        activation_scales,
        activation_sums,
        implementation="avx2",
        validated=True,
    )
    aligned_scalar = lambda: gemv_q8_group128_2bit_row_scale4(
        aligned_weights[:subset_rows],
        subset_indices,
        anchors[:subset_rows],
        codebook,
        activation_q8,
        activation_scales,
        activation_sums,
        implementation="scalar",
        validated=True,
    )
    scalar_repetitions = int(prereg["measurement"]["scalar_repetitions"])
    scalar_samples, scalar_output = measure(direct_scalar, scalar_repetitions)
    avx2_subset_samples, avx2_subset_output = measure(
        direct_avx2, scalar_repetitions
    )
    _, aligned_scalar_output = measure(aligned_scalar, 1)
    maximum_absolute_error = float(
        np.max(np.abs(scalar_output - avx2_subset_output), initial=0.0)
    )
    exact_scalar_equivalence = bool(
        np.array_equal(scalar_output, aligned_scalar_output)
    )

    direct_median = statistics.median(direct_samples)
    control_median = statistics.median(control_samples)
    throughput_fraction = control_median / direct_median
    avx2_speedup = statistics.median(scalar_samples) / statistics.median(
        avx2_subset_samples
    )
    gate_spec = prereg["gate"]
    finite = all(
        math.isfinite(value) and value > 0
        for value in direct_samples + control_samples + scalar_samples + avx2_subset_samples
    ) and bool(
        np.all(np.isfinite(direct_output))
        and np.all(np.isfinite(control_output))
        and np.all(np.isfinite(scalar_output))
        and np.all(np.isfinite(avx2_subset_output))
    )
    gate = {
        **gate_spec,
        "observed_direct_avx2_vs_direct_scalar_absolute_error": maximum_absolute_error,
        "observed_direct_avx2_speedup_over_direct_scalar_subset": avx2_speedup,
        "observed_direct_throughput_fraction_of_aligned_two_bit_control": (
            throughput_fraction
        ),
        "observed_exact_scalar_equivalence_to_aligned_two_bit_scalar": (
            exact_scalar_equivalence
        ),
        "observed_all_samples_and_outputs_finite": finite,
    }
    gate["passed"] = (
        maximum_absolute_error
        <= gate_spec["maximum_direct_avx2_vs_direct_scalar_absolute_error"]
        and avx2_speedup
        >= gate_spec["minimum_direct_avx2_speedup_over_direct_scalar_subset"]
        and throughput_fraction
        >= gate_spec["minimum_direct_throughput_fraction_of_aligned_two_bit_control"]
        and exact_scalar_equivalence
        and finite
    )
    receipt = {
        "schema": "qwen38_27b_block203_row_scale4_direct_gemv_receipt_v1",
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
        "payload_bytes": {
            "direct_block203_weights": block203_weights.nbytes,
            "aligned_two_bit_weights": aligned_weights.nbytes,
            "packed_scale_indices": packed_scale_codes.nbytes,
            "fp16_row_anchors": anchors.nbytes,
            "fp16_ratio_codebook": codebook.nbytes,
        },
        "paired_full_shape": {
            "direct_samples_seconds": direct_samples,
            "control_samples_seconds": control_samples,
            "direct_median_seconds": direct_median,
            "control_median_seconds": control_median,
            "direct_rows_per_second": rows / direct_median,
            "control_rows_per_second": rows / control_median,
            "direct_throughput_fraction_of_control": throughput_fraction,
        },
        "correctness_subset": {
            "rows": subset_rows,
            "direct_scalar_samples_seconds": scalar_samples,
            "direct_avx2_samples_seconds": avx2_subset_samples,
            "direct_avx2_speedup_over_scalar": avx2_speedup,
            "maximum_absolute_error": maximum_absolute_error,
            "exact_scalar_equivalence_to_aligned_two_bit": exact_scalar_equivalence,
        },
        "gate": gate,
        "claim_boundary": {
            "qwen38_exact_shape_direct_block203_microkernel_only": True,
            "whole_model_container_integrated": False,
            "model_quality_proven": False,
            "end_to_end_tokens_per_second_proven": False,
            "scientific_breakthrough_proven": False
        }
    }
    atomic_bytes(args.receipt, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
