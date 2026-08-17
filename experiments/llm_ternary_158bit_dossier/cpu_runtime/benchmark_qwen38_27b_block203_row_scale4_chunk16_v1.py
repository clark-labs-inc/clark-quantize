#!/usr/bin/env python3
"""Benchmark hierarchical 16-trit block203 GEMV for Qwen3.8-27B."""
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
            raise RuntimeError("non-finite chunk16 timing")
        samples.append(elapsed)
    assert output is not None
    return samples, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-block203-row-scale4-chunk16-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / (
        "qwen38_27b_block203_row_scale4_chunk16_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if not native_available() or not native_library or not Path(native_library).is_file():
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
    packed_scale_codes = np.frombuffer(pack_nibbles(scale_codes), dtype=np.uint8).copy()
    activation = generator.normal(size=columns).astype(np.float32)
    activation_q8, activation_scales, activation_sums = quantize_q8(activation)

    def direct(implementation: str) -> np.ndarray:
        return gemv_q8_group128_block203_row_scale4(
            block203_weights,
            packed_scale_codes,
            anchors,
            codebook,
            activation_q8,
            activation_scales,
            activation_sums,
            implementation=implementation,
            validated=True,
        )

    functions = (
        ("chunk16", lambda: direct("avx2_chunk16")),
        ("constdiv", lambda: direct("avx2_constdiv")),
        ("generic", lambda: direct("avx2")),
        (
            "aligned",
            lambda: gemv_q8_group128_2bit_row_scale4(
                aligned_weights,
                packed_scale_codes,
                anchors,
                codebook,
                activation_q8,
                activation_scales,
                activation_sums,
                implementation="avx2",
                validated=True,
            ),
        ),
    )
    for _ in range(int(prereg["measurement"]["warmups_each"])):
        for _, function in functions:
            function()

    samples = {name: [] for name, _ in functions}
    outputs: dict[str, np.ndarray] = {}
    repetitions = int(prereg["measurement"]["rotating_four_way_repetitions"])
    for repetition in range(repetitions):
        offset = repetition % len(functions)
        order = functions[offset:] + functions[:offset]
        for name, function in order:
            started = time.perf_counter_ns()
            output = function()
            elapsed = (time.perf_counter_ns() - started) / 1e9
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("non-finite paired chunk16 timing")
            samples[name].append(elapsed)
            outputs[name] = output

    subset_rows = int(prereg["measurement"]["scalar_correctness_rows"])
    subset_groups = subset_rows * groups
    subset_indices = packed_scale_codes[: (subset_groups + 1) // 2].copy()
    subset_block = pack_block203_rows(codes[:subset_rows])

    def subset(implementation: str) -> np.ndarray:
        return gemv_q8_group128_block203_row_scale4(
            subset_block,
            subset_indices,
            anchors[:subset_rows],
            codebook,
            activation_q8,
            activation_scales,
            activation_sums,
            implementation=implementation,
            validated=True,
        )

    scalar_repetitions = int(prereg["measurement"]["scalar_repetitions"])
    chunk16_scalar_samples, chunk16_scalar = measure(
        lambda: subset("scalar_chunk16"), scalar_repetitions
    )
    generic_scalar_samples, generic_scalar = measure(
        lambda: subset("scalar"), scalar_repetitions
    )
    chunk16_avx = subset("avx2_chunk16")
    constdiv_avx = subset("avx2_constdiv")
    maximum_absolute_error = float(
        np.max(np.abs(chunk16_avx - generic_scalar), initial=0.0)
    )
    exact_scalar = bool(np.array_equal(chunk16_scalar, generic_scalar))
    exact_avx = bool(np.array_equal(chunk16_avx, constdiv_avx))

    medians = {name: statistics.median(values) for name, values in samples.items()}
    speedup_constdiv = medians["constdiv"] / medians["chunk16"]
    speedup_generic = medians["generic"] / medians["chunk16"]
    aligned_fraction = medians["aligned"] / medians["chunk16"]
    finite = all(
        math.isfinite(value) and value > 0
        for values in samples.values()
        for value in values
    ) and all(
        math.isfinite(value) and value > 0
        for value in chunk16_scalar_samples + generic_scalar_samples
    ) and all(
        bool(np.all(np.isfinite(output)))
        for output in (
            *outputs.values(),
            chunk16_scalar,
            generic_scalar,
            chunk16_avx,
            constdiv_avx,
        )
    )
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_chunk16_avx2_vs_generic_scalar_absolute_error": maximum_absolute_error,
        "observed_chunk16_speedup_over_constdiv": speedup_constdiv,
        "observed_chunk16_speedup_over_generic_direct": speedup_generic,
        "observed_chunk16_throughput_fraction_of_aligned_two_bit_control": aligned_fraction,
        "observed_exact_chunk16_scalar_equivalence_to_generic_scalar": exact_scalar,
        "observed_exact_chunk16_avx2_equivalence_to_constdiv_avx2": exact_avx,
        "observed_all_samples_and_outputs_finite": finite,
    }
    gate["passed"] = (
        maximum_absolute_error
        <= gate_spec["maximum_chunk16_avx2_vs_generic_scalar_absolute_error"]
        and speedup_constdiv >= gate_spec["minimum_chunk16_speedup_over_constdiv"]
        and speedup_generic >= gate_spec["minimum_chunk16_speedup_over_generic_direct"]
        and aligned_fraction
        >= gate_spec["minimum_chunk16_throughput_fraction_of_aligned_two_bit_control"]
        and exact_scalar
        and exact_avx
        and finite
    )
    receipt = {
        "schema": "qwen38_27b_block203_row_scale4_chunk16_receipt_v1",
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
        "full_shape": {
            "samples_seconds": samples,
            "medians_seconds": medians,
            "chunk16_speedup_over_constdiv": speedup_constdiv,
            "chunk16_speedup_over_generic": speedup_generic,
            "chunk16_throughput_fraction_of_aligned": aligned_fraction,
        },
        "correctness_subset": {
            "rows": subset_rows,
            "chunk16_scalar_samples_seconds": chunk16_scalar_samples,
            "generic_scalar_samples_seconds": generic_scalar_samples,
            "maximum_absolute_error": maximum_absolute_error,
            "exact_scalar_equivalence": exact_scalar,
            "exact_avx2_equivalence": exact_avx,
        },
        "gate": gate,
        "claim_boundary": {
            "qwen38_exact_shape_direct_block203_microkernel_only": True,
            "whole_model_container_integrated": False,
            "model_quality_proven": False,
            "end_to_end_tokens_per_second_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.receipt, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
