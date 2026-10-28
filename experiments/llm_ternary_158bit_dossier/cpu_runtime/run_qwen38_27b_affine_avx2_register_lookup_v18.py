#!/usr/bin/env python3
"""Correctness and exact-shape timing for the frozen AFF3 register lookup v18."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Callable

import numpy as np

import run_qwen38_27b_affine_avx2_x16_fused_v17 as v17
from clark_quantize.qwen38_affine_artifact import (
    load_compressed_affine_tensor,
    verify_descriptor,
)


MODEL = "Qwen/Qwen3.8-27B"
MINIMUM_V17_OVER_V18_SPEEDUP = 1.05


def load_library(path: Path) -> ctypes.CDLL:
    library = v17.load_library(path)
    gemv_arguments = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.cq3_aff3_gemv_x16_register_v18.argtypes = gemv_arguments
    library.cq3_aff3_gemv_x16_register_v18.restype = ctypes.c_int
    library.cq3_aff3_debug_lookup_v18.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.cq3_aff3_debug_lookup_v18.restype = ctypes.c_int
    return library


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def verify_lookup_bits(
    library: ctypes.CDLL,
    codebook: np.ndarray,
    indices: np.ndarray,
) -> None:
    observed = np.empty(16, dtype=np.float32)
    status = library.cq3_aff3_debug_lookup_v18(
        v17.pointer(codebook.view(np.uint16), ctypes.c_uint16),
        v17.pointer(indices, ctypes.c_uint8),
        v17.pointer(observed, ctypes.c_float),
    )
    expected = np.asarray(codebook[indices], dtype=np.float32)
    if status or not np.array_equal(observed.view(np.uint32), expected.view(np.uint32)):
        raise AssertionError("register lookup changed FP16 replay bits")


def compare_candidate(
    library: ctypes.CDLL,
    x16: np.ndarray,
    params: tuple[np.ndarray, ...],
    values: np.ndarray,
    scales: np.ndarray,
    sums: np.ndarray,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.empty(rows, dtype=np.float32)
    candidate = np.empty(rows, dtype=np.float32)
    v17.call_gemv(
        library.cq3_aff3_gemv_x16_v17,
        x16,
        params,
        values,
        scales,
        sums,
        rows,
        columns,
        baseline,
    )
    v17.call_gemv(
        library.cq3_aff3_gemv_x16_register_v18,
        x16,
        params,
        values,
        scales,
        sums,
        rows,
        columns,
        candidate,
    )
    if not np.array_equal(baseline.view(np.uint32), candidate.view(np.uint32)):
        raise AssertionError("v17/v18 output bits changed")
    if not np.all(np.isfinite(candidate)):
        raise AssertionError("v18 produced non-finite output")
    return baseline, candidate


def self_test(library: ctypes.CDLL) -> dict[str, object]:
    indices = np.arange(16, dtype=np.uint8)
    params16 = v17.parameters(16)
    verify_lookup_bits(library, params16[2], indices)
    verify_lookup_bits(library, params16[3], indices[::-1].copy())
    if set(int(value) for value in indices) != set(range(16)):
        raise AssertionError("all sixteen codebook indices were not exercised")

    shapes = []
    for shape_index, (rows, columns) in enumerate(((16, 128), (32, 512), (48, 5120))):
        groups = columns // v17.GROUP_SIZE
        codes = v17.deterministic_codes(rows, columns, 701 + shape_index)
        scale_indices = v17.combined_indices(rows, groups, 811 + shape_index)
        source = v17.row_major_image(codes, scale_indices)
        _, x16 = v17.repack(library, source, rows, groups)
        params = v17.parameters(rows)
        values, scales, sums = v17.activation(columns)
        baseline, candidate = compare_candidate(
            library, x16, params, values, scales, sums, rows, columns
        )
        reference = v17.scalar_reference(
            codes, scale_indices, params, values, scales, sums
        )
        maximum_delta = v17.scaled_delta(reference, candidate)
        if maximum_delta > 0.0001:
            raise AssertionError("v18 changed the exact affine algebra")
        debug_dots = np.empty(16, dtype=np.int32)
        status = library.cq3_aff3_debug_dot_x16_v17(
            x16[:528].ctypes.data,
            v17.pointer(values[: v17.GROUP_SIZE], ctypes.c_int8),
            int(sums[0]),
            v17.pointer(debug_dots, ctypes.c_int32),
        )
        expected_dots = (
            codes[:16, : v17.GROUP_SIZE].astype(np.int32)
            @ values[: v17.GROUP_SIZE].astype(np.int32)
        )
        if status or not np.array_equal(debug_dots, expected_dots):
            raise AssertionError("v18 integer dot changed")
        broken = source.copy()
        broken[0, 0, 0] = 0xFF
        broken_output = np.empty_like(x16)
        if library.cq3_aff3_repack_x16_v17(
            broken.ctypes.data, broken_output.ctypes.data, rows, groups
        ) != 3:
            raise AssertionError("v18 accepted reserved two-bit codes")
        shapes.append(
            {
                "rows": rows,
                "columns": columns,
                "integer_dot_exact": True,
                "v17_v18_output_bit_exact": True,
                "all_outputs_finite": True,
                "maximum_scaled_float_error": maximum_delta,
                "output_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
                "baseline_output_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
            }
        )
    sizes = {
        "row_group": library.cq3_aff3_row_group_size_v17(),
        "x8_group": library.cq3_aff3_x8_group_size_v17(),
        "x16_group": library.cq3_aff3_x16_group_size_v17(),
    }
    if sizes != {"row_group": 33, "x8_group": 264, "x16_group": 528}:
        raise AssertionError("AFF3_X16 runtime layout changed")
    return {
        "schema": "qwen38_27b_affine_avx2_register_lookup_native_correctness_v18",
        "model": MODEL,
        "repository_commit": git_commit(),
        "v17_native_contract_replayed": True,
        "runtime_sizes": sizes,
        "reserved_code_rejected": True,
        "all_sixteen_codebook_indices_exercised": True,
        "lookup_fp32_bits_match_scalar_fp16_replay": True,
        "additional_persistent_bytes": 0,
        "additional_runtime_compute_image_bytes": 0,
        "runtime_compute_image_bpw_at_5120_columns": 2.06875,
        "shapes": shapes,
        "passed": True,
        "complete": True,
        "terminal": True,
    }


def artifact_self_test(
    library: ctypes.CDLL,
    artifact_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "qwen38_affine_butterfly_site_v1"
        or manifest.get("complete") is not True
        or manifest.get("terminal") is not True
    ):
        raise ValueError("actual affine site manifest is not terminal")
    angles_path = artifact_root / manifest["angles"]["path"]
    angles = angles_path.read_bytes()
    if (
        len(angles) != manifest["angles"]["bytes"]
        or hashlib.sha256(angles).hexdigest() != manifest["angles"]["sha256"]
    ):
        raise ValueError("actual affine rotation failed authentication")
    eligible = [
        descriptor
        for descriptor in manifest["tensors"]
        if len(descriptor["shape"]) == 2
        and descriptor["shape"][0] % 16 == 0
        and descriptor["shape"][1] % v17.GROUP_SIZE == 0
    ]
    if not eligible:
        raise ValueError("actual affine site has no AFF3_X16-compatible tensor")
    descriptor = min(eligible, key=lambda item: math.prod(item["shape"]))
    verify_descriptor(artifact_root, descriptor)
    compressed = load_compressed_affine_tensor(artifact_root, descriptor)
    codes = np.ascontiguousarray(compressed.payload.codes.numpy(), dtype=np.int8)
    rows, columns = codes.shape
    groups = columns // v17.GROUP_SIZE
    alpha_indices = np.ascontiguousarray(compressed.alpha.indices, dtype=np.uint8)
    mu_indices = np.ascontiguousarray(compressed.mu.indices, dtype=np.uint8)
    scale_indices = np.asarray(alpha_indices | (mu_indices << 4), dtype=np.uint8)
    source = v17.row_major_image(codes, scale_indices)
    _, x16 = v17.repack(library, source, rows, groups)
    params = (
        np.ascontiguousarray(compressed.alpha.anchors, dtype=np.float16),
        np.ascontiguousarray(compressed.mu.anchors, dtype=np.float16),
        np.ascontiguousarray(compressed.alpha.values, dtype=np.float16),
        np.ascontiguousarray(compressed.mu.values, dtype=np.float16),
    )
    values, scales, sums = v17.activation(columns)
    baseline, candidate = compare_candidate(
        library, x16, params, values, scales, sums, rows, columns
    )
    debug_dots = np.empty(16, dtype=np.int32)
    status = library.cq3_aff3_debug_dot_x16_v17(
        x16[:528].ctypes.data,
        v17.pointer(values[: v17.GROUP_SIZE], ctypes.c_int8),
        int(sums[0]),
        v17.pointer(debug_dots, ctypes.c_int32),
    )
    expected_dots = (
        codes[:16, : v17.GROUP_SIZE].astype(np.int32)
        @ values[: v17.GROUP_SIZE].astype(np.int32)
    )
    if status or not np.array_equal(debug_dots, expected_dots):
        raise AssertionError("actual v18 integer dot changed")
    return {
        "schema": "qwen38_27b_affine_avx2_register_lookup_actual_artifact_v18",
        "model": MODEL,
        "repository_commit": git_commit(),
        "artifact_root": str(artifact_root),
        "manifest": str(manifest_path),
        "tensor": descriptor["tensor"],
        "shape": [rows, columns],
        "canonical_codes_sha256": descriptor["codes"]["sha256"],
        "all_source_blobs_authenticated": True,
        "rotation_authenticated": True,
        "integer_dot_exact": True,
        "v17_v18_output_bit_exact": True,
        "baseline_output_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
        "candidate_output_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
        "all_outputs_finite": True,
        "passed": True,
        "complete": True,
        "terminal": True,
    }


def ensure_idle_host() -> None:
    if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) != 1:
        raise SystemExit("infrastructure-invalid: benchmark must be pinned to one physical core")
    units = (
        ("quant-qwen38-27b-affine-butterfly-block-gate-v1.service", "PTQ"),
        ("quant-qwen38-27b-affine-butterfly-block-qat-v1.service", "QAT"),
    )
    for unit, label in units:
        active = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit], check=False
        ).returncode == 0
        if active:
            message = (
                "infrastructure-invalid: QAT service is active"
                if label == "QAT"
                else "infrastructure-invalid: PTQ service is active"
            )
            raise SystemExit(message)


def timed_batch(run: Callable[[], None]) -> float:
    started = time.perf_counter_ns()
    for _ in range(v17.INNER_ITERATIONS):
        run()
    return (time.perf_counter_ns() - started) / 1000.0 / v17.INNER_ITERATIONS


def benchmark(library: ctypes.CDLL) -> dict[str, object]:
    ensure_idle_host()
    sources = [v17.synthetic_row_major(v17.ROWS, v17.GROUPS, seed) for seed in (101, 211)]
    images = [v17.repack(library, source, v17.ROWS, v17.GROUPS)[1] for source in sources]
    params = v17.parameters(v17.ROWS)
    values, scales, sums = v17.activation(v17.COLUMNS)
    outputs = {
        name: [np.empty(v17.ROWS, dtype=np.float32), np.empty(v17.ROWS, dtype=np.float32)]
        for name in ("v17", "v18")
    }

    def make_run(name: str, function: object) -> Callable[[], None]:
        def run() -> None:
            for matrix in range(2):
                v17.call_gemv(
                    function,
                    images[matrix],
                    params,
                    values,
                    scales,
                    sums,
                    v17.ROWS,
                    v17.COLUMNS,
                    outputs[name][matrix],
                )
        return run

    runs = {
        "v17": make_run("v17", library.cq3_aff3_gemv_x16_v17),
        "v18": make_run("v18", library.cq3_aff3_gemv_x16_register_v18),
    }
    for run in runs.values():
        run()
    for matrix in range(2):
        if not np.array_equal(
            outputs["v17"][matrix].view(np.uint32),
            outputs["v18"][matrix].view(np.uint32),
        ):
            raise AssertionError("v17/v18 output bits changed")
    for _ in range(v17.WARMUPS):
        for run in runs.values():
            run()

    eviction = np.zeros(v17.CACHE_EVICTION_BYTES, dtype=np.uint8)
    samples: dict[str, list[float]] = {name: [] for name in runs}
    orders = (("v17", "v18"), ("v18", "v17"))
    for batch in range(v17.TIMED_BATCHES):
        for name in orders[batch % 2]:
            eviction[::64] += np.uint8(batch + len(samples[name]) + 1)
            samples[name].append(timed_batch(runs[name]))
    summary = {name: v17.summarize(values_) for name, values_ in samples.items()}
    speedup = summary["v17"]["median_us"] / summary["v18"]["median_us"]
    checks = {
        "minimum_v17_over_v18_speedup": speedup >= MINIMUM_V17_OVER_V18_SPEEDUP,
        "v18_median_absolute_deviation_fraction":
            summary["v18"]["median_absolute_deviation_fraction"] <= 0.03,
        "v18_p90_over_p10_ratio": summary["v18"]["p90_over_p10_ratio"] <= 1.08,
        "v17_v18_output_bit_exact": True,
        "additional_runtime_compute_image_bytes_zero": True,
    }
    return {
        "schema": "qwen38_27b_affine_avx2_register_lookup_exact_shape_benchmark_v18",
        "model": MODEL,
        "repository_commit": git_commit(),
        "rows_per_matrix": v17.ROWS,
        "matrices": 2,
        "columns": v17.COLUMNS,
        "warmups": v17.WARMUPS,
        "timed_batches": v17.TIMED_BATCHES,
        "inner_iterations_per_timed_batch": v17.INNER_ITERATIONS,
        "cache_eviction_bytes": v17.CACHE_EVICTION_BYTES,
        "minimum_v17_over_v18_speedup": MINIMUM_V17_OVER_V18_SPEEDUP,
        "v17_over_v18_speedup": speedup,
        "samples": summary,
        "checks": checks,
        "passed": all(checks.values()),
        "complete": True,
        "terminal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("self-test", "artifact-self-test", "benchmark"), required=True
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    library = load_library(args.library)
    if args.mode == "self-test":
        result = self_test(library)
    elif args.mode == "artifact-self-test":
        if args.artifact_root is None or args.manifest is None:
            parser.error("artifact-self-test requires --artifact-root and --manifest")
        result = artifact_self_test(library, args.artifact_root, args.manifest)
    else:
        result = benchmark(library)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(encoded)
        os.replace(temporary, args.output)
    print(encoded, end="", flush=True)
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
