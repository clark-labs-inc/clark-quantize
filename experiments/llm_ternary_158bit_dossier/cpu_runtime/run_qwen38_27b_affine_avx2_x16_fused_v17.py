#!/usr/bin/env python3
"""Correctness and exact-shape timing harness for the frozen AFF3_X16 kernel."""
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

from clark_quantize.ternary_cpu import (
    pack_block203_rows,
    transcode_block203_rows_to_group128_2bit,
    unpack_block203_rows,
)
from clark_quantize.qwen38_affine_artifact import (
    load_compressed_affine_tensor,
    verify_descriptor,
)


MODEL = "Qwen/Qwen3.8-27B"
ROWS = 17_408
COLUMNS = 5_120
GROUP_SIZE = 128
GROUPS = COLUMNS // GROUP_SIZE
WARMUPS = 3
TIMED_BATCHES = 11
INNER_ITERATIONS = 8
CACHE_EVICTION_BYTES = 256 * 1024 * 1024


def pointer(array: np.ndarray, ctype: type[ctypes._SimpleCData]) -> object:
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.cq3_aff3_row_group_size_v17.restype = ctypes.c_size_t
    library.cq3_aff3_x8_group_size_v17.restype = ctypes.c_size_t
    library.cq3_aff3_x16_group_size_v17.restype = ctypes.c_size_t
    repack_arguments = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.cq3_aff3_repack_x8_v17.argtypes = repack_arguments
    library.cq3_aff3_repack_x8_v17.restype = ctypes.c_int
    library.cq3_aff3_repack_x16_v17.argtypes = repack_arguments
    library.cq3_aff3_repack_x16_v17.restype = ctypes.c_int
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
    for name in (
        "cq3_aff3_gemv_ordinary_v17",
        "cq3_aff3_gemv_x8_v17",
        "cq3_aff3_gemv_x16_v17",
    ):
        function = getattr(library, name)
        function.argtypes = gemv_arguments
        function.restype = ctypes.c_int
    library.cq3_aff3_debug_dot_x16_v17.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.cq3_aff3_debug_dot_x16_v17.restype = ctypes.c_int
    library.cq3_aff3_debug_decode_scales_x16_v17.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.cq3_aff3_debug_decode_scales_x16_v17.restype = ctypes.c_int
    return library


def deterministic_codes(rows: int, columns: int, seed: int) -> np.ndarray:
    row = np.arange(rows, dtype=np.int64)[:, None]
    column = np.arange(columns, dtype=np.int64)[None, :]
    return np.asarray((row * 17 + column * 29 + seed) % 3 - 1, dtype=np.int8)


def combined_indices(rows: int, groups: int, seed: int) -> np.ndarray:
    row = np.arange(rows, dtype=np.uint32)[:, None]
    group = np.arange(groups, dtype=np.uint32)[None, :]
    alpha = (row * 5 + group * 3 + seed) & 15
    mu = (row * 7 + group * 11 + seed * 3) & 15
    return np.asarray(alpha | (mu << 4), dtype=np.uint8)


def row_major_image(codes: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows, columns = codes.shape
    groups = columns // GROUP_SIZE
    canonical = pack_block203_rows(codes)
    rebuilt = unpack_block203_rows(canonical, rows, columns)
    if not np.array_equal(rebuilt, codes):
        raise AssertionError("canonical base-3 mapping changed")
    compute = transcode_block203_rows_to_group128_2bit(canonical, rows, columns)
    result = np.empty((rows, groups, 33), dtype=np.uint8)
    result[:, :, :32] = compute.reshape(rows, groups, 32)
    result[:, :, 32] = indices
    return result


def synthetic_row_major(rows: int, groups: int, seed: int) -> np.ndarray:
    row = np.arange(rows, dtype=np.uint32)[:, None, None]
    group = np.arange(groups, dtype=np.uint32)[None, :, None]
    byte = np.arange(32, dtype=np.uint32)[None, None, :]
    packed = np.zeros((rows, groups, 32), dtype=np.uint8)
    for quarter in range(4):
        digits = (row * 17 + group * 13 + byte * 5 + seed + quarter * 11) % 3
        packed |= np.asarray(digits << (quarter * 2), dtype=np.uint8)
    result = np.empty((rows, groups, 33), dtype=np.uint8)
    result[:, :, :32] = packed
    result[:, :, 32] = combined_indices(rows, groups, seed)
    return result


def parameters(rows: int) -> tuple[np.ndarray, ...]:
    alpha_anchor = np.asarray(
        0.006 + (np.arange(rows, dtype=np.float32) % 31) * 0.00017,
        dtype=np.float16,
    )
    mu_anchor = np.asarray(
        ((np.arange(rows, dtype=np.float32) % 19) - 9) * 0.00013,
        dtype=np.float16,
    )
    alpha_codebook = np.asarray(
        np.geomspace(0.55, 1.65, 16, dtype=np.float32), dtype=np.float16
    )
    mu_codebook = np.asarray(np.linspace(-1.2, 1.2, 16), dtype=np.float16)
    return alpha_anchor, mu_anchor, alpha_codebook, mu_codebook


def activation(columns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(
        ((np.arange(columns, dtype=np.int32) * 37 + 17) % 255) - 127,
        dtype=np.int8,
    )
    groups = columns // GROUP_SIZE
    scales = np.asarray(
        0.002 + (np.arange(groups, dtype=np.float32) % 17) * 0.00007,
        dtype=np.float32,
    )
    sums = values.reshape(groups, GROUP_SIZE).astype(np.int32).sum(1, dtype=np.int32)
    return values, scales, sums


def repack(library: ctypes.CDLL, source: np.ndarray, rows: int, groups: int) -> tuple[np.ndarray, np.ndarray]:
    x8 = np.empty(rows // 8 * groups * 264, dtype=np.uint8)
    x16 = np.empty(rows // 16 * groups * 528, dtype=np.uint8)
    if library.cq3_aff3_repack_x8_v17(source.ctypes.data, x8.ctypes.data, rows, groups):
        raise AssertionError("AFF3_X8 repack failed")
    if library.cq3_aff3_repack_x16_v17(source.ctypes.data, x16.ctypes.data, rows, groups):
        raise AssertionError("AFF3_X16 repack failed")
    return x8, x16


def call_gemv(
    function: object,
    weights: np.ndarray,
    params: tuple[np.ndarray, ...],
    values: np.ndarray,
    scales: np.ndarray,
    sums: np.ndarray,
    rows: int,
    columns: int,
    output: np.ndarray,
) -> None:
    alpha_anchor, mu_anchor, alpha_codebook, mu_codebook = params
    status = function(
        weights.ctypes.data,
        pointer(alpha_anchor.view(np.uint16), ctypes.c_uint16),
        pointer(mu_anchor.view(np.uint16), ctypes.c_uint16),
        pointer(alpha_codebook.view(np.uint16), ctypes.c_uint16),
        pointer(mu_codebook.view(np.uint16), ctypes.c_uint16),
        pointer(values, ctypes.c_int8),
        pointer(scales, ctypes.c_float),
        pointer(sums, ctypes.c_int32),
        rows,
        columns,
        pointer(output, ctypes.c_float),
    )
    if status:
        raise RuntimeError(f"native AFF3 GEMV failed with status {status}")


def scalar_reference(
    codes: np.ndarray,
    indices: np.ndarray,
    params: tuple[np.ndarray, ...],
    values: np.ndarray,
    scales: np.ndarray,
    sums: np.ndarray,
) -> np.ndarray:
    rows, columns = codes.shape
    groups = columns // GROUP_SIZE
    alpha_anchor, mu_anchor, alpha_codebook, mu_codebook = params
    result = np.zeros(rows, dtype=np.float32)
    shaped_codes = codes.reshape(rows, groups, GROUP_SIZE).astype(np.int32)
    shaped_values = values.reshape(groups, GROUP_SIZE).astype(np.int32)
    for row in range(rows):
        total = np.float32(0)
        for group in range(groups):
            dot = int(np.dot(shaped_codes[row, group], shaped_values[group]))
            index = int(indices[row, group])
            alpha = np.float32(alpha_anchor[row]) * np.float32(alpha_codebook[index & 15])
            mu = np.float32(mu_anchor[row]) * np.float32(mu_codebook[index >> 4])
            affine = np.float32(np.float32(dot) * alpha + np.float32(sums[group]) * mu)
            total = np.float32(total + affine * scales[group])
        result[row] = total
    return result


def scaled_delta(reference: np.ndarray, observed: np.ndarray) -> float:
    denominator = np.maximum(1.0, np.maximum(np.abs(reference), np.abs(observed)))
    return float(np.max(np.abs(reference - observed) / denominator))


def self_test(library: ctypes.CDLL) -> dict[str, object]:
    shape_results = []
    for shape_index, (rows, columns) in enumerate(((16, 128), (32, 512), (48, 5120))):
        groups = columns // GROUP_SIZE
        codes = deterministic_codes(rows, columns, 101 + shape_index)
        indices = combined_indices(rows, groups, 23 + shape_index)
        source = row_major_image(codes, indices)
        x8, x16 = repack(library, source, rows, groups)
        params = parameters(rows)
        values, scales, sums = activation(columns)
        ordinary = np.empty(rows, dtype=np.float32)
        output_x8 = np.empty(rows, dtype=np.float32)
        output_x16 = np.empty(rows, dtype=np.float32)
        call_gemv(library.cq3_aff3_gemv_ordinary_v17, source, params,
            values, scales, sums, rows, columns, ordinary)
        call_gemv(library.cq3_aff3_gemv_x8_v17, x8, params,
            values, scales, sums, rows, columns, output_x8)
        call_gemv(library.cq3_aff3_gemv_x16_v17, x16, params,
            values, scales, sums, rows, columns, output_x16)
        reference = scalar_reference(codes, indices, params, values, scales, sums)
        if not np.array_equal(output_x8.view(np.uint32), output_x16.view(np.uint32)):
            raise AssertionError("AFF3_X8 and AFF3_X16 changed output bits")
        maximum_delta = max(scaled_delta(reference, output_x16), scaled_delta(ordinary, output_x16))
        if maximum_delta > 0.0001 or not np.all(np.isfinite(output_x16)):
            raise AssertionError("AFF3_X16 numerical gate failed")

        debug_dots = np.empty(16, dtype=np.int32)
        first_group = x16[:528]
        status = library.cq3_aff3_debug_dot_x16_v17(
            first_group.ctypes.data,
            pointer(values[:GROUP_SIZE], ctypes.c_int8),
            int(sums[0]),
            pointer(debug_dots, ctypes.c_int32),
        )
        expected_dots = codes[:16, :GROUP_SIZE].astype(np.int32) @ values[:GROUP_SIZE].astype(np.int32)
        if status or not np.array_equal(debug_dots, expected_dots):
            raise AssertionError("AFF3_X16 integer dot changed")

        alpha_decoded = np.empty(16, dtype=np.float32)
        mu_decoded = np.empty(16, dtype=np.float32)
        first_indices = np.ascontiguousarray(indices[:16, 0])
        status = library.cq3_aff3_debug_decode_scales_x16_v17(
            pointer(first_indices, ctypes.c_uint8),
            pointer(params[0][:16].view(np.uint16), ctypes.c_uint16),
            pointer(params[1][:16].view(np.uint16), ctypes.c_uint16),
            pointer(params[2].view(np.uint16), ctypes.c_uint16),
            pointer(params[3].view(np.uint16), ctypes.c_uint16),
            pointer(alpha_decoded, ctypes.c_float),
            pointer(mu_decoded, ctypes.c_float),
        )
        expected_alpha = np.asarray(params[0][:16], dtype=np.float32) * np.asarray(
            params[2][indices[:16, 0] & 15], dtype=np.float32
        )
        expected_mu = np.asarray(params[1][:16], dtype=np.float32) * np.asarray(
            params[3][indices[:16, 0] >> 4], dtype=np.float32
        )
        if status or not np.array_equal(alpha_decoded.view(np.uint32), expected_alpha.view(np.uint32)):
            raise AssertionError("AFF3 alpha decode changed bits")
        if not np.array_equal(mu_decoded.view(np.uint32), expected_mu.view(np.uint32)):
            raise AssertionError("AFF3 mu decode changed bits")

        broken = source.copy()
        broken[0, 0, 0] = 0xff
        broken_output = np.empty_like(x16)
        if library.cq3_aff3_repack_x16_v17(
            broken.ctypes.data, broken_output.ctypes.data, rows, groups
        ) != 3:
            raise AssertionError("AFF3_X16 accepted reserved two-bit codes")
        shape_results.append({
            "rows": rows,
            "columns": columns,
            "integer_dot_exact": True,
            "x8_x16_bit_exact": True,
            "scale_decode_bit_exact": True,
            "maximum_scaled_float_error": maximum_delta,
        })
    sizes = {
        "row_group": library.cq3_aff3_row_group_size_v17(),
        "x8_group": library.cq3_aff3_x8_group_size_v17(),
        "x16_group": library.cq3_aff3_x16_group_size_v17(),
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
        and descriptor["shape"][1] % GROUP_SIZE == 0
    ]
    if not eligible:
        raise ValueError("actual affine site has no AFF3_X16-compatible tensor")
    descriptor = min(eligible, key=lambda item: math.prod(item["shape"]))
    verify_descriptor(artifact_root, descriptor)
    compressed = load_compressed_affine_tensor(artifact_root, descriptor)
    codes = np.ascontiguousarray(compressed.payload.codes.numpy(), dtype=np.int8)
    rows, columns = codes.shape
    groups = columns // GROUP_SIZE
    alpha_indices = np.ascontiguousarray(compressed.alpha.indices, dtype=np.uint8)
    mu_indices = np.ascontiguousarray(compressed.mu.indices, dtype=np.uint8)
    indices = np.asarray(alpha_indices | (mu_indices << 4), dtype=np.uint8)
    canonical = pack_block203_rows(codes)
    if hashlib.sha256(canonical.tobytes()).hexdigest() != descriptor["codes"]["sha256"]:
        raise AssertionError("actual affine canonical base-3 bytes changed")
    source = row_major_image(codes, indices)
    x8, x16 = repack(library, source, rows, groups)
    params = (
        np.ascontiguousarray(compressed.alpha.anchors, dtype=np.float16),
        np.ascontiguousarray(compressed.mu.anchors, dtype=np.float16),
        np.ascontiguousarray(compressed.alpha.values, dtype=np.float16),
        np.ascontiguousarray(compressed.mu.values, dtype=np.float16),
    )
    values, scales, sums = activation(columns)
    ordinary = np.empty(rows, dtype=np.float32)
    output_x8 = np.empty(rows, dtype=np.float32)
    output_x16 = np.empty(rows, dtype=np.float32)
    call_gemv(library.cq3_aff3_gemv_ordinary_v17, source, params,
        values, scales, sums, rows, columns, ordinary)
    call_gemv(library.cq3_aff3_gemv_x8_v17, x8, params,
        values, scales, sums, rows, columns, output_x8)
    call_gemv(library.cq3_aff3_gemv_x16_v17, x16, params,
        values, scales, sums, rows, columns, output_x16)
    reference = scalar_reference(codes, indices, params, values, scales, sums)
    if not np.array_equal(output_x8.view(np.uint32), output_x16.view(np.uint32)):
        raise AssertionError("actual AFF3 x8/x16 outputs changed bits")
    maximum_delta = max(scaled_delta(reference, output_x16), scaled_delta(ordinary, output_x16))
    if maximum_delta > 0.0001 or not np.all(np.isfinite(output_x16)):
        raise AssertionError("actual AFF3_X16 numerical gate failed")
    debug_dots = np.empty(16, dtype=np.int32)
    status = library.cq3_aff3_debug_dot_x16_v17(
        x16[:528].ctypes.data,
        pointer(values[:GROUP_SIZE], ctypes.c_int8),
        int(sums[0]),
        pointer(debug_dots, ctypes.c_int32),
    )
    expected_dots = codes[:16, :GROUP_SIZE].astype(np.int32) @ values[:GROUP_SIZE].astype(np.int32)
    if status or not np.array_equal(debug_dots, expected_dots):
        raise AssertionError("actual AFF3_X16 integer dot changed")
    return {
        "schema": "qwen38_27b_affine_avx2_x16_actual_artifact_correctness_v17",
        "model": MODEL,
        "artifact_root": str(artifact_root),
        "manifest": str(manifest_path),
        "manifest_repository_commit": manifest["repository_commit"],
        "manifest_preregistration_sha256": manifest["preregistration_sha256"],
        "tensor": descriptor["tensor"],
        "shape": [rows, columns],
        "canonical_codes_sha256": descriptor["codes"]["sha256"],
        "all_source_blobs_authenticated": True,
        "rotation_authenticated": True,
        "canonical_base3_reencoded_identically": True,
        "integer_dot_exact": True,
        "x8_x16_bit_exact": True,
        "maximum_scaled_float_error": maximum_delta,
        "all_outputs_finite": True,
        "passed": True,
        "complete": True,
        "terminal": True,
    }
    if sizes != {"row_group": 33, "x8_group": 264, "x16_group": 528}:
        raise AssertionError("AFF3 runtime size contract changed")
    return {
        "schema": "qwen38_27b_affine_avx2_x16_native_correctness_v17",
        "model": MODEL,
        "runtime_sizes": sizes,
        "runtime_compute_image_bpw_at_5120_columns": 2.06875,
        "reserved_code_rejected": True,
        "shapes": shape_results,
        "passed": True,
        "complete": True,
        "terminal": True,
    }


def timed_batch(run: Callable[[], None], inner_iterations: int) -> float:
    started = time.perf_counter_ns()
    for _ in range(inner_iterations):
        run()
    return (time.perf_counter_ns() - started) / 1000.0 / inner_iterations


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(samples, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return {
        "samples_us": samples,
        "median_us": median,
        "median_absolute_deviation_fraction": mad / median,
        "p90_over_p10_ratio": float(np.percentile(array, 90) / np.percentile(array, 10)),
    }


def benchmark(library: ctypes.CDLL) -> dict[str, object]:
    if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) != 1:
        raise SystemExit("infrastructure-invalid: benchmark must be pinned to one physical core")
    unit = "quant-qwen38-27b-affine-butterfly-block-gate-v1.service"
    if subprocess.run(["systemctl", "is-active", "--quiet", unit], check=False).returncode == 0:
        raise SystemExit("infrastructure-invalid: affine PTQ service is active")
    sources = [synthetic_row_major(ROWS, GROUPS, seed) for seed in (101, 211)]
    x8_images = []
    x16_images = []
    repack_started = time.perf_counter_ns()
    for source in sources:
        x8, x16 = repack(library, source, ROWS, GROUPS)
        x8_images.append(x8)
        x16_images.append(x16)
    repack_us = (time.perf_counter_ns() - repack_started) / 1000.0
    params = parameters(ROWS)
    values, scales, sums = activation(COLUMNS)
    outputs = {
        name: [np.empty(ROWS, dtype=np.float32), np.empty(ROWS, dtype=np.float32)]
        for name in ("ordinary", "x8", "x16")
    }

    def make_run(name: str, function: object, images: list[np.ndarray]) -> Callable[[], None]:
        def run() -> None:
            for matrix in range(2):
                call_gemv(function, images[matrix], params, values, scales, sums,
                    ROWS, COLUMNS, outputs[name][matrix])
        return run

    runs = {
        "ordinary": make_run("ordinary", library.cq3_aff3_gemv_ordinary_v17, sources),
        "x8": make_run("x8", library.cq3_aff3_gemv_x8_v17, x8_images),
        "x16": make_run("x16", library.cq3_aff3_gemv_x16_v17, x16_images),
    }
    for run in runs.values():
        run()
    for matrix in range(2):
        if not np.array_equal(
            outputs["x8"][matrix].view(np.uint32), outputs["x16"][matrix].view(np.uint32)
        ):
            raise AssertionError("exact-shape x8/x16 outputs changed bits")
    for _ in range(WARMUPS):
        for run in runs.values():
            run()

    eviction = np.zeros(CACHE_EVICTION_BYTES, dtype=np.uint8)
    samples: dict[str, list[float]] = {name: [] for name in runs}
    orders = (
        ("ordinary", "x8", "x16"),
        ("x16", "ordinary", "x8"),
        ("x8", "x16", "ordinary"),
    )
    for batch in range(TIMED_BATCHES):
        for name in orders[batch % len(orders)]:
            eviction[::64] += np.uint8(batch + len(samples[name]) + 1)
            samples[name].append(timed_batch(runs[name], INNER_ITERATIONS))
    summary = {name: summarize(values_) for name, values_ in samples.items()}
    ordinary_over_x16 = summary["ordinary"]["median_us"] / summary["x16"]["median_us"]
    x8_over_x16 = summary["x8"]["median_us"] / summary["x16"]["median_us"]
    checks = {
        "ordinary_over_x16_at_least_1_35": ordinary_over_x16 >= 1.35,
        "x8_over_x16_at_least_1_06": x8_over_x16 >= 1.06,
        "x16_mad_fraction_at_most_0_03":
            summary["x16"]["median_absolute_deviation_fraction"] <= 0.03,
        "x16_p90_over_p10_at_most_1_08": summary["x16"]["p90_over_p10_ratio"] <= 1.08,
    }
    return {
        "schema": "qwen38_27b_affine_avx2_x16_exact_shape_benchmark_v17",
        "model": MODEL,
        "rows_per_matrix": ROWS,
        "matrices": 2,
        "columns": COLUMNS,
        "warmups": WARMUPS,
        "timed_batches": TIMED_BATCHES,
        "inner_iterations_per_timed_batch": INNER_ITERATIONS,
        "cache_eviction_bytes": CACHE_EVICTION_BYTES,
        "repack_us_excluded": repack_us,
        "x8_x16_bit_exact": True,
        "samples": summary,
        "ordinary_over_x16_speedup": ordinary_over_x16,
        "x8_over_x16_speedup": x8_over_x16,
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
    if not all(math.isfinite(value) for shape in result.get("shapes", [])
               for key, value in shape.items() if key == "maximum_scaled_float_error"):
        raise RuntimeError("non-finite AFF3 result")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
