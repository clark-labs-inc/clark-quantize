#!/usr/bin/env python3
"""Benchmark the first Clark group-128 ternary CPU GEMV on a real Qwen matrix."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader
from clark_quantize.ternary_cpu import (
    cpu_features,
    gemv_q8_group128_2bit,
    memory_read_sum,
    native_available,
    pack_group128_2bit,
    quantize_q8,
    unpack_group128_2bit,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_preregistration(value: dict, args: argparse.Namespace) -> dict:
    tensor = value.get("tensor", {})
    performance = value.get("performance", {})
    if (
        value.get("schema") != "clark_ternary_cpu_gemv_preregistration_v5"
        or tensor.get("name") != args.tensor
        or tensor.get("rows") != args.rows
        or tensor.get("columns") != args.columns
        or tensor.get("group_size") != 128
        or performance.get("warmups") != args.warmups
        or performance.get("repetitions") != args.repetitions
        or performance.get("threads") != args.threads
        or performance.get("thread_placement") != "persistent_workers_pinned_once_to_distinct_physical_cores"
        or performance.get("cache_eviction_bytes") != args.cache_eviction_bytes
    ):
        raise ValueError("GEMV arguments differ from preregistration")
    return value


def physical_cpu_ids() -> list[int]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    selected: list[int] = []
    physical_cores: set[tuple[str, str]] = set()
    for cpu in affinity:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return affinity
        key = (package, core)
        if key not in physical_cores:
            physical_cores.add(key)
            selected.append(cpu)
    return selected


def pinned_executor(cpu_ids: list[int]) -> ThreadPoolExecutor:
    pending: queue.SimpleQueue[int] = queue.SimpleQueue()
    for cpu_id in cpu_ids:
        pending.put(cpu_id)

    def initialize() -> None:
        os.sched_setaffinity(0, {pending.get()})

    return ThreadPoolExecutor(max_workers=len(cpu_ids), initializer=initialize)


def load_tensor_codes(
    reader: QwenBase3ArtifactReader,
    name: str,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if name not in reader.records_by_tensor:
        raise KeyError(name)
    code_batches: list[np.ndarray] = []
    scale_batches: list[np.ndarray] = []
    record_indices: list[int] = []
    loaded_rows = 0
    for record in reader.records_by_tensor[name]:
        codes, scales = reader.raw_batch(record)
        take = min(rows - loaded_rows, codes.shape[0])
        code_batches.append(codes[:take])
        scale_batches.append(scales[:take])
        record_indices.append(int(record["index"]))
        loaded_rows += take
        if loaded_rows == rows:
            break
    codes = np.ascontiguousarray(np.concatenate(code_batches, axis=0), dtype=np.int8)
    scales = np.ascontiguousarray(np.concatenate(scale_batches, axis=0), dtype=np.float16)
    if codes.shape != (rows, columns) or scales.shape != (rows, math.ceil(columns / 128)):
        raise ValueError("artifact tensor shape differs from preregistration")
    return codes, scales, record_indices


def measure(operation, warmups: int, repetitions: int, evict) -> tuple[list[float], object]:
    result = None
    for _ in range(warmups):
        evict()
        result = operation()
    samples = []
    for _ in range(repetitions):
        evict()
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
    return samples, result


def timing_summary(seconds: list[float], bytes_read: int) -> dict:
    median = statistics.median(seconds)
    return {
        "samples_seconds": seconds,
        "median_seconds": median,
        "minimum_seconds": min(seconds),
        "maximum_seconds": max(seconds),
        "effective_weight_gbytes_per_second": bytes_read / median / 1e9,
    }


def main() -> None:
    default_root = Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q")
    default_prereg = Path(__file__).with_name("gemv_preregistration_v5.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=default_root)
    parser.add_argument("--pack-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-full.json"))
    parser.add_argument("--receipt", type=Path, default=Path("/workspace/quant-state/qwen38-ternary-gemv-avx2-v5.json"))
    parser.add_argument("--preregistration", type=Path, default=default_prereg)
    parser.add_argument("--tensor", default="model.language_model.layers.0.mlp.gate_proj.weight")
    parser.add_argument("--rows", type=int, default=17408)
    parser.add_argument("--columns", type=int, default=5120)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--threads", type=lambda value: [int(item) for item in value.split(",")], default=[1, 2, 4])
    parser.add_argument("--cache-eviction-bytes", type=int, default=268435456)
    args = parser.parse_args()

    preregistration_content = args.preregistration.read_bytes()
    preregistration = validate_preregistration(json.loads(preregistration_content), args)
    features = cpu_features()
    if not native_available() or not features["avx2"] or not features["f16c"] or not features["fma"]:
        raise RuntimeError("the preregistered run requires native AVX2, F16C, and FMA")
    physical_cpus = physical_cpu_ids()
    if len(physical_cpus) < max(args.threads):
        raise RuntimeError("the preregistered run requires four distinct physical CPU cores")
    pack_receipt = json.loads(args.pack_receipt.read_text())
    if (
        pack_receipt.get("promotion", {}).get("passed") is not True
        or pack_receipt.get("identity", {}).get("model_revision") != preregistration["model_revision"]
    ):
        raise ValueError("full-pack receipt is not a promoted preregistered parent")

    reader = QwenBase3ArtifactReader(args.artifact)
    if reader.manifest.get("schema") != preregistration["artifact_schema"]:
        raise ValueError("artifact schema differs from preregistration")
    codes, weight_scales, records = load_tensor_codes(
        reader, args.tensor, args.rows, args.columns,
    )
    raw_code_sha256 = hashlib.sha256(codes.tobytes()).hexdigest()
    packed_flat = pack_group128_2bit(codes.reshape(-1))
    packed = packed_flat.reshape(args.rows, args.columns // 4)
    decoded = unpack_group128_2bit(packed_flat, codes.size).reshape(codes.shape)
    exact_pack = bool(np.array_equal(decoded, codes))
    if not exact_pack:
        raise AssertionError("aligned-2 compute image changed artifact codes")
    del decoded

    generator = np.random.default_rng(preregistration["activation"]["seed"])
    activation = generator.normal(size=args.columns).astype(np.float32)
    activation_q8, activation_scales, activation_sums = quantize_q8(activation)
    scales_finite = bool(
        np.all(np.isfinite(weight_scales))
        and np.all(np.isfinite(activation_scales))
    )
    if not scales_finite:
        raise ValueError("the artifact or activation contains a non-finite scale")
    correctness_rows = int(preregistration["correctness"]["rows"])
    scalar_correctness = gemv_q8_group128_2bit(
        packed[:correctness_rows], weight_scales[:correctness_rows],
        activation_q8, activation_scales, activation_sums,
        implementation="scalar",
        validated=True,
    )
    avx2_correctness = gemv_q8_group128_2bit(
        packed[:correctness_rows], weight_scales[:correctness_rows],
        activation_q8, activation_scales, activation_sums,
        implementation="avx2",
        validated=True,
    )
    exact_scalar_avx2 = bool(np.array_equal(scalar_correctness, avx2_correctness))
    max_abs_error = float(np.max(np.abs(scalar_correctness - avx2_correctness)))
    max_rel_error = float(np.max(
        np.abs(scalar_correctness - avx2_correctness)
        / np.maximum(np.abs(scalar_correctness), 1e-12)
    ))

    eviction = np.arange(args.cache_eviction_bytes, dtype=np.uint8)
    expected_eviction_sum = int(eviction.sum(dtype=np.uint64))
    executors = {
        threads: pinned_executor(physical_cpus[:threads])
        for threads in args.threads
        if threads > 1
    }

    def evict() -> None:
        threads = max(args.threads)
        actual = memory_read_sum(
            eviction,
            implementation="avx2",
            threads=threads,
            executor=executors.get(threads),
            cpu_ids=None,
        )
        if actual != expected_eviction_sum:
            raise AssertionError("cache-eviction bandwidth control checksum changed")

    stream_results = {}
    for threads in args.threads:
        samples, checksum = measure(
            lambda threads=threads: memory_read_sum(
                eviction,
                implementation="avx2",
                threads=threads,
                executor=executors.get(threads),
                cpu_ids=None if threads > 1 else physical_cpus[:1],
            ),
            args.warmups,
            args.repetitions,
            lambda: None,
        )
        if checksum != expected_eviction_sum:
            raise AssertionError("STREAM-like read checksum changed")
        summary = timing_summary(samples, eviction.nbytes)
        summary["gbytes_per_second"] = summary.pop("effective_weight_gbytes_per_second")
        stream_results[str(threads)] = summary

    scalar_rows = int(preregistration["performance"]["scalar_comparison_rows"])
    scalar_samples, scalar_output = measure(
        lambda: gemv_q8_group128_2bit(
            packed[:scalar_rows], weight_scales[:scalar_rows],
            activation_q8, activation_scales, activation_sums,
            implementation="scalar",
            cpu_ids=physical_cpus[:1],
            validated=True,
        ),
        args.warmups,
        args.repetitions,
        evict,
    )
    avx2_subset_samples, avx2_subset_output = measure(
        lambda: gemv_q8_group128_2bit(
            packed[:scalar_rows], weight_scales[:scalar_rows],
            activation_q8, activation_scales, activation_sums,
            implementation="avx2",
            cpu_ids=physical_cpus[:1],
            validated=True,
        ),
        args.warmups,
        args.repetitions,
        evict,
    )
    numerical_limit = float(preregistration["correctness"]["maximum_absolute_error"])
    subset_max_abs_error = float(np.max(np.abs(scalar_output - avx2_subset_output)))
    if subset_max_abs_error > numerical_limit:
        raise AssertionError("scalar and AVX2 subset outputs exceed the preregistered bound")
    scalar_speedup = statistics.median(scalar_samples) / statistics.median(avx2_subset_samples)

    weight_bytes = packed.nbytes + weight_scales.nbytes
    kernel_results = {}
    deterministic = True
    reference_output = None
    for threads in args.threads:
        samples, output = measure(
            lambda threads=threads: gemv_q8_group128_2bit(
                packed, weight_scales, activation_q8, activation_scales, activation_sums,
                implementation="avx2", threads=threads,
                executor=executors.get(threads),
                cpu_ids=None if threads > 1 else physical_cpus[:1],
                validated=True,
            ),
            args.warmups,
            args.repetitions,
            evict,
        )
        if reference_output is None:
            reference_output = output.copy()
        else:
            deterministic = deterministic and bool(np.array_equal(reference_output, output))
        summary = timing_summary(samples, weight_bytes)
        summary["fraction_of_stream_read_bandwidth"] = (
            summary["effective_weight_gbytes_per_second"]
            / stream_results[str(threads)]["gbytes_per_second"]
        )
        summary["output_sha256"] = hashlib.sha256(output.tobytes()).hexdigest()
        kernel_results[str(threads)] = summary
    for executor in executors.values():
        executor.shutdown()

    groups = args.columns // 128
    compute_bits = weight_bytes * 8 / codes.size
    promotion_rules = preregistration["promotion"]
    max_threads = str(max(args.threads))
    promotion_passed = (
        exact_pack
        and scales_finite
        and max_abs_error <= numerical_limit
        and subset_max_abs_error <= numerical_limit
        and deterministic
        and scalar_speedup >= promotion_rules["minimum_avx2_speedup_over_scalar"]
        and kernel_results[max_threads]["fraction_of_stream_read_bandwidth"]
            >= promotion_rules["minimum_fraction_of_stream_read_bandwidth"]
        and compute_bits <= promotion_rules["maximum_compute_image_bits_per_weight_including_fp16_scales"]
    )
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    native_candidates = [
        REPO / "clark_quantize/_ternary_cpu.so",
        REPO / "clark_quantize/_ternary_cpu.dylib",
    ]
    native_library = next(path for path in native_candidates if path.exists())
    result = {
        "schema": "clark_qwen38_ternary_gemv_avx2_v5",
        "hypothesis": (
            "A group-128 aligned-2 AVX2 A8-by-ternary GEMV can preserve exact integer dots, "
            "stay within the preregistered FP32 accumulation bound, and reach at least 70% "
            "of STREAM-like read bandwidth."
        ),
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(preregistration_content).hexdigest(),
            "rules": preregistration,
        },
        "parent": {
            "artifact": str(args.artifact),
            "manifest_sha256": file_sha256(reader.manifest_path),
            "pack_receipt": str(args.pack_receipt),
            "pack_receipt_sha256": file_sha256(args.pack_receipt),
            "model_revision": preregistration["model_revision"],
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "affinity": affinity,
            "logical_cpus": os.cpu_count(),
            "physical_cpu_ids": physical_cpus,
            "cpu_features": features,
        },
        "implementation": {
            "c_source_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.c"),
            "python_wrapper_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.py"),
            "benchmark_sha256": file_sha256(Path(__file__)),
            "build_script_sha256": file_sha256(REPO / "scripts/cpu/build_ternary_cpu.sh"),
            "native_library": str(native_library),
            "native_library_sha256": file_sha256(native_library),
            "compiler": subprocess.check_output([os.environ.get("CC", "cc"), "--version"], text=True).splitlines()[0],
        },
        "tensor": {
            "name": args.tensor,
            "shape": [args.rows, args.columns],
            "group_size": 128,
            "groups_per_row": groups,
            "artifact_records": records,
            "raw_code_sha256": raw_code_sha256,
            "weight_scale_sha256": hashlib.sha256(weight_scales.tobytes()).hexdigest(),
        },
        "compute_image": {
            "layout": "group128-transposed 2-bit: byte j stores codes j,j+32,j+64,j+96 as digit=code+1",
            "packed_code_bytes": packed.nbytes,
            "fp16_scale_bytes": weight_scales.nbytes,
            "total_weight_bytes": weight_bytes,
            "bits_per_weight_including_fp16_scales": compute_bits,
            "exact_pack_roundtrip": exact_pack,
            "packed_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        },
        "activation": {
            "q8_sha256": hashlib.sha256(activation_q8.tobytes()).hexdigest(),
            "scale_sha256": hashlib.sha256(activation_scales.tobytes()).hexdigest(),
            "sum_sha256": hashlib.sha256(activation_sums.tobytes()).hexdigest(),
            "all_scales_finite": scales_finite,
        },
        "correctness": {
            "rows": correctness_rows,
            "exact_scalar_avx2_output": exact_scalar_avx2,
            "maximum_absolute_error": max_abs_error,
            "maximum_relative_error": max_rel_error,
            "subset_maximum_absolute_error": subset_max_abs_error,
            "exact_across_thread_counts": deterministic,
        },
        "controls": {
            "cache_eviction_bytes": eviction.nbytes,
            "cache_eviction_checksum": expected_eviction_sum,
            "stream_read": stream_results,
            "scalar_comparison_rows": scalar_rows,
            "scalar_samples_seconds": scalar_samples,
            "avx2_samples_seconds": avx2_subset_samples,
            "avx2_speedup_over_scalar": scalar_speedup,
        },
        "kernel": kernel_results,
        "promotion": {
            "passed": promotion_passed,
            "scope": (
                "K2 aligned-2 AVX2 microkernel only; no graph-level quality, end-to-end speed, "
                "energy, or scientific-breakthrough claim is authorized"
            ),
        },
        "stop_condition": "One preregistered real Qwen matrix or first provenance, exactness, ISA, or resource failure.",
    }
    atomic_bytes(args.receipt, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not promotion_passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
