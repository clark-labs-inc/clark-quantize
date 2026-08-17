#!/usr/bin/env python3
"""Benchmark the preregistered ARM DOTPROD GEMV on a real Qwen matrix."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_ternary_gemv import file_sha256, measure, timing_summary
from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.ternary_cpu import (
    cpu_features,
    gemv_q8_group128_2bit,
    memory_read_sum,
    native_available,
    pack_group128_2bit,
    quantize_q8,
    unpack_group128_2bit,
)


def validate_preregistration(value: dict, args: argparse.Namespace) -> dict:
    tensor = value.get("tensor", {})
    performance = value.get("performance", {})
    if (
        value.get("schema") != "clark_ternary_cpu_gemv_arm_preregistration_v1"
        or value.get("status") != "frozen_before_authoritative_real-matrix_run"
        or tensor.get("rows") != 17408
        or tensor.get("columns") != 5120
        or tensor.get("group_size") != 128
        or performance.get("warmups") != args.warmups
        or performance.get("repetitions") != args.repetitions
        or performance.get("threads") != args.threads
        or performance.get("cache_eviction_bytes") != args.cache_eviction_bytes
    ):
        raise ValueError("ARM GEMV arguments differ from preregistration")
    return value


def mac_hardware() -> dict[str, str | None]:
    def read(name: str) -> str | None:
        try:
            return subprocess.check_output(
                ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "model": read("hw.model"),
        "cpu_brand": read("machdep.cpu.brand_string"),
        "physical_cpus": read("hw.physicalcpu"),
        "performance_cpus": read("hw.perflevel0.physicalcpu"),
        "efficiency_cpus": read("hw.perflevel1.physicalcpu"),
        "memory_bytes": read("hw.memsize"),
    }


def main() -> None:
    default_root = Path("/Users/stan/.cache/clark-quantize/qwen38")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture", type=Path,
        default=default_root / "qwen38-layer0-gate-proj-fixture-v1.npz",
    )
    parser.add_argument(
        "--fixture-receipt", type=Path,
        default=Path(__file__).with_name("receipts")
        / "qwen38-layer0-gate-proj-fixture-v1.json",
    )
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path(__file__).with_name("gemv_arm_preregistration_v1.json"),
    )
    parser.add_argument(
        "--receipt", required=True, type=Path,
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument(
        "--threads", type=lambda value: [int(item) for item in value.split(",")],
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--cache-eviction-bytes", type=int, default=536870912)
    args = parser.parse_args()
    if not args.receipt.is_absolute():
        parser.error("--receipt must be absolute")

    preregistration_content = args.preregistration.read_bytes()
    preregistration = validate_preregistration(
        json.loads(preregistration_content), args
    )
    fixture_receipt = json.loads(args.fixture_receipt.read_text())
    frozen_fixture = preregistration["fixture"]
    if (
        fixture_receipt.get("schema") != frozen_fixture["schema"]
        or fixture_receipt.get("model_revision") != preregistration["model_revision"]
        or file_sha256(args.fixture) != frozen_fixture["file_sha256"]
        or file_sha256(args.fixture_receipt) != frozen_fixture["receipt_file_sha256"]
        or fixture_receipt.get("fixture", {}).get("sha256")
        != frozen_fixture["file_sha256"]
    ):
        raise ValueError("real-matrix fixture identity differs from preregistration")
    features = cpu_features()
    if not native_available() or not features["neon"] or not features["dotprod"]:
        raise RuntimeError("the preregistered run requires native NEON DOTPROD")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("the v1 ARM receipt is scoped to Darwin arm64")

    with np.load(args.fixture, allow_pickle=False) as fixture:
        codes = np.ascontiguousarray(fixture["codes"], dtype=np.int8)
        weight_scales = np.ascontiguousarray(fixture["scales"], dtype=np.float16)
    tensor = preregistration["tensor"]
    if (
        codes.shape != (tensor["rows"], tensor["columns"])
        or weight_scales.shape != (tensor["rows"], tensor["columns"] // 128)
        or hashlib.sha256(codes.tobytes()).hexdigest()
        != frozen_fixture["codes_sha256"]
        or hashlib.sha256(weight_scales.tobytes()).hexdigest()
        != frozen_fixture["scales_sha256"]
    ):
        raise ValueError("fixture arrays differ from preregistration")

    packed_flat = pack_group128_2bit(codes.reshape(-1))
    packed = packed_flat.reshape(tensor["rows"], tensor["columns"] // 4)
    decoded = unpack_group128_2bit(packed_flat, codes.size).reshape(codes.shape)
    exact_pack = bool(np.array_equal(decoded, codes))
    del decoded, codes
    generator = np.random.default_rng(preregistration["activation"]["seed"])
    activation = generator.normal(size=tensor["columns"]).astype(np.float32)
    activation_q8, activation_scales, activation_sums = quantize_q8(activation)

    correctness_rows = preregistration["correctness"]["rows"]
    scalar_correctness = gemv_q8_group128_2bit(
        packed[:correctness_rows], weight_scales[:correctness_rows],
        activation_q8, activation_scales, activation_sums,
        implementation="scalar", validated=True,
    )
    arm_correctness = gemv_q8_group128_2bit(
        packed[:correctness_rows], weight_scales[:correctness_rows],
        activation_q8, activation_scales, activation_sums,
        implementation="arm_dotprod", validated=True,
    )
    max_abs_error = float(np.max(np.abs(scalar_correctness - arm_correctness)))
    max_rel_error = float(np.max(
        np.abs(scalar_correctness - arm_correctness)
        / np.maximum(np.abs(scalar_correctness), 1e-12)
    ))

    eviction = np.arange(args.cache_eviction_bytes, dtype=np.uint8)
    expected_eviction_sum = int(eviction.sum(dtype=np.uint64))
    executors = {
        threads: ThreadPoolExecutor(max_workers=threads)
        for threads in args.threads if threads > 1
    }

    def stream(threads: int) -> int:
        return memory_read_sum(
            eviction, implementation="neon", threads=threads,
            executor=executors.get(threads),
        )

    stream_results = {}
    for threads in args.threads:
        samples, checksum = measure(
            lambda threads=threads: stream(threads),
            args.warmups, args.repetitions, lambda: None,
        )
        if checksum != expected_eviction_sum:
            raise AssertionError("NEON read control checksum changed")
        summary = timing_summary(samples, eviction.nbytes)
        summary["gbytes_per_second"] = summary.pop(
            "effective_weight_gbytes_per_second"
        )
        stream_results[str(threads)] = summary

    def evict() -> None:
        if stream(max(args.threads)) != expected_eviction_sum:
            raise AssertionError("cache-eviction checksum changed")

    scalar_rows = preregistration["performance"]["scalar_comparison_rows"]
    scalar_samples, scalar_output = measure(
        lambda: gemv_q8_group128_2bit(
            packed[:scalar_rows], weight_scales[:scalar_rows],
            activation_q8, activation_scales, activation_sums,
            implementation="scalar", validated=True,
        ),
        args.warmups, args.repetitions, evict,
    )
    arm_subset_samples, arm_subset_output = measure(
        lambda: gemv_q8_group128_2bit(
            packed[:scalar_rows], weight_scales[:scalar_rows],
            activation_q8, activation_scales, activation_sums,
            implementation="arm_dotprod", validated=True,
        ),
        args.warmups, args.repetitions, evict,
    )
    subset_max_abs_error = float(
        np.max(np.abs(scalar_output - arm_subset_output))
    )
    scalar_speedup = (
        statistics.median(scalar_samples)
        / statistics.median(arm_subset_samples)
    )

    weight_bytes = packed.nbytes + weight_scales.nbytes
    kernel_results = {}
    deterministic = True
    reference_output = None
    for threads in args.threads:
        samples, output = measure(
            lambda threads=threads: gemv_q8_group128_2bit(
                packed, weight_scales, activation_q8,
                activation_scales, activation_sums,
                implementation="arm_dotprod", threads=threads,
                executor=executors.get(threads), validated=True,
            ),
            args.warmups, args.repetitions, evict,
        )
        if reference_output is None:
            reference_output = output.copy()
        else:
            deterministic = deterministic and bool(
                np.array_equal(reference_output, output)
            )
        summary = timing_summary(samples, weight_bytes)
        summary["fraction_of_stream_read_bandwidth"] = (
            summary["effective_weight_gbytes_per_second"]
            / stream_results[str(threads)]["gbytes_per_second"]
        )
        summary["output_sha256"] = hashlib.sha256(output.tobytes()).hexdigest()
        kernel_results[str(threads)] = summary
    for executor in executors.values():
        executor.shutdown()

    numerical_limit = preregistration["correctness"]["maximum_absolute_error"]
    compute_bits = weight_bytes * 8 / (tensor["rows"] * tensor["columns"])
    rules = preregistration["promotion"]
    promotion_passed = (
        exact_pack
        and max_abs_error <= numerical_limit
        and subset_max_abs_error <= numerical_limit
        and deterministic
        and scalar_speedup >= rules["minimum_arm_speedup_over_scalar"]
        and kernel_results["4"]["fraction_of_stream_read_bandwidth"]
        >= rules["minimum_four_thread_fraction_of_stream_read_bandwidth"]
        and kernel_results["4"]["effective_weight_gbytes_per_second"]
        >= rules["minimum_four_thread_effective_weight_gbytes_per_second"]
        and kernel_results["8"]["effective_weight_gbytes_per_second"]
        >= rules["minimum_eight_thread_effective_weight_gbytes_per_second"]
        and compute_bits
        <= rules["maximum_compute_image_bits_per_weight_including_fp16_scales"]
    )
    native_library = REPO / "clark_quantize/_ternary_cpu.dylib"
    report = {
        "schema": "clark_qwen38_ternary_gemv_arm_v1",
        "hypothesis": (
            "An aligned-2 A8-by-ternary NEON DOTPROD kernel can preserve the "
            "exact real Qwen code/scale contract, exceed 20x the scalar oracle, "
            "and sustain at least 50 GB/s on four M5 Max CPU workers."
        ),
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(preregistration_content).hexdigest(),
            "rules": preregistration,
        },
        "fixture": {
            "path": str(args.fixture),
            "sha256": file_sha256(args.fixture),
            "receipt": str(args.fixture_receipt),
            "receipt_sha256": file_sha256(args.fixture_receipt),
            "parent_artifact": fixture_receipt["artifact"],
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
            "cpu_features": features,
            "hardware": mac_hardware(),
        },
        "implementation": {
            "c_source_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.c"),
            "python_wrapper_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.py"),
            "benchmark_sha256": file_sha256(Path(__file__)),
            "build_script_sha256": file_sha256(REPO / "scripts/cpu/build_ternary_cpu.sh"),
            "native_library_sha256": file_sha256(native_library),
            "compiler": subprocess.check_output(
                [os.environ.get("CC", "cc"), "--version"], text=True
            ).splitlines()[0],
        },
        "tensor": {
            **tensor,
            "codes_sha256": frozen_fixture["codes_sha256"],
            "scales_sha256": frozen_fixture["scales_sha256"],
        },
        "compute_image": {
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
        },
        "correctness": {
            "rows": correctness_rows,
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
            "arm_samples_seconds": arm_subset_samples,
            "arm_speedup_over_scalar": scalar_speedup,
        },
        "kernel": kernel_results,
        "promotion": {
            "passed": promotion_passed,
            "scope": (
                "K2 M5 Max aligned-2 ARM DOTPROD microkernel v1 only; the "
                "70%-roofline/full-graph, quality, energy, and breakthrough "
                "gates remain open"
            ),
        },
    }
    atomic_bytes(
        args.receipt,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if promotion_passed else 3)


if __name__ == "__main__":
    main()
