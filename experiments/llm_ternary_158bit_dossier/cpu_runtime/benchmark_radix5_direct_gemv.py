#!/usr/bin/env python3
"""Compare direct group-128 radix-5 GEMV with the aligned-two-bit control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader
from clark_quantize.ternary_cpu import (
    cpu_features,
    expand_q8_radix5,
    gemv_q8_group128_2bit,
    gemv_q8_group128_radix5,
    memory_read_sum,
    native_available,
    pack_group128_2bit,
    pack_group128_radix5,
    quantize_q8,
    unpack_group128_radix5,
)
from benchmark_ternary_gemv import (
    file_sha256,
    load_tensor_codes,
    physical_cpu_ids,
    pinned_executor,
)


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_preregistration(value: dict, args: argparse.Namespace) -> None:
    tensor = value.get("tensor", {})
    performance = value.get("performance", {})
    if (
        value.get("schema") != "clark_radix5_direct_gemv_preregistration_v1"
        or tensor.get("name") != args.tensor
        or tensor.get("rows") != args.rows
        or tensor.get("columns") != args.columns
        or tensor.get("group_size") != 128
        or performance.get("warmups") != args.warmups
        or performance.get("repetitions") != args.repetitions
        or performance.get("threads") != args.threads
        or performance.get("cache_eviction_bytes") != args.cache_eviction_bytes
    ):
        raise ValueError("radix-5 GEMV arguments differ from preregistration")


def summarize(samples: list[float], weight_bytes: int) -> dict:
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "effective_weight_gbytes_per_second": weight_bytes / median / 1e9,
    }


def main() -> None:
    default_root = Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q")
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=default_root)
    parser.add_argument("--pack-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-full.json"))
    parser.add_argument("--receipt", type=Path, default=Path("/workspace/quant-state/qwen38-radix5-direct-gemv-v1.json"))
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("radix5_direct_gemv_preregistration_v1.json"))
    parser.add_argument("--tensor", default="model.language_model.layers.0.mlp.gate_proj.weight")
    parser.add_argument("--rows", type=int, default=17408)
    parser.add_argument("--columns", type=int, default=5120)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--threads", type=lambda value: [int(item) for item in value.split(",")], default=[1, 2, 4])
    parser.add_argument("--cache-eviction-bytes", type=int, default=268435456)
    args = parser.parse_args()

    preregistration_content = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_content)
    validate_preregistration(preregistration, args)
    features = cpu_features()
    if not native_available() or not all(features[name] for name in ("avx2", "f16c", "fma")):
        raise RuntimeError("the preregistered run requires native AVX2, F16C, and FMA")
    physical_cpus = physical_cpu_ids()
    if len(physical_cpus) < max(args.threads):
        raise RuntimeError("the preregistered run requires four distinct physical CPU cores")

    pack_receipt = json.loads(args.pack_receipt.read_text())
    if (
        pack_receipt.get("promotion", {}).get("passed") is not True
        or pack_receipt.get("identity", {}).get("model_revision") != preregistration["model_revision"]
    ):
        raise ValueError("full-pack receipt is not the promoted preregistered parent")
    reader = QwenBase3ArtifactReader(args.artifact)
    if reader.manifest.get("schema") != preregistration["artifact_schema"]:
        raise ValueError("artifact schema differs from preregistration")
    codes, weight_scales, record_indices = load_tensor_codes(
        reader, args.tensor, args.rows, args.columns
    )
    groups = args.columns // 128
    packed_aligned = pack_group128_2bit(codes.reshape(-1)).reshape(args.rows, groups * 32)
    packed_radix5 = pack_group128_radix5(codes.reshape(-1)).reshape(args.rows, groups * 26)
    decoded = unpack_group128_radix5(packed_radix5.reshape(-1), codes.size).reshape(codes.shape)
    exact_roundtrip = bool(np.array_equal(decoded, codes))
    del decoded

    generator = np.random.default_rng(preregistration["activation"]["seed"])
    activation = generator.normal(size=args.columns).astype(np.float32)
    activation_q8, activation_scales, activation_sums = quantize_q8(activation)
    activation_radix5 = expand_q8_radix5(activation_q8)
    check_rows = int(preregistration["correctness"]["rows"])
    aligned_scalar = gemv_q8_group128_2bit(
        packed_aligned[:check_rows], weight_scales[:check_rows],
        activation_q8, activation_scales, activation_sums,
        implementation="scalar", validated=True,
    )
    radix5_scalar = gemv_q8_group128_radix5(
        packed_radix5[:check_rows], weight_scales[:check_rows],
        activation_radix5, activation_scales, activation_sums,
        implementation="scalar", validated=True,
    )
    radix5_avx2 = gemv_q8_group128_radix5(
        packed_radix5[:check_rows], weight_scales[:check_rows],
        activation_radix5, activation_scales, activation_sums,
        implementation="avx2", validated=True,
    )
    scalar_max_abs = float(np.max(np.abs(radix5_scalar - aligned_scalar)))
    avx2_max_abs = float(np.max(np.abs(radix5_avx2 - aligned_scalar)))

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
            raise AssertionError("cache-eviction checksum changed")

    aligned_weight_bytes = packed_aligned.nbytes + weight_scales.nbytes
    radix5_weight_bytes = packed_radix5.nbytes + weight_scales.nbytes
    results = {}
    deterministic = True
    cross_format_max_abs = 0.0
    reference_by_threads: dict[int, np.ndarray] = {}
    for threads in args.threads:
        operations = {
            "aligned2": lambda threads=threads: gemv_q8_group128_2bit(
                packed_aligned, weight_scales, activation_q8,
                activation_scales, activation_sums,
                implementation="avx2", threads=threads,
                executor=executors.get(threads),
                cpu_ids=None if threads > 1 else physical_cpus[:1],
                validated=True,
            ),
            "radix5": lambda threads=threads: gemv_q8_group128_radix5(
                packed_radix5, weight_scales, activation_radix5,
                activation_scales, activation_sums,
                implementation="avx2", threads=threads,
                executor=executors.get(threads),
                cpu_ids=None if threads > 1 else physical_cpus[:1],
                validated=True,
            ),
        }
        for _ in range(args.warmups):
            operations["aligned2"]()
            operations["radix5"]()
        samples = {"aligned2": [], "radix5": []}
        outputs = {"aligned2": [], "radix5": []}
        for repetition in range(args.repetitions):
            order = ("radix5", "aligned2") if repetition % 2 == 0 else ("aligned2", "radix5")
            for name in order:
                evict()
                started = time.perf_counter()
                output = operations[name]()
                samples[name].append(time.perf_counter() - started)
                outputs[name].append(output)
        for name in outputs:
            deterministic = deterministic and all(
                np.array_equal(outputs[name][0], output) for output in outputs[name][1:]
            )
        reference_by_threads[threads] = outputs["aligned2"][0]
        cross_format_max_abs = max(
            cross_format_max_abs,
            float(np.max(np.abs(outputs["radix5"][0] - outputs["aligned2"][0]))),
        )
        aligned_summary = summarize(samples["aligned2"], aligned_weight_bytes)
        radix5_summary = summarize(samples["radix5"], radix5_weight_bytes)
        radix5_summary["throughput_fraction_of_aligned2"] = (
            aligned_summary["median_seconds"] / radix5_summary["median_seconds"]
        )
        aligned_summary["output_sha256"] = hashlib.sha256(outputs["aligned2"][0].tobytes()).hexdigest()
        radix5_summary["output_sha256"] = hashlib.sha256(outputs["radix5"][0].tobytes()).hexdigest()
        results[str(threads)] = {"aligned2": aligned_summary, "radix5": radix5_summary}
    for executor in executors.values():
        executor.shutdown()
    deterministic = deterministic and all(
        np.array_equal(reference_by_threads[min(args.threads)], output)
        for output in reference_by_threads.values()
    )

    aligned_bits = aligned_weight_bytes * 8 / codes.size
    radix5_bits = radix5_weight_bytes * 8 / codes.size
    rules = preregistration["promotion"]
    numerical_limit = preregistration["correctness"]["maximum_absolute_error_vs_aligned2"]
    maximum_threads = str(max(args.threads))
    performance_fraction = results[maximum_threads]["radix5"]["throughput_fraction_of_aligned2"]
    passed = (
        exact_roundtrip
        and deterministic
        and scalar_max_abs <= numerical_limit
        and avx2_max_abs <= numerical_limit
        and cross_format_max_abs <= numerical_limit
        and radix5_bits <= rules["maximum_radix5_compute_image_bits_per_weight_including_fp16_scales"]
        and performance_fraction >= rules["minimum_radix5_throughput_fraction_of_aligned2"]
    )
    native_library = next(
        path for path in (REPO / "clark_quantize/_ternary_cpu.so", REPO / "clark_quantize/_ternary_cpu.dylib")
        if path.exists()
    )
    report = {
        "schema": "clark_qwen38_radix5_direct_gemv_v1",
        "created_at": time.time(),
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
            "tensor": args.tensor,
            "record_indices": record_indices,
            "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
        },
        "build": {
            "native_library": str(native_library),
            "native_library_sha256": file_sha256(native_library),
            "c_source_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.c"),
            "python_wrapper_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.py"),
            "benchmark_sha256": file_sha256(Path(__file__)),
            "cpu_features": features,
            "physical_cpu_ids": physical_cpus,
            "affinity": sorted(os.sched_getaffinity(0)),
        },
        "format": {
            "aligned2_weight_bytes": aligned_weight_bytes,
            "radix5_weight_bytes": radix5_weight_bytes,
            "aligned2_bits_per_weight_including_scales": aligned_bits,
            "radix5_bits_per_weight_including_scales": radix5_bits,
            "radix5_byte_reduction_percent": 100.0 * (1.0 - radix5_weight_bytes / aligned_weight_bytes),
            "exact_radix5_roundtrip": exact_roundtrip,
        },
        "correctness": {
            "scalar_radix5_vs_aligned2_max_abs": scalar_max_abs,
            "avx2_radix5_vs_aligned2_max_abs": avx2_max_abs,
            "full_matrix_cross_format_max_abs": cross_format_max_abs,
            "thread_deterministic": deterministic,
        },
        "performance": results,
        "promotion": {
            "passed": passed,
            "four_core_throughput_fraction_of_aligned2": performance_fraction,
            "four_core_pareto_speed_win": performance_fraction >= rules["pareto_speed_threshold"],
            "scope": "direct radix-5 K2 prototype only; no model-quality or breakthrough claim",
        },
    }
    report["report_content_sha256"] = canonical_digest(report)
    atomic_bytes(args.receipt, json.dumps(report, indent=2, ensure_ascii=False).encode() + b"\n")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
