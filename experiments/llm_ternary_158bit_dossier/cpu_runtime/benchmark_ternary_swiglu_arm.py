#!/usr/bin/env python3
"""Benchmark preregistered fused ARM ternary gate/up plus SwiGLU."""
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
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from benchmark_ternary_gemv import file_sha256
from benchmark_ternary_gemv_arm import mac_hardware
from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.ternary_cpu import (
    cpu_features,
    gemv_q8_group128_2bit,
    gemv_q8_group128_2bit_swiglu,
    memory_read_sum,
    native_available,
    pack_group128_2bit,
    quantize_q8,
    swiglu_f32,
    unpack_group128_2bit,
)


def load_fixture(path: Path, frozen: dict, rows: int, columns: int):
    if file_sha256(path) != frozen["file_sha256"]:
        raise ValueError(f"fixture hash changed: {path}")
    with np.load(path, allow_pickle=False) as fixture:
        codes = np.ascontiguousarray(fixture["codes"], dtype=np.int8)
        scales = np.ascontiguousarray(fixture["scales"], dtype=np.float16)
    if codes.shape != (rows, columns) or scales.shape != (rows, columns // 128):
        raise ValueError(f"fixture shape changed: {path}")
    if (
        hashlib.sha256(codes.tobytes()).hexdigest() != frozen["codes_sha256"]
        or hashlib.sha256(scales.tobytes()).hexdigest() != frozen["scales_sha256"]
    ):
        raise ValueError(f"fixture array identity changed: {path}")
    packed_flat = pack_group128_2bit(codes.reshape(-1))
    exact = bool(np.array_equal(unpack_group128_2bit(packed_flat, codes.size), codes.reshape(-1)))
    return packed_flat.reshape(rows, columns // 4), scales, exact


def timing(samples: list[float], weight_bytes: int) -> dict:
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "effective_weight_gbytes_per_second": weight_bytes / median / 1e9,
    }


def main() -> None:
    cache = Path("/Users/stan/.cache/clark-quantize/qwen38")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-fixture", type=Path, default=cache / "qwen38-layer0-gate-proj-fixture-v1.npz")
    parser.add_argument("--up-fixture", type=Path, default=cache / "qwen38-layer0-up-proj-fixture-v1.npz")
    parser.add_argument("--gate-receipt", type=Path, default=HERE / "receipts/qwen38-layer0-gate-proj-fixture-v1.json")
    parser.add_argument("--up-receipt", type=Path, default=HERE / "receipts/qwen38-layer0-up-proj-fixture-v1.json")
    parser.add_argument("--preregistration", type=Path, default=HERE / "gemv_swiglu_arm_preregistration_v1.json")
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--threads", type=lambda value: [int(item) for item in value.split(",")], default=[1, 2, 4, 8])
    parser.add_argument("--cache-eviction-bytes", type=int, default=536870912)
    args = parser.parse_args()
    if not args.receipt.is_absolute():
        parser.error("--receipt must be absolute")

    prereg_bytes = args.preregistration.read_bytes()
    prereg = json.loads(prereg_bytes)
    perf = prereg.get("performance", {})
    if (
        prereg.get("schema") != "clark_ternary_cpu_swiglu_arm_preregistration_v1"
        or prereg.get("status") != "frozen_before_authoritative_real-matrix_run"
        or perf.get("warmups") != args.warmups
        or perf.get("repetitions") != args.repetitions
        or perf.get("threads") != args.threads
        or perf.get("cache_eviction_bytes") != args.cache_eviction_bytes
    ):
        raise ValueError("fused SwiGLU arguments differ from preregistration")
    features = cpu_features()
    if (
        not native_available() or not features["neon"] or not features["dotprod"]
        or platform.system() != "Darwin" or platform.machine() != "arm64"
    ):
        raise RuntimeError("the v1 fused run requires Darwin arm64 NEON DOTPROD")

    tensors = prereg["tensors"]
    rows, columns = tensors["rows"], tensors["columns"]
    fixtures = prereg["fixtures"]
    receipt_paths = {"gate": args.gate_receipt, "up": args.up_receipt}
    fixture_paths = {"gate": args.gate_fixture, "up": args.up_fixture}
    fixture_receipts = {}
    for name in ("gate", "up"):
        fixture_receipts[name] = json.loads(receipt_paths[name].read_text())
        if (
            file_sha256(receipt_paths[name]) != fixtures[name]["receipt_file_sha256"]
            or fixture_receipts[name].get("model_revision") != prereg["model_revision"]
            or fixture_receipts[name].get("fixture", {}).get("sha256") != fixtures[name]["file_sha256"]
        ):
            raise ValueError(f"{name} fixture receipt identity changed")
    gate, gate_scales, gate_exact = load_fixture(args.gate_fixture, fixtures["gate"], rows, columns)
    up, up_scales, up_exact = load_fixture(args.up_fixture, fixtures["up"], rows, columns)

    generator = np.random.default_rng(prereg["activation"]["seed"])
    activation = generator.normal(size=columns).astype(np.float32)
    activation_q8, activation_scales, activation_sums = quantize_q8(activation)
    eviction = np.arange(args.cache_eviction_bytes, dtype=np.uint8)
    expected_eviction_sum = int(eviction.sum(dtype=np.uint64))
    executors = {
        threads: ThreadPoolExecutor(max_workers=threads)
        for threads in args.threads if threads > 1
    }

    def evict(threads: int) -> None:
        checksum = memory_read_sum(
            eviction, implementation="neon", threads=threads,
            executor=executors.get(threads),
        )
        if checksum != expected_eviction_sum:
            raise AssertionError("cache-eviction checksum changed")

    def control(threads: int) -> np.ndarray:
        executor = executors.get(threads)
        gate_output = gemv_q8_group128_2bit(
            gate, gate_scales, activation_q8, activation_scales, activation_sums,
            implementation="arm_dotprod", threads=threads, executor=executor,
            validated=True,
        )
        up_output = gemv_q8_group128_2bit(
            up, up_scales, activation_q8, activation_scales, activation_sums,
            implementation="arm_dotprod", threads=threads, executor=executor,
            validated=True,
        )
        return swiglu_f32(gate_output, up_output)

    def candidate(threads: int) -> np.ndarray:
        return gemv_q8_group128_2bit_swiglu(
            gate, gate_scales, up, up_scales,
            activation_q8, activation_scales, activation_sums,
            threads=threads, executor=executors.get(threads), validated=True,
        )

    weight_bytes = gate.nbytes + gate_scales.nbytes + up.nbytes + up_scales.nbytes
    results = {}
    reference = None
    exact_across_threads = True
    for threads in args.threads:
        control_output = control(threads)
        candidate_output = candidate(threads)
        exact_control = bool(np.array_equal(control_output, candidate_output))
        if reference is None:
            reference = candidate_output.copy()
        else:
            exact_across_threads &= bool(np.array_equal(reference, candidate_output))
        for _ in range(args.warmups):
            evict(threads); control(threads)
            evict(threads); candidate(threads)
        samples = {"control": [], "candidate": []}
        last = {}
        for repetition in range(args.repetitions):
            order = (("control", control), ("candidate", candidate))
            if repetition % 2:
                order = tuple(reversed(order))
            for name, operation in order:
                evict(threads)
                started = time.perf_counter()
                last[name] = operation(threads)
                samples[name].append(time.perf_counter() - started)
        if not np.array_equal(last["control"], last["candidate"]):
            exact_control = False
        control_summary = timing(samples["control"], weight_bytes)
        candidate_summary = timing(samples["candidate"], weight_bytes)
        results[str(threads)] = {
            "control": control_summary,
            "candidate": candidate_summary,
            "candidate_speedup": control_summary["median_seconds"] / candidate_summary["median_seconds"],
            "exact_control_candidate_output": exact_control,
            "output_sha256": hashlib.sha256(candidate_output.tobytes()).hexdigest(),
        }
    for executor in executors.values():
        executor.shutdown()

    rules = prereg["promotion"]
    compute_bits = weight_bytes * 8 / (2 * rows * columns)
    passed = (
        gate_exact and up_exact and exact_across_threads
        and all(result["exact_control_candidate_output"] for result in results.values())
        and results["1"]["candidate_speedup"] >= rules["minimum_one_thread_speedup"]
        and results["4"]["candidate_speedup"] >= rules["minimum_four_thread_speedup"]
        and results["8"]["candidate_speedup"] >= rules["minimum_eight_thread_speedup"]
        and compute_bits <= rules["maximum_compute_image_bits_per_weight_including_fp16_scales"]
    )
    native_library = REPO / "clark_quantize/_ternary_cpu.dylib"
    report = {
        "schema": "clark_qwen38_ternary_swiglu_arm_v1",
        "hypothesis": "Fusing real Qwen ternary gate/up projections with SwiGLU preserves exact output and improves the eight-thread median by at least 15 percent without materially regressing one thread.",
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(prereg_bytes).hexdigest(),
            "rules": prereg,
        },
        "fixtures": {
            name: {
                "path": str(fixture_paths[name]),
                "sha256": file_sha256(fixture_paths[name]),
                "receipt": str(receipt_paths[name]),
                "receipt_sha256": file_sha256(receipt_paths[name]),
                "parent_artifact": fixture_receipts[name]["artifact"],
            }
            for name in ("gate", "up")
        },
        "host": {
            "node": platform.node(), "platform": platform.platform(),
            "machine": platform.machine(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "cpu_features": features,
            "hardware": mac_hardware(),
        },
        "implementation": {
            "c_source_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.c"),
            "python_wrapper_sha256": file_sha256(REPO / "clark_quantize/ternary_cpu.py"),
            "benchmark_sha256": file_sha256(Path(__file__)),
            "build_script_sha256": file_sha256(REPO / "scripts/cpu/build_ternary_cpu.sh"),
            "native_library_sha256": file_sha256(native_library),
            "compiler": subprocess.check_output([os.environ.get("CC", "cc"), "--version"], text=True).splitlines()[0],
        },
        "compute_image": {
            "two_projection_weight_bytes": weight_bytes,
            "bits_per_weight_including_fp16_scales": compute_bits,
            "exact_gate_pack_roundtrip": gate_exact,
            "exact_up_pack_roundtrip": up_exact,
        },
        "activation": {
            "q8_sha256": hashlib.sha256(activation_q8.tobytes()).hexdigest(),
            "scale_sha256": hashlib.sha256(activation_scales.tobytes()).hexdigest(),
            "sum_sha256": hashlib.sha256(activation_sums.tobytes()).hexdigest(),
        },
        "correctness": {"exact_across_thread_counts": exact_across_threads},
        "controls": {
            "cache_eviction_bytes": eviction.nbytes,
            "cache_eviction_checksum": expected_eviction_sum,
            "paired_order": perf["paired_order"],
        },
        "threads": results,
        "promotion": {
            "passed": passed,
            "scope": "K4 fused M5 Max gate/up plus SwiGLU operator only; full-graph quality, token speed, energy, and breakthrough gates remain open",
        },
    }
    atomic_bytes(args.receipt, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
