#!/usr/bin/env python3
"""Authoritative AMD benchmark for the frozen v37 rank-16 residual payloads."""

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
from typing import Any, Callable

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_calibration_span_residual import load_int8_adapter
from clark_quantize.qwen38_calibration_residual_cpu import (
    load_calibration_residual_cpu_library,
    native_calibration_residual_f32,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources(preregistration: dict[str, Any]) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen calibration-residual source changed: {label}")


def measure(
    function: Callable[[], int], *, warmups: int, repetitions: int, inner: int
) -> list[float]:
    for _ in range(warmups):
        for _ in range(inner):
            if function() != 0:
                raise RuntimeError("trusted native warmup failed")
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        for _ in range(inner):
            if function() != 0:
                raise RuntimeError("trusted native benchmark failed")
        samples.append((time.perf_counter_ns() - started) / inner)
    return samples


def summarize(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    median = float(statistics.median(ordered))
    return {
        "samples_ns": samples,
        "median_ns": median,
        "minimum_ns": float(ordered[0]),
        "maximum_ns": float(ordered[-1]),
        "p90_ns": float(ordered[math.ceil(0.9 * len(ordered)) - 1]),
        "range_fraction_of_median": float((ordered[-1] - ordered[0]) / median),
    }


def native_arguments(
    adapter: Any, values: np.ndarray, output: np.ndarray
) -> tuple[tuple[Any, ...], tuple[np.ndarray, ...]]:
    left_codes = np.ascontiguousarray(adapter.left_codes.numpy(), dtype=np.int8)
    left_scales = np.ascontiguousarray(adapter.left_scales.numpy(), dtype=np.float16)
    right_codes = np.ascontiguousarray(adapter.right_codes.numpy(), dtype=np.int8)
    right_scales = np.ascontiguousarray(adapter.right_scales.numpy(), dtype=np.float16)
    bias = np.ascontiguousarray(adapter.bias.numpy(), dtype=np.float16)
    scratch = np.empty(16, dtype=np.float32)
    f32p = ctypes.POINTER(ctypes.c_float)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    arguments = (
        values.ctypes.data_as(f32p),
        left_codes.ctypes.data_as(i8p),
        left_scales.view(np.uint16).ctypes.data_as(u16p),
        right_codes.ctypes.data_as(i8p),
        right_scales.view(np.uint16).ctypes.data_as(u16p),
        bias.view(np.uint16).ctypes.data_as(u16p),
        left_codes.shape[0],
        right_codes.shape[1],
        16,
        scratch.ctypes.data_as(f32p),
        output.ctypes.data_as(f32p),
        1,
    )
    keepalive = (
        values,
        left_codes,
        left_scales,
        right_codes,
        right_scales,
        bias,
        scratch,
        output,
    )
    return arguments, keepalive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-calibration-residual-cpu-v38/receipt.json"
        ),
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions-per-half", type=int, default=16)
    parser.add_argument("--inner-iterations", type=int, default=128)
    args = parser.parse_args()
    if (args.warmups, args.repetitions_per_half, args.inner_iterations) != (5, 16, 128):
        raise ValueError("v38 timing denominator is frozen to 5, 16, and 128")

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_calibration_residual_cpu_v38.json"
    prereg = json.loads(prereg_path.read_text())
    if prereg.get("status") != "frozen_before_authoritative_amd_execution":
        raise RuntimeError("v38 preregistration is not frozen")
    validate_frozen_sources(prereg)

    parent_path = REPO / prereg["parent"]["terminal_decision_path"]
    parent = json.loads(parent_path.read_text())
    if (
        parent.get("schema")
        != "qwen38_27b_sensitivity_gated_residual_terminal_decision_v37"
        or parent.get("scientific_gate_passed") is not True
        or parent.get("decision") != "admit_exact_algebra_to_native_kernel_gate_only"
        or sha256_file(parent_path) != prereg["parent"]["terminal_decision_sha256"]
    ):
        raise RuntimeError("v38 positive parent changed")

    native = load_calibration_residual_cpu_library(args.library)
    if native is None:
        raise RuntimeError("v38 native library did not load")
    features = native.features()
    if not all(features.values()):
        raise RuntimeError("v38 authoritative run requires AVX2, FMA, and F16C")
    scalar_call = native.library.cq3_calibration_residual_f32_scalar_trusted
    avx2_call = native.library.cq3_calibration_residual_f32_avx2_trusted

    records = []
    for case_index, case in enumerate(prereg["cases"]):
        payload_path = REPO / case["payload_path"]
        if (
            payload_path.stat().st_size != case["payload_bytes"]
            or sha256_file(payload_path) != case["payload_sha256"]
        ):
            raise RuntimeError(f"v38 payload changed for layer {case['layer']}")
        adapter = load_int8_adapter(
            payload_path,
            input_features=5120,
            output_features=5120,
            rank=16,
            ridge_fraction=0.01,
        )
        generator = np.random.default_rng(
            int(prereg["measurement"]["seed"]) + case_index
        )
        values = generator.normal(0.0, 0.5, size=5120).astype(np.float32)
        base = generator.normal(0.0, 0.5, size=5120).astype(np.float32)
        reference = (
            torch.from_numpy(base)
            + adapter.correction(torch.from_numpy(values)[None, :])[0]
        ).numpy()
        scalar = native_calibration_residual_f32(
            values,
            adapter,
            library=native,
            implementation="scalar",
            base_output=base,
        )
        avx2 = native_calibration_residual_f32(
            values,
            adapter,
            library=native,
            implementation="avx2",
            base_output=base,
        )
        scalar_error = float(np.max(np.abs(scalar - reference)))
        avx2_error = float(np.max(np.abs(avx2 - reference)))
        cross_error = float(np.max(np.abs(avx2 - scalar)))

        scalar_output = base.copy()
        avx2_output = base.copy()
        scalar_args, scalar_keepalive = native_arguments(adapter, values, scalar_output)
        avx2_args, avx2_keepalive = native_arguments(adapter, values, avx2_output)
        scalar_first = measure(
            lambda: scalar_call(*scalar_args),
            warmups=args.warmups,
            repetitions=args.repetitions_per_half,
            inner=args.inner_iterations,
        )
        avx2_first = measure(
            lambda: avx2_call(*avx2_args),
            warmups=args.warmups,
            repetitions=args.repetitions_per_half,
            inner=args.inner_iterations,
        )
        avx2_second = measure(
            lambda: avx2_call(*avx2_args),
            warmups=args.warmups,
            repetitions=args.repetitions_per_half,
            inner=args.inner_iterations,
        )
        scalar_second = measure(
            lambda: scalar_call(*scalar_args),
            warmups=args.warmups,
            repetitions=args.repetitions_per_half,
            inner=args.inner_iterations,
        )
        scalar_timing = summarize(scalar_first + scalar_second)
        avx2_timing = summarize(avx2_first + avx2_second)
        records.append(
            {
                "layer": case["layer"],
                "payload_path": case["payload_path"],
                "payload_bytes": payload_path.stat().st_size,
                "payload_sha256": sha256_file(payload_path),
                "input_features": 5120,
                "output_features": 5120,
                "rank": 16,
                "rank_scratch_bytes": 64,
                "maximum_absolute_error_scalar_vs_pytorch": scalar_error,
                "maximum_absolute_error_avx2_vs_pytorch": avx2_error,
                "maximum_absolute_error_avx2_vs_scalar": cross_error,
                "scalar_timing": scalar_timing,
                "avx2_timing": avx2_timing,
                "avx2_speedup_vs_scalar": scalar_timing["median_ns"]
                / avx2_timing["median_ns"],
                "projected_all_64_layers_avx2_ns": avx2_timing["median_ns"] * 64,
            }
        )
        # Keep the ctypes backing arrays alive through all four timing arms.
        del scalar_keepalive, avx2_keepalive

    gate_spec = prereg["gate"]
    maximum_error = max(
        max(
            record["maximum_absolute_error_scalar_vs_pytorch"],
            record["maximum_absolute_error_avx2_vs_pytorch"],
            record["maximum_absolute_error_avx2_vs_scalar"],
        )
        for record in records
    )
    minimum_speedup = min(record["avx2_speedup_vs_scalar"] for record in records)
    maximum_projected_ns = max(
        record["projected_all_64_layers_avx2_ns"] for record in records
    )
    maximum_range_fraction = max(
        record["avx2_timing"]["range_fraction_of_median"] for record in records
    )
    checks = {
        "both_exact_v37_payloads_executed": len(records)
        == int(gate_spec["required_exact_payloads"]),
        "maximum_absolute_error_within_gate": maximum_error
        <= float(gate_spec["maximum_absolute_error"]),
        "minimum_avx2_speedup_within_gate": minimum_speedup
        >= float(gate_spec["minimum_avx2_speedup_vs_scalar"]),
        "conservative_64_layer_latency_within_gate": maximum_projected_ns
        <= float(gate_spec["maximum_projected_all_64_layers_avx2_ns"]),
        "candidate_sample_range_within_gate": maximum_range_fraction
        <= float(gate_spec["maximum_candidate_sample_range_fraction"]),
        "payload_bytes_exact": all(record["payload_bytes"] == 174144 for record in records),
        "scratch_is_rank16_fp32_only": all(record["rank_scratch_bytes"] == 64 for record in records),
    }
    gate = {
        "specification": gate_spec,
        "observed_maximum_absolute_error": maximum_error,
        "observed_minimum_avx2_speedup_vs_scalar": minimum_speedup,
        "observed_maximum_projected_all_64_layers_avx2_ns": maximum_projected_ns,
        "observed_maximum_candidate_sample_range_fraction": maximum_range_fraction,
        "checks": checks,
        "passed": all(checks.values()),
    }
    receipt = {
        "schema": "qwen38_27b_calibration_residual_cpu_receipt_v38",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parent_terminal_decision_sha256": sha256_file(parent_path),
        "library_sha256": sha256_file(args.library),
        "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
        "cpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
        "features": features,
        "timing_order": ["scalar_a", "avx2_a", "avx2_b", "scalar_b"],
        "records": records,
        "gate": gate,
        "claim_boundary": {
            "exact_trained_v37_payloads_executed": True,
            "no_high_precision_factor_materialization": True,
            "native_rank16_arithmetic_proven": gate["passed"],
            "llama_cpp_integration_proven": False,
            "complete_model_quality_or_speed_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
