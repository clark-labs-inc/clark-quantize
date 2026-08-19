#!/usr/bin/env python3
"""Block-median stability confirmation for the unchanged v38 AVX2 kernel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

import benchmark_qwen38_27b_calibration_residual_cpu_v38 as v38

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_calibration_span_residual import load_int8_adapter
from clark_quantize.qwen38_calibration_residual_cpu import (
    load_calibration_residual_cpu_library,
    native_calibration_residual_f32,
)


def measure_block(
    function: Callable[[], int], *, warmups: int, samples: int, inner: int
) -> dict[str, Any]:
    for _ in range(warmups):
        for _ in range(inner):
            if function() != 0:
                raise RuntimeError("v39 trusted native warmup failed")
    observed = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        for _ in range(inner):
            if function() != 0:
                raise RuntimeError("v39 trusted native timing failed")
        observed.append((time.perf_counter_ns() - started) / inner)
    ordered = sorted(observed)
    median = float(statistics.median(ordered))
    return {
        "samples_ns": observed,
        "median_ns": median,
        "minimum_ns": float(ordered[0]),
        "maximum_ns": float(ordered[-1]),
        "p90_ns": float(ordered[math.ceil(0.9 * len(ordered)) - 1]),
    }


def summarize_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [float(block["median_ns"]) for block in blocks]
    median = float(statistics.median(medians))
    return {
        "blocks": blocks,
        "block_medians_ns": medians,
        "median_of_block_medians_ns": median,
        "minimum_block_median_ns": min(medians),
        "maximum_block_median_ns": max(medians),
        "block_median_range_fraction": (max(medians) - min(medians)) / median,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-calibration-residual-stability-v39/receipt.json"
        ),
    )
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--warmups-per-block", type=int, default=3)
    parser.add_argument("--samples-per-block", type=int, default=9)
    parser.add_argument("--inner-iterations", type=int, default=128)
    args = parser.parse_args()
    if (
        args.blocks,
        args.warmups_per_block,
        args.samples_per_block,
        args.inner_iterations,
    ) != (7, 3, 9, 128):
        raise ValueError("v39 timing denominator is frozen to 7, 3, 9, and 128")

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_calibration_residual_stability_v39.json"
    prereg = json.loads(prereg_path.read_text())
    if prereg.get("status") != "frozen_after_terminal_v38_before_confirmation_run":
        raise RuntimeError("v39 preregistration is not frozen")
    v38.validate_frozen_sources(prereg)

    parent_decision_path = REPO / prereg["parent"]["terminal_decision_path"]
    parent_receipt_path = REPO / prereg["parent"]["receipt_path"]
    parent_decision = json.loads(parent_decision_path.read_text())
    parent_receipt = json.loads(parent_receipt_path.read_text())
    failed = [
        name for name, passed in parent_receipt["gate"]["checks"].items() if not passed
    ]
    if (
        v38.sha256_file(parent_decision_path)
        != prereg["parent"]["terminal_decision_sha256"]
        or v38.sha256_file(parent_receipt_path) != prereg["parent"]["receipt_sha256"]
        or parent_decision.get("decision")
        != "reject_v38_on_preregistered_full_range_stability_gate"
        or parent_receipt.get("gate", {}).get("passed") is not False
        or failed != ["candidate_sample_range_within_gate"]
    ):
        raise RuntimeError("v39 terminal v38 parent changed")

    native = load_calibration_residual_cpu_library(args.library)
    if native is None:
        raise RuntimeError("v39 native library did not load")
    features = native.features()
    if not all(features.values()):
        raise RuntimeError("v39 authoritative run requires AVX2, FMA, and F16C")
    if v38.sha256_file(args.library) != prereg["unchanged_kernel"]["library_sha256"]:
        raise RuntimeError("v39 compiled kernel binary changed from v38")
    scalar_call = native.library.cq3_calibration_residual_f32_scalar_trusted
    avx2_call = native.library.cq3_calibration_residual_f32_avx2_trusted

    records = []
    for case_index, case in enumerate(prereg["cases"]):
        payload_path = REPO / case["payload_path"]
        if (
            payload_path.stat().st_size != case["payload_bytes"]
            or v38.sha256_file(payload_path) != case["payload_sha256"]
        ):
            raise RuntimeError(f"v39 payload changed for layer {case['layer']}")
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
            values, adapter, library=native, implementation="scalar", base_output=base
        )
        avx2 = native_calibration_residual_f32(
            values, adapter, library=native, implementation="avx2", base_output=base
        )
        maximum_error = max(
            float(np.max(np.abs(scalar - reference))),
            float(np.max(np.abs(avx2 - reference))),
            float(np.max(np.abs(avx2 - scalar))),
        )

        scalar_output = base.copy()
        avx2_output = base.copy()
        scalar_args, scalar_keepalive = v38.native_arguments(
            adapter, values, scalar_output
        )
        avx2_args, avx2_keepalive = v38.native_arguments(adapter, values, avx2_output)
        scalar_blocks = []
        avx2_blocks = []
        block_order = []
        for block_index in range(args.blocks):
            arms = (
                (("scalar", scalar_call, scalar_args), ("avx2", avx2_call, avx2_args))
                if block_index % 2 == 0
                else (("avx2", avx2_call, avx2_args), ("scalar", scalar_call, scalar_args))
            )
            block_order.append([arm[0] for arm in arms])
            for label, function, arguments in arms:
                result = measure_block(
                    lambda function=function, arguments=arguments: function(*arguments),
                    warmups=args.warmups_per_block,
                    samples=args.samples_per_block,
                    inner=args.inner_iterations,
                )
                result["block"] = block_index
                (scalar_blocks if label == "scalar" else avx2_blocks).append(result)
        scalar_timing = summarize_blocks(scalar_blocks)
        avx2_timing = summarize_blocks(avx2_blocks)
        scalar_median = scalar_timing["median_of_block_medians_ns"]
        avx2_median = avx2_timing["median_of_block_medians_ns"]
        records.append(
            {
                "layer": case["layer"],
                "payload_sha256": case["payload_sha256"],
                "maximum_absolute_error": maximum_error,
                "block_order": block_order,
                "scalar_timing": scalar_timing,
                "avx2_timing": avx2_timing,
                "avx2_speedup_vs_scalar": scalar_median / avx2_median,
                "projected_all_64_layers_avx2_ns": avx2_median * 64,
            }
        )
        del scalar_keepalive, avx2_keepalive

    gate_spec = prereg["gate"]
    observed = {
        "maximum_absolute_error": max(row["maximum_absolute_error"] for row in records),
        "minimum_avx2_speedup_vs_scalar": min(
            row["avx2_speedup_vs_scalar"] for row in records
        ),
        "maximum_projected_all_64_layers_avx2_ns": max(
            row["projected_all_64_layers_avx2_ns"] for row in records
        ),
        "maximum_avx2_block_median_range_fraction": max(
            row["avx2_timing"]["block_median_range_fraction"] for row in records
        ),
    }
    checks = {
        "both_exact_payloads_executed": len(records)
        == int(gate_spec["required_exact_payloads"]),
        "numerical_error_within_gate": observed["maximum_absolute_error"]
        <= float(gate_spec["maximum_absolute_error"]),
        "speedup_within_unchanged_gate": observed["minimum_avx2_speedup_vs_scalar"]
        >= float(gate_spec["minimum_avx2_speedup_vs_scalar"]),
        "latency_within_unchanged_gate": observed[
            "maximum_projected_all_64_layers_avx2_ns"
        ]
        <= float(gate_spec["maximum_projected_all_64_layers_avx2_ns"]),
        "block_median_stability_within_gate": observed[
            "maximum_avx2_block_median_range_fraction"
        ]
        <= float(gate_spec["maximum_avx2_block_median_range_fraction"]),
        "seven_blocks_per_arm_per_payload": all(
            len(row["scalar_timing"]["blocks"]) == 7
            and len(row["avx2_timing"]["blocks"]) == 7
            for row in records
        ),
    }
    gate = {
        "specification": gate_spec,
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }
    receipt = {
        "schema": "qwen38_27b_calibration_residual_stability_receipt_v39",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": v38.sha256_file(prereg_path),
        "parent_terminal_decision_sha256": v38.sha256_file(parent_decision_path),
        "parent_receipt_sha256": v38.sha256_file(parent_receipt_path),
        "library_sha256": v38.sha256_file(args.library),
        "cpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
        "features": features,
        "records": records,
        "gate": gate,
        "claim_boundary": {
            "unchanged_v38_kernel_confirmed": gate["passed"],
            "block_median_stability_proven": gate["passed"],
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
