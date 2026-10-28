#!/usr/bin/env python3
"""Gate a derived aligned-2 CPU cache from the real Qwen3.8-27B container."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen38_cq3_runtime import Qwen38CQ3RuntimeImage
from clark_quantize.ternary_cpu import cpu_features, native_available, quantize_q8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_status() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
            name, value, unit = line.split()
            if unit != "kB":
                raise RuntimeError("unexpected /proc memory unit")
            values[name.rstrip(":")] = int(value) * 1024
    if set(values) != {"VmRSS", "VmHWM", "VmSize"}:
        raise SystemExit("infrastructure-invalid: Linux memory status is absent")
    return values


def physical_cpu_ids() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise SystemExit("infrastructure-invalid: CPU affinity is unavailable")
    selected: list[int] = []
    seen: set[tuple[str, str]] = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (
            (topology / "physical_package_id").read_text().strip(),
            (topology / "core_id").read_text().strip(),
        )
        if key not in seen:
            seen.add(key)
            selected.append(cpu)
    return selected


def validate_frozen_inputs(preregistration: dict) -> None:
    for label, record in preregistration["frozen_inputs"].items():
        path = REPO / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen aligned2 cache input changed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-cq3-row-normalized-container-v1/"
            "Qwen3.8-27B-row-normalized.cq3"
        ),
    )
    parser.add_argument(
        "--mmap-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-cq3-runtime-mmap-v1.json"
        ),
    )
    parser.add_argument(
        "--export-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-cq3-row-normalized-export-v1.json"
        ),
    )
    parser.add_argument(
        "--mlp-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-cq3-layer0-mlp-v1.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-cache-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_aligned2_cache_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if not native_available() or not native_library or not Path(native_library).is_file():
        raise SystemExit("infrastructure-invalid: pinned native library is absent")
    features = cpu_features()
    if not all(features[name] for name in ("avx2", "f16c", "fma")):
        raise SystemExit("infrastructure-invalid: AVX2/F16C/FMA are required")
    threads = int(prereg["measurement"]["physical_threads"])
    cpu_ids = physical_cpu_ids()
    if len(cpu_ids) < threads:
        raise SystemExit(
            f"infrastructure-invalid: only {len(cpu_ids)} physical CPUs are available"
        )
    cpu_ids = cpu_ids[:threads]

    mmap_parent = json.loads(args.mmap_receipt.read_text())
    export_parent = json.loads(args.export_receipt.read_text())
    if (
        mmap_parent.get("schema") != "qwen38_27b_cq3_runtime_mmap_receipt_v1"
        or mmap_parent.get("gate", {}).get("passed") is not True
        or export_parent.get("schema")
        != "qwen38_27b_cq3_row_normalized_export_receipt_v1"
        or export_parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("aligned2 cache parents are not passed frozen receipts")
    mlp_parent = (
        json.loads(args.mlp_receipt.read_text())
        if args.mlp_receipt.is_file()
        else None
    )
    artifact = export_parent["artifact"]
    if (
        args.container.resolve() != Path(artifact["path"]).resolve()
        or args.container.stat().st_size != int(artifact["bytes"])
    ):
        raise RuntimeError("aligned2 cache container identity differs from its parents")
    container_sha256 = sha256_file(args.container)
    if (
        container_sha256 != artifact["sha256"]
        or container_sha256 != mmap_parent["container"]["sha256"]
    ):
        raise RuntimeError("aligned2 cache container bytes differ from its parents")

    before = memory_status()
    tensor_name = prereg["scope"]["tensor"]
    with Qwen38CQ3RuntimeImage(args.container) as image:
        direct = image.ternary_tensor(tensor_name)
        if direct.shape != tuple(prereg["scope"]["shape"]):
            raise RuntimeError("aligned2 cache tensor shape changed")
        started = time.perf_counter_ns()
        aligned = direct.derive_aligned2()
        transcode_seconds = (time.perf_counter_ns() - started) / 1e9
        repeated = direct.derive_aligned2()
        exact_transcode = bool(
            np.array_equal(aligned.packed_weights, repeated.packed_weights)
        )
        del repeated
        if (
            aligned.packed_weights.nbytes
            != int(prereg["scope"]["aligned2_weight_bytes"])
        ):
            raise RuntimeError("aligned2 cache tensor byte ledger changed")
        mapped = memory_status()
        generator = np.random.default_rng(int(prereg["candidate"]["seed"]))
        activation = generator.normal(size=direct.columns).astype(np.float32)
        activation_q8, activation_scales, activation_sums = quantize_q8(
            activation
        )

        direct_one = lambda: direct.gemv_q8(
            activation_q8,
            activation_scales,
            activation_sums,
            implementation="avx2_chunk16",
            threads=1,
            cpu_ids=cpu_ids[:1],
        )
        aligned_one = lambda: aligned.gemv_q8(
            activation_q8,
            activation_scales,
            activation_sums,
            implementation="avx2",
            threads=1,
            cpu_ids=cpu_ids[:1],
        )
        with ThreadPoolExecutor(max_workers=threads) as executor:
            direct_many = lambda: direct.gemv_q8(
                activation_q8,
                activation_scales,
                activation_sums,
                implementation="avx2_chunk16",
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
            )
            aligned_many = lambda: aligned.gemv_q8(
                activation_q8,
                activation_scales,
                activation_sums,
                implementation="avx2",
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
            )
            functions = (
                ("direct_one", direct_one),
                ("direct_four", direct_many),
                ("aligned_one", aligned_one),
                ("aligned_four", aligned_many),
            )
            for _ in range(int(prereg["measurement"]["warmups_each"])):
                for _, function in functions:
                    function()
            samples = {name: [] for name, _ in functions}
            outputs: dict[str, np.ndarray] = {}
            repetitions = int(prereg["measurement"]["rotating_repetitions"])
            for repetition in range(repetitions):
                offset = repetition % len(functions)
                order = functions[offset:] + functions[:offset]
                for name, function in order:
                    started = time.perf_counter_ns()
                    output = function()
                    elapsed = (time.perf_counter_ns() - started) / 1e9
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise RuntimeError("non-finite aligned2 cache timing")
                    samples[name].append(elapsed)
                    outputs[name] = output
        after = memory_status()

    medians = {name: statistics.median(values) for name, values in samples.items()}
    aligned_speedup_over_direct = medians["direct_four"] / medians["aligned_four"]
    aligned_four_thread_speedup = medians["aligned_one"] / medians["aligned_four"]
    maximum_error = float(
        np.max(
            np.abs(outputs["aligned_four"] - outputs["direct_four"]),
            initial=0.0,
        )
    )
    exact_threading = bool(
        np.array_equal(outputs["aligned_one"], outputs["aligned_four"])
    )
    tensor_parameters = direct.rows * direct.columns
    transcode_weights_per_second = tensor_parameters / transcode_seconds
    projected_seconds = (
        int(prereg["whole_model_projection"]["ternary_parameters"])
        / transcode_weights_per_second
    )
    runtime_bytes = int(prereg["whole_model_projection"]["runtime_bytes"])
    parameters = int(prereg["whole_model_projection"]["parameters"])
    runtime_bpw = runtime_bytes * 8 / parameters
    rss_growth = max(0, after["VmHWM"] - before["VmHWM"])
    finite = bool(
        math.isfinite(transcode_seconds)
        and transcode_seconds > 0
        and all(
            math.isfinite(value) and value > 0
            for values in samples.values()
            for value in values
        )
        and all(np.all(np.isfinite(output)) for output in outputs.values())
    )
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_exact_repeated_transcode": exact_transcode,
        "observed_maximum_aligned_vs_direct_absolute_error": maximum_error,
        "observed_exact_aligned_one_vs_four_thread_output": exact_threading,
        "observed_aligned_speedup_over_direct": aligned_speedup_over_direct,
        "observed_aligned_four_thread_speedup": aligned_four_thread_speedup,
        "observed_projected_whole_model_transcode_seconds": projected_seconds,
        "observed_runtime_bits_per_weight": runtime_bpw,
        "observed_peak_rss_growth_bytes": rss_growth,
        "observed_all_outputs_and_samples_finite": finite,
    }
    gate["passed"] = (
        exact_transcode
        and maximum_error <= gate_spec["maximum_aligned_vs_direct_absolute_error"]
        and exact_threading
        and aligned_speedup_over_direct
        >= gate_spec["minimum_aligned_speedup_over_direct"]
        and aligned_four_thread_speedup
        >= gate_spec["minimum_aligned_four_thread_speedup"]
        and projected_seconds
        <= gate_spec["maximum_projected_whole_model_transcode_seconds"]
        and runtime_bpw <= gate_spec["maximum_runtime_bits_per_weight"]
        and rss_growth <= gate_spec["maximum_peak_rss_growth_bytes"]
        and finite
    )
    receipt = {
        "schema": "qwen38_27b_aligned2_cache_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "native_library": {
            "path": native_library,
            "sha256": sha256_file(Path(native_library)),
            "features": features,
        },
        "parents": {
            "mmap_receipt": str(args.mmap_receipt),
            "mmap_receipt_sha256": sha256_file(args.mmap_receipt),
            "export_receipt": str(args.export_receipt),
            "export_receipt_sha256": sha256_file(args.export_receipt),
            "direct_mlp_receipt": str(args.mlp_receipt) if mlp_parent else None,
            "direct_mlp_gate_pass": (
                mlp_parent.get("gate", {}).get("passed") is True
                if mlp_parent
                else None
            ),
        },
        "container": {
            "path": str(args.container),
            "bytes": args.container.stat().st_size,
            "sha256": container_sha256,
            "complete_bits_per_parameter": export_parent["gate"][
                "observed_complete_container_bpw"
            ],
        },
        "tensor": {
            "name": tensor_name,
            "shape": list(direct.shape),
            "parameters": tensor_parameters,
            "block203_bytes": direct.packed_weights.nbytes,
            "aligned2_bytes": aligned.packed_weights.nbytes,
            "aligned2_sha256": hashlib.sha256(
                aligned.packed_weights.tobytes()
            ).hexdigest(),
        },
        "transcode": {
            "seconds": transcode_seconds,
            "weights_per_second": transcode_weights_per_second,
            "projected_whole_model_seconds": projected_seconds,
            "full_int8_weight_matrix_materialized": False,
        },
        "throughput": {
            "samples_seconds": samples,
            "medians_seconds": medians,
            "aligned_speedup_over_direct": aligned_speedup_over_direct,
            "aligned_four_thread_speedup": aligned_four_thread_speedup,
        },
        "correctness": {
            "exact_repeated_transcode": exact_transcode,
            "maximum_aligned_vs_direct_absolute_error": maximum_error,
            "exact_aligned_one_vs_four_thread_output": exact_threading,
        },
        "whole_model_projection": {
            **prereg["whole_model_projection"],
            "runtime_bits_per_weight": runtime_bpw,
            "canonical_storage_container_remains_source_of_truth": True,
        },
        "memory": {
            "before": before,
            "after_transcode": mapped,
            "after_measurement": after,
            "peak_rss_growth_bytes": rss_growth,
        },
        "cpu": {
            "physical_cpu_ids": cpu_ids,
            "allowed_cpu_ids": sorted(os.sched_getaffinity(0)),
        },
        "gate": gate,
        "claim_boundary": {
            "real_qwen38_27b_tensor_cache_derived": True,
            "whole_model_cache_materialized": False,
            "whole_model_graph_runnable": False,
            "capability_or_perplexity_proven": False,
            "end_to_end_tokens_per_second_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(
        args.receipt,
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
