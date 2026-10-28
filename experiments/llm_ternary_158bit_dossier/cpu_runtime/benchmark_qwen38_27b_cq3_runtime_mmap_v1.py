#!/usr/bin/env python3
"""Gate direct mmap execution from the real Qwen3.8-27B CQ3 container."""
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
from clark_quantize.ternary_cpu import (
    cpu_features,
    gemv_q8_group128_block203_row_scale4,
    native_available,
    quantize_q8,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_status() -> dict[str, int]:
    status = Path("/proc/self/status")
    if not status.is_file():
        raise SystemExit("infrastructure-invalid: Linux /proc status is absent")
    values: dict[str, int] = {}
    for line in status.read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
            name, value, unit = line.split()
            if unit != "kB":
                raise RuntimeError("unexpected /proc memory unit")
            values[name.rstrip(":")] = int(value) * 1024
    if set(values) != {"VmRSS", "VmHWM", "VmSize"}:
        raise RuntimeError("incomplete /proc memory status")
    return values


def physical_cpu_ids() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise SystemExit("infrastructure-invalid: CPU affinity is unavailable")
    selected: list[int] = []
    seen: set[tuple[str, str]] = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        except OSError as error:
            raise SystemExit(
                "infrastructure-invalid: CPU topology is unavailable"
            ) from error
        if key not in seen:
            seen.add(key)
            selected.append(cpu)
    return selected


def validate_frozen_inputs(preregistration: dict) -> None:
    for label, record in preregistration["frozen_inputs"].items():
        path = REPO / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen runtime input changed: {label}")


def measure(function, repetitions: int) -> tuple[list[float], np.ndarray]:
    samples: list[float] = []
    output = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = function()
        elapsed = (time.perf_counter_ns() - started) / 1e9
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("non-finite mmap runtime timing")
        samples.append(elapsed)
    assert output is not None
    return samples, output


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
        "--export-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-cq3-row-normalized-export-v1.json"
        ),
    )
    parser.add_argument(
        "--chunk16-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/"
            "qwen38-27b-block203-row-scale4-chunk16-v1.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-cq3-runtime-mmap-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_cq3_runtime_mmap_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if not native_available() or not native_library or not Path(native_library).is_file():
        raise SystemExit("infrastructure-invalid: pinned native library is absent")
    features = cpu_features()
    if not all(features[name] for name in ("avx2", "f16c", "fma")):
        raise SystemExit("infrastructure-invalid: AVX2/F16C/FMA are required")
    cpu_ids = physical_cpu_ids()
    threads = int(prereg["measurement"]["physical_threads"])
    if len(cpu_ids) < threads:
        raise SystemExit(
            f"infrastructure-invalid: only {len(cpu_ids)} physical CPUs are available"
        )
    cpu_ids = cpu_ids[:threads]

    export = json.loads(args.export_receipt.read_text())
    chunk16 = json.loads(args.chunk16_receipt.read_text())
    if (
        export.get("schema")
        != "qwen38_27b_cq3_row_normalized_export_receipt_v1"
        or export.get("gate", {}).get("passed") is not True
        or chunk16.get("schema")
        != "qwen38_27b_block203_row_scale4_chunk16_receipt_v1"
        or chunk16.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("runtime mmap parents are not passed frozen receipts")
    artifact = export["artifact"]
    if args.container.resolve() != Path(artifact["path"]).resolve():
        raise RuntimeError("container path differs from its export receipt")
    if args.container.stat().st_size != int(artifact["bytes"]):
        raise RuntimeError("container byte count differs from its export receipt")
    container_sha256 = sha256_file(args.container)
    if container_sha256 != artifact["sha256"]:
        raise RuntimeError("container SHA-256 differs from its export receipt")

    before = memory_status()
    tensor_name = prereg["scope"]["tensor"]
    with Qwen38CQ3RuntimeImage(args.container) as image:
        view = image.ternary_tensor(tensor_name)
        scope = prereg["scope"]
        if (
            view.shape != tuple(scope["shape"])
            or view.rows != int(scope["rows"])
            or view.columns != int(scope["columns"])
        ):
            raise RuntimeError("real container tensor shape differs from preregistration")
        mapping = image._mapping
        zero_copy = mapping is not None and all(
            np.shares_memory(array, mapping)
            for array in (
                view.packed_weights,
                view.packed_scale_indices,
                view.row_anchors,
                view.ratio_codebook,
            )
        )
        mapped = memory_status()
        generator = np.random.default_rng(int(prereg["candidate"]["seed"]))
        activation = generator.normal(size=view.columns).astype(np.float32)
        activation_q8, activation_scales, activation_sums = quantize_q8(
            activation
        )

        groups = view.columns // 128
        subset_rows = int(prereg["correctness"]["rows"])
        row_weight_bytes = groups * 203 // 8
        row_scale_bytes = groups // 2
        subset_weights = view.packed_weights[: subset_rows * row_weight_bytes]
        subset_indices = view.packed_scale_indices[
            : subset_rows * row_scale_bytes
        ]
        subset_anchors = view.row_anchors[:subset_rows]

        def subset(implementation: str, copied: bool = False) -> np.ndarray:
            weights = subset_weights.copy() if copied else subset_weights
            indices = subset_indices.copy() if copied else subset_indices
            anchors = subset_anchors.copy() if copied else subset_anchors
            codebook = (
                view.ratio_codebook.copy() if copied else view.ratio_codebook
            )
            return gemv_q8_group128_block203_row_scale4(
                weights,
                indices,
                anchors,
                codebook,
                activation_q8,
                activation_scales,
                activation_sums,
                implementation=implementation,
                validated=True,
            )

        subset_generic = subset("scalar")
        subset_chunk16 = subset("scalar_chunk16")
        subset_avx2 = subset("avx2_chunk16")
        subset_copied = subset("avx2_chunk16", copied=True)
        subset_error = float(
            np.max(np.abs(subset_avx2 - subset_generic), initial=0.0)
        )
        exact_scalar = bool(np.array_equal(subset_chunk16, subset_generic))
        exact_mmap_copy = bool(np.array_equal(subset_avx2, subset_copied))

        implementation = prereg["candidate"]["implementation"]
        one = lambda: view.gemv_q8(
            activation_q8,
            activation_scales,
            activation_sums,
            implementation=implementation,
            threads=1,
            cpu_ids=cpu_ids[:1],
        )
        with ThreadPoolExecutor(max_workers=threads) as executor:
            many = lambda: view.gemv_q8(
                activation_q8,
                activation_scales,
                activation_sums,
                implementation=implementation,
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
            )
            for _ in range(int(prereg["measurement"]["warmups_each"])):
                one()
                many()
            one_samples: list[float] = []
            many_samples: list[float] = []
            one_output = many_output = None
            repetitions = int(prereg["measurement"]["paired_repetitions"])
            for repetition in range(repetitions):
                order = ((one_samples, one), (many_samples, many))
                if repetition & 1:
                    order = order[::-1]
                for samples, function in order:
                    measured, output = measure(function, 1)
                    samples.extend(measured)
                    if function is one:
                        one_output = output
                    else:
                        many_output = output
        assert one_output is not None and many_output is not None
        exact_threading = bool(np.array_equal(one_output, many_output))
        finite = bool(
            np.all(np.isfinite(one_output))
            and np.all(np.isfinite(many_output))
            and all(math.isfinite(value) and value > 0 for value in one_samples)
            and all(math.isfinite(value) and value > 0 for value in many_samples)
        )
        after = memory_status()

    one_median = statistics.median(one_samples)
    many_median = statistics.median(many_samples)
    speedup = one_median / many_median
    rss_growth = max(0, after["VmHWM"] - before["VmHWM"])
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_zero_copy_all_runtime_arrays": zero_copy,
        "observed_exact_scalar_chunk16_vs_generic": exact_scalar,
        "observed_exact_mmap_vs_copied_compressed_payload": exact_mmap_copy,
        "observed_maximum_avx2_vs_generic_scalar_absolute_error": subset_error,
        "observed_exact_one_vs_four_thread_output": exact_threading,
        "observed_four_thread_speedup": speedup,
        "observed_peak_rss_growth_bytes": rss_growth,
        "observed_all_outputs_and_samples_finite": finite,
    }
    gate["passed"] = (
        zero_copy
        and exact_scalar
        and exact_mmap_copy
        and subset_error
        <= gate_spec["maximum_avx2_vs_generic_scalar_absolute_error"]
        and exact_threading
        and speedup >= gate_spec["minimum_four_thread_speedup"]
        and rss_growth <= gate_spec["maximum_peak_rss_growth_bytes"]
        and finite
    )
    receipt = {
        "schema": "qwen38_27b_cq3_runtime_mmap_receipt_v1",
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
            "export_receipt": str(args.export_receipt),
            "export_receipt_sha256": sha256_file(args.export_receipt),
            "chunk16_receipt": str(args.chunk16_receipt),
            "chunk16_receipt_sha256": sha256_file(args.chunk16_receipt),
        },
        "container": {
            "path": str(args.container),
            "bytes": args.container.stat().st_size,
            "sha256": container_sha256,
            "complete_bits_per_parameter": export["gate"][
                "observed_complete_container_bpw"
            ],
        },
        "tensor": {
            "name": tensor_name,
            "shape": list(view.shape),
            "block203_bytes": view.packed_weights.nbytes,
            "scale_index_bytes": view.packed_scale_indices.nbytes,
            "row_anchor_bytes": view.row_anchors.nbytes,
            "ratio_codebook_bytes": view.ratio_codebook.nbytes,
        },
        "cpu": {
            "physical_cpu_ids": cpu_ids,
            "allowed_cpu_ids": sorted(os.sched_getaffinity(0)),
        },
        "correctness": {
            "rows": subset_rows,
            "maximum_avx2_vs_generic_scalar_absolute_error": subset_error,
            "exact_scalar_chunk16_vs_generic": exact_scalar,
            "exact_mmap_vs_copied_compressed_payload": exact_mmap_copy,
            "exact_one_vs_four_thread_output": exact_threading,
        },
        "throughput": {
            "one_thread_samples_seconds": one_samples,
            "four_thread_samples_seconds": many_samples,
            "one_thread_median_seconds": one_median,
            "four_thread_median_seconds": many_median,
            "four_thread_speedup": speedup,
            "four_thread_matrices_per_second": 1.0 / many_median,
        },
        "memory": {
            "before": before,
            "after_mapping_and_tensor_authentication": mapped,
            "after_measurement": after,
            "peak_rss_growth_bytes": rss_growth,
            "expanded_two_bit_weight_image_created": False,
        },
        "gate": gate,
        "claim_boundary": {
            "real_qwen38_27b_container_tensor_integrated": True,
            "whole_model_graph_runnable": False,
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
