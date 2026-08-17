#!/usr/bin/env python3
"""Benchmark the real Qwen3.8-27B layer-0 MLP from direct CQ3 storage."""
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
from clark_quantize.qwen38_cq3_runtime import (
    Qwen38CQ3MLP,
    Qwen38CQ3RuntimeImage,
)
from clark_quantize.ternary_cpu import cpu_features, native_available


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
            raise RuntimeError(f"frozen layer-0 MLP input changed: {label}")


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
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-cq3-layer0-mlp-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_cq3_layer0_mlp_preregistration_v1.json"
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
        raise RuntimeError("layer-0 MLP parents are not passed frozen receipts")
    artifact = export_parent["artifact"]
    if (
        args.container.resolve() != Path(artifact["path"]).resolve()
        or args.container.stat().st_size != int(artifact["bytes"])
        or mmap_parent["container"]["sha256"] != artifact["sha256"]
    ):
        raise RuntimeError("layer-0 MLP container identity differs from its parents")

    before = memory_status()
    with Qwen38CQ3RuntimeImage(args.container) as image:
        config = json.loads(image.auxiliary_file("config.json").tobytes())
        text_config = config.get("text_config", {})
        expected_config = prereg["model_config"]
        observed_config = {
            key: text_config.get(key)
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "full_attention_interval",
            )
        }
        if observed_config != expected_config:
            raise RuntimeError("authenticated Qwen3.8-27B config changed")
        mlp = Qwen38CQ3MLP.from_image(image, 0)
        expected_shapes = prereg["scope"]["projection_shapes"]
        observed_shapes = {
            "gate": list(mlp.gate.shape),
            "up": list(mlp.up.shape),
            "down": list(mlp.down.shape),
        }
        if observed_shapes != expected_shapes:
            raise RuntimeError("layer-0 MLP projection shapes changed")
        projections = (mlp.gate, mlp.up, mlp.down)
        payload_bytes = {
            "block203": sum(view.packed_weights.nbytes for view in projections),
            "scale_indices": sum(
                view.packed_scale_indices.nbytes for view in projections
            ),
            "row_anchors": sum(view.row_anchors.nbytes for view in projections),
            "ratio_codebooks": sum(
                view.ratio_codebook.nbytes for view in projections
            ),
        }
        if (
            payload_bytes["block203"]
            != int(prereg["scope"]["block203_payload_bytes"])
            or sum(
                payload_bytes[name]
                for name in ("scale_indices", "row_anchors", "ratio_codebooks")
            )
            != int(prereg["scope"]["row_normalized_scale_payload_bytes"])
        ):
            raise RuntimeError("layer-0 MLP compressed payload ledger changed")
        mapping = image._mapping
        arrays = (
            mlp.gate.packed_weights,
            mlp.gate.packed_scale_indices,
            mlp.gate.row_anchors,
            mlp.gate.ratio_codebook,
            mlp.up.packed_weights,
            mlp.up.packed_scale_indices,
            mlp.up.row_anchors,
            mlp.up.ratio_codebook,
            mlp.down.packed_weights,
            mlp.down.packed_scale_indices,
            mlp.down.row_anchors,
            mlp.down.ratio_codebook,
        )
        zero_copy = mapping is not None and all(
            np.shares_memory(array, mapping) for array in arrays
        )
        mapped = memory_status()
        generator = np.random.default_rng(int(prereg["candidate"]["seed"]))
        activation = generator.normal(size=mlp.gate.columns).astype(np.float32)
        implementation = prereg["candidate"]["implementation"]

        one = lambda: mlp.forward(
            activation,
            implementation=implementation,
            threads=1,
            cpu_ids=cpu_ids[:1],
        )
        with ThreadPoolExecutor(max_workers=threads) as executor:
            many = lambda: mlp.forward(
                activation,
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
                    started = time.perf_counter_ns()
                    output = function()
                    elapsed = (time.perf_counter_ns() - started) / 1e9
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise RuntimeError("non-finite layer-0 MLP timing")
                    samples.append(elapsed)
                    if function is one:
                        one_output = output
                    else:
                        many_output = output
        assert one_output is not None and many_output is not None
        after = memory_status()

    one_median = statistics.median(one_samples)
    many_median = statistics.median(many_samples)
    speedup = one_median / many_median
    layers = int(prereg["model_config"]["num_hidden_layers"])
    optimistic_mlp_only_tokens_per_second = 1.0 / (layers * many_median)
    exact_threading = bool(np.array_equal(one_output, many_output))
    finite = bool(
        np.all(np.isfinite(one_output))
        and np.all(np.isfinite(many_output))
        and all(math.isfinite(value) and value > 0 for value in one_samples)
        and all(math.isfinite(value) and value > 0 for value in many_samples)
    )
    rss_growth = max(0, after["VmHWM"] - before["VmHWM"])
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_zero_copy_all_projection_arrays": zero_copy,
        "observed_exact_one_vs_four_thread_output": exact_threading,
        "observed_four_thread_speedup": speedup,
        "observed_optimistic_mlp_only_tokens_per_second": (
            optimistic_mlp_only_tokens_per_second
        ),
        "observed_peak_rss_growth_bytes": rss_growth,
        "observed_all_outputs_and_samples_finite": finite,
    }
    gate["passed"] = (
        zero_copy
        and exact_threading
        and speedup >= gate_spec["minimum_four_thread_speedup"]
        and optimistic_mlp_only_tokens_per_second
        >= gate_spec["minimum_optimistic_mlp_only_tokens_per_second"]
        and rss_growth <= gate_spec["maximum_peak_rss_growth_bytes"]
        and finite
    )
    receipt = {
        "schema": "qwen38_27b_cq3_layer0_mlp_receipt_v1",
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
        },
        "container": {
            "path": str(args.container),
            "bytes": args.container.stat().st_size,
            "sha256": artifact["sha256"],
        },
        "model_config": observed_config,
        "projection_shapes": observed_shapes,
        "compressed_projection_payload_bytes": payload_bytes,
        "activation": {
            "seed": int(prereg["candidate"]["seed"]),
            "sha256": hashlib.sha256(activation.tobytes()).hexdigest(),
        },
        "output": {
            "sha256": hashlib.sha256(many_output.tobytes()).hexdigest(),
            "minimum": float(np.min(many_output)),
            "maximum": float(np.max(many_output)),
            "l2_norm": float(np.linalg.norm(many_output)),
        },
        "cpu": {
            "physical_cpu_ids": cpu_ids,
            "allowed_cpu_ids": sorted(os.sched_getaffinity(0)),
        },
        "throughput": {
            "one_thread_samples_seconds": one_samples,
            "four_thread_samples_seconds": many_samples,
            "one_thread_median_seconds": one_median,
            "four_thread_median_seconds": many_median,
            "four_thread_speedup": speedup,
            "optimistic_mlp_only_tokens_per_second": (
                optimistic_mlp_only_tokens_per_second
            ),
            "optimistic_upper_bound_excludes": [
                "attention and recurrent linear-attention kernels",
                "normalization, residuals, embeddings, sampling, and KV state",
                "all runtime orchestration outside the MLP operator",
            ],
        },
        "memory": {
            "before": before,
            "after_mapping_and_authentication": mapped,
            "after_measurement": after,
            "peak_rss_growth_bytes": rss_growth,
            "expanded_weight_image_created": False,
        },
        "gate": gate,
        "claim_boundary": {
            "real_qwen38_27b_layer0_mlp_runnable": True,
            "whole_model_graph_runnable": False,
            "capability_or_perplexity_proven": False,
            "end_to_end_tokens_per_second_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.receipt, json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode() + b"\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
