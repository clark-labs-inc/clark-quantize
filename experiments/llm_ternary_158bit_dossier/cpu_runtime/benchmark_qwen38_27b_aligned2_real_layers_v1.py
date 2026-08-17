#!/usr/bin/env python3
"""Run real-weight Qwen3.8-27B DeltaNet and attention decoder layers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_aligned2_cache import Qwen38Aligned2RuntimeImage
from clark_quantize.qwen38_cq3_runtime import (
    Qwen38AttentionState,
    Qwen38CQ3DeltaNet,
)
from clark_quantize.ternary_cpu import cpu_features, native_available


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


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
            raise RuntimeError(f"frozen real-layer input changed: {label}")


def new_state(kind: str):
    if kind == "linear_attention":
        return (
            np.zeros((3, Qwen38CQ3DeltaNet.CONV_DIM), dtype=np.float32),
            np.zeros(
                (
                    Qwen38CQ3DeltaNet.VALUE_HEADS,
                    Qwen38CQ3DeltaNet.HEAD_DIM,
                    Qwen38CQ3DeltaNet.HEAD_DIM,
                ),
                dtype=np.float32,
            ),
        )
    if kind == "full_attention":
        return Qwen38AttentionState.allocate(1)
    raise ValueError("Qwen3.8 real-layer mixer kind changed")


def state_arrays(state) -> tuple[np.ndarray, ...]:
    if isinstance(state, tuple):
        return state
    if isinstance(state, Qwen38AttentionState):
        return state.key, state.value
    raise TypeError("Qwen3.8 real-layer state type changed")


def state_length(state) -> int | None:
    return state.length if isinstance(state, Qwen38AttentionState) else None


def state_bytes(state) -> int:
    return sum(int(value.nbytes) for value in state_arrays(state))


def comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict:
    delta = np.asarray(candidate, dtype=np.float32) - np.asarray(
        reference, dtype=np.float32
    )
    reference_norm = float(np.linalg.norm(reference.astype(np.float64)))
    delta_norm = float(np.linalg.norm(delta.astype(np.float64)))
    return {
        "allclose": bool(np.allclose(reference, candidate, rtol=rtol, atol=atol)),
        "maximum_absolute_error": float(np.max(np.abs(delta), initial=0.0)),
        "relative_l2_error": delta_norm / max(reference_norm, 1e-30),
        "reference_finite": bool(np.all(np.isfinite(reference))),
        "candidate_finite": bool(np.all(np.isfinite(candidate))),
    }


def compare_states(reference, candidate, *, rtol: float, atol: float) -> dict:
    pairs = zip(state_arrays(reference), state_arrays(candidate))
    records = [
        comparison(left, right, rtol=rtol, atol=atol)
        for left, right in pairs
    ]
    return {
        "allclose": bool(
            records
            and all(record["allclose"] for record in records)
            and state_length(reference) == state_length(candidate)
        ),
        "array_comparisons": records,
        "reference_length": state_length(reference),
        "candidate_length": state_length(candidate),
        "maximum_absolute_error": max(
            record["maximum_absolute_error"] for record in records
        ),
        "maximum_relative_l2_error": max(
            record["relative_l2_error"] for record in records
        ),
        "all_finite": all(
            record["reference_finite"] and record["candidate_finite"]
            for record in records
        ),
    }


def run_layer(
    layer,
    hidden: np.ndarray,
    *,
    implementation: str,
    threads: int,
    executor: ThreadPoolExecutor | None,
    cpu_ids: list[int],
):
    state = new_state(layer.mixer_kind)
    started = time.perf_counter_ns()
    output = layer.forward(
        hidden,
        state,
        0,
        gemv_implementation=implementation,
        elementwise_implementation=implementation,
        threads=threads,
        executor=executor,
        cpu_ids=cpu_ids,
        validated_state=True,
    )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("non-finite real-layer timing")
    return elapsed, output, state


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
        "--cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-aligned2-cache-v1/"
            "Qwen3.8-27B-row-normalized.a2"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-aligned2-cache-v1/"
            "Qwen3.8-27B-row-normalized.a2.manifest.json"
        ),
    )
    parser.add_argument(
        "--parent-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-full-cache-v1.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-real-layers-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = (
        runtime / "qwen38_27b_aligned2_real_layers_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    if prereg["scope"]["active_model_only"] != "Qwen/Qwen3.8-27B":
        raise ValueError("real-layer gate is not frozen to Qwen3.8-27B")
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

    parent_content = args.parent_receipt.read_bytes()
    parent = json.loads(parent_content)
    if (
        parent.get("schema") != "qwen38_27b_aligned2_full_cache_receipt_v1"
        or parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("complete aligned2 parent gate did not pass")
    source = parent["source_container"]
    cache = parent["cache"]
    if (
        args.container.resolve() != Path(source["path"]).resolve()
        or args.container.stat().st_size != int(source["bytes"])
        or args.cache.resolve() != Path(cache["path"]).resolve()
        or args.cache.stat().st_size != int(cache["bytes"])
        or args.manifest.resolve() != Path(cache["manifest_path"]).resolve()
        or sha256_file(args.manifest) != cache["manifest_sha256"]
    ):
        raise RuntimeError("real-layer artifact identities differ from their parent")

    measurement = prereg["measurement"]
    rtol = float(prereg["gate"]["maximum_scalar_avx2_relative_tolerance"])
    atol = float(prereg["gate"]["maximum_scalar_avx2_absolute_tolerance"])
    token_id = int(prereg["input"]["token_id"])
    layer_records: dict[str, dict] = {}
    with Qwen38Aligned2RuntimeImage(
        args.container,
        args.cache,
        args.manifest,
        verify_source_container=False,
        verify_full_cache=False,
    ) as image:
        embedding = image.aligned_tensor(
            "model.language_model.embed_tokens.weight"
        )
        hidden = embedding.decode_row_f32(token_id)
        if hidden.shape != (5120,) or not np.all(np.isfinite(hidden)):
            raise RuntimeError("frozen Qwen3.8 embedding row is invalid")
        layers = {
            str(layer_index): image.decoder_layer(layer_index)
            for layer_index in prereg["scope"]["layers"]
        }
        expected_kinds = prereg["scope"]["expected_layer_types"]
        if {
            index: layer.mixer_kind for index, layer in layers.items()
        } != expected_kinds:
            raise RuntimeError("Qwen3.8 real-layer types changed")

        with ThreadPoolExecutor(max_workers=threads) as executor:
            for index, layer in layers.items():
                for _ in range(int(measurement["warmups_per_implementation"])):
                    run_layer(
                        layer,
                        hidden,
                        implementation="scalar",
                        threads=1,
                        executor=None,
                        cpu_ids=cpu_ids[:1],
                    )
                    run_layer(
                        layer,
                        hidden,
                        implementation="avx2",
                        threads=threads,
                        executor=executor,
                        cpu_ids=cpu_ids,
                    )
                layer_faults_before = resource.getrusage(resource.RUSAGE_SELF)
                samples = {"scalar": [], "avx2_four": []}
                outputs = {}
                states = {}
                repetitions = int(measurement["paired_repetitions"])
                for repetition in range(repetitions):
                    order = ("scalar", "avx2_four")
                    if repetition & 1:
                        order = tuple(reversed(order))
                    for name in order:
                        is_scalar = name == "scalar"
                        elapsed, output, state = run_layer(
                            layer,
                            hidden,
                            implementation="scalar" if is_scalar else "avx2",
                            threads=1 if is_scalar else threads,
                            executor=None if is_scalar else executor,
                            cpu_ids=cpu_ids[:1] if is_scalar else cpu_ids,
                        )
                        samples[name].append(elapsed)
                        outputs[name] = output
                        states[name] = state
                layer_faults_after = resource.getrusage(resource.RUSAGE_SELF)
                output_parity = comparison(
                    outputs["scalar"],
                    outputs["avx2_four"],
                    rtol=rtol,
                    atol=atol,
                )
                state_parity = compare_states(
                    states["scalar"],
                    states["avx2_four"],
                    rtol=rtol,
                    atol=atol,
                )
                scalar_median = statistics.median(samples["scalar"])
                avx2_median = statistics.median(samples["avx2_four"])
                layer_records[index] = {
                    "layer": int(index),
                    "mixer_kind": layer.mixer_kind,
                    "state_bytes": state_bytes(states["avx2_four"]),
                    "scalar_samples_seconds": samples["scalar"],
                    "avx2_four_samples_seconds": samples["avx2_four"],
                    "scalar_median_seconds": scalar_median,
                    "avx2_four_median_seconds": avx2_median,
                    "avx2_four_speedup_over_scalar": (
                        scalar_median / avx2_median
                    ),
                    "minor_page_faults_during_paired_region": int(
                        layer_faults_after.ru_minflt
                        - layer_faults_before.ru_minflt
                    ),
                    "major_page_faults_during_paired_region": int(
                        layer_faults_after.ru_majflt
                        - layer_faults_before.ru_majflt
                    ),
                    "output_parity": output_parity,
                    "state_parity": state_parity,
                }
        verified_tensors = sorted(image._verified_tensors)

    major_faults = sum(
        record["major_page_faults_during_paired_region"]
        for record in layer_records.values()
    )
    minor_faults = sum(
        record["minor_page_faults_during_paired_region"]
        for record in layer_records.values()
    )
    peak_rss = peak_rss_bytes()
    gate_spec = prereg["gate"]
    layer_passes = {
        index: bool(
            record["output_parity"]["allclose"]
            and record["state_parity"]["allclose"]
            and record["output_parity"]["reference_finite"]
            and record["output_parity"]["candidate_finite"]
            and record["state_parity"]["all_finite"]
            and all(
                math.isfinite(value) and value > 0.0
                for name in ("scalar_samples_seconds", "avx2_four_samples_seconds")
                for value in record[name]
            )
            and record["avx2_four_median_seconds"]
            <= float(gate_spec["maximum_avx2_seconds_per_layer"])
        )
        for index, record in layer_records.items()
    }
    gate = {
        **gate_spec,
        "observed_layer_passes": layer_passes,
        "observed_authenticated_aligned_tensor_count": len(verified_tensors),
        "observed_process_peak_rss_bytes": peak_rss,
        "observed_major_page_faults": major_faults,
    }
    gate["passed"] = bool(
        len(layer_passes) == len(prereg["scope"]["layers"])
        and all(layer_passes.values())
        and len(verified_tensors)
        >= int(gate_spec["minimum_authenticated_aligned_tensor_count"])
        and peak_rss <= int(gate_spec["maximum_process_peak_rss_bytes"])
        and major_faults == 0
    )
    receipt = {
        "schema": "qwen38_27b_aligned2_real_layers_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parent": {
            "path": str(args.parent_receipt),
            "sha256": hashlib.sha256(parent_content).hexdigest(),
        },
        "artifacts": {
            "source_container": source,
            "cache": cache,
            "full_source_and_cache_sha256_reused_from_parent": True,
            "every_executed_aligned_tensor_sha256_reverified": True,
        },
        "native_library": {
            "path": native_library,
            "sha256": sha256_file(Path(native_library)),
        },
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "physical_cpu_ids": cpu_ids,
        "cpu_features": features,
        "input": {
            **prereg["input"],
            "embedding_finite": True,
            "embedding_l2_norm": float(
                np.linalg.norm(hidden.astype(np.float64))
            ),
        },
        "layers": layer_records,
        "authenticated_aligned_tensors": verified_tensors,
        "measurement": {
            "minor_page_faults": minor_faults,
            "major_page_faults": major_faults,
            "process_peak_rss_bytes": peak_rss,
        },
        "gate": gate,
        "claim_boundary": {
            "actual_qwen38_27b_weights_used": True,
            "full_deltanet_decoder_layer_zero_state_proven_if_passed": (
                gate["passed"]
            ),
            "full_attention_decoder_layer_one_token_proven_if_passed": (
                gate["passed"]
            ),
            "full_64_layer_text_graph_proven": False,
            "teacher_or_reference_logit_parity_proven": False,
            "multi_token_state_correctness_proven": False,
            "model_capability_proven": False,
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
