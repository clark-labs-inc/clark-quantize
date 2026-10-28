#!/usr/bin/env python3
"""Measure the exact CQ3S1 static codec on frozen real Qwen3.8-27B chunks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize import base3
from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_static_ternary_codec import (
    frame_metadata as static_metadata,
    native_decoder_available,
    pack as static_pack,
    unpack as static_unpack,
)
from clark_quantize.qwen38_ternary_entropy import empirical_entropy_bits_per_trit


MANIFEST_SHA256 = "c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timed_decode(operation, repetitions: int) -> tuple[list[float], np.ndarray]:
    samples = []
    output = None
    for _ in range(repetitions):
        start = time.perf_counter()
        output = operation()
        samples.append(time.perf_counter() - start)
    if output is None:
        raise AssertionError("decode benchmark did not execute")
    return samples, output


def validate_sources(preregistration: dict) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen static-codec source changed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-base3-full/artifacts/"
            "qwen38-optimal.cq3q/codes.cq3d"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-static-entropy-codec-sample-v1"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-static-entropy-codec-sample-v1.json"),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_static_entropy_codec_sample_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_sources(prereg)
    if not native_decoder_available():
        raise RuntimeError("CQ3S1 native decoder is required for the authoritative screen")

    manifest_path = args.store / "manifest.json"
    manifest_content = manifest_path.read_bytes()
    if sha256_bytes(manifest_content) != MANIFEST_SHA256:
        raise RuntimeError("Qwen3.8-27B base-3 manifest identity changed")
    manifest = json.loads(manifest_content)
    if manifest.get("schema") != "clark_chunked_base3_v1" or manifest.get("complete") is not True:
        raise RuntimeError("Qwen3.8-27B base-3 source store is incomplete")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total_trits = total_static_bytes = total_uniform_bytes = 0
    total_entropy_bits = 0.0
    static_decode_seconds: list[float] = []
    uniform_decode_seconds: list[float] = []
    decode_repetitions = int(prereg["measurement"]["decode_repetitions"])
    for frozen in prereg["scope"]["chunks"]:
        index = int(frozen["index"])
        source_record = manifest["chunks"][index]
        for key in ("trits", "raw_sha256", "frame_sha256"):
            if source_record.get(key) != frozen[key]:
                raise RuntimeError(f"frozen Qwen3.8 chunk {index} {key} changed")
        if source_record.get("descriptor", {}).get("tensor") != frozen["tensor"]:
            raise RuntimeError(f"frozen Qwen3.8 chunk {index} tensor changed")
        source_path = args.store / "chunks" / f"{index:08d}.cq3a"
        uniform_frame = source_path.read_bytes()
        if sha256_bytes(uniform_frame) != frozen["frame_sha256"]:
            raise RuntimeError(f"frozen Qwen3.8 chunk {index} frame bytes changed")
        codes = base3.unpack_base3(uniform_frame)
        if sha256_bytes(codes.tobytes()) != frozen["raw_sha256"]:
            raise RuntimeError(f"frozen Qwen3.8 chunk {index} decoded trits changed")
        counts = np.bincount(codes.astype(np.int16) + 1, minlength=3)
        entropy = empirical_entropy_bits_per_trit(counts)

        encode_start = time.perf_counter()
        static_frame = static_pack(codes)
        encode_seconds = time.perf_counter() - encode_start
        output_path = args.artifact_dir / f"{index:08d}.cq3s"
        atomic_bytes(output_path, static_frame)
        metadata = static_metadata(static_frame)

        static_samples, static_codes = timed_decode(
            lambda: static_unpack(static_frame), decode_repetitions
        )
        uniform_samples, uniform_codes = timed_decode(
            lambda: base3.unpack_base3(uniform_frame), decode_repetitions
        )
        exact = bool(np.array_equal(static_codes, codes) and np.array_equal(uniform_codes, codes))
        if not exact:
            raise RuntimeError(f"CQ3S1 exact round-trip failed for chunk {index}")

        trits = int(codes.size)
        static_rate = len(static_frame) * 8 / trits
        uniform_rate = len(uniform_frame) * 8 / trits
        record = {
            "index": index,
            "tensor": frozen["tensor"],
            "trits": trits,
            "counts": [int(value) for value in counts],
            "empirical_entropy_bits_per_trit": entropy,
            "uniform_frame": {
                "path": str(source_path),
                "bytes": len(uniform_frame),
                "sha256": sha256_bytes(uniform_frame),
                "bits_per_trit": uniform_rate,
            },
            "static_frame": {
                "path": str(output_path),
                "bytes": len(static_frame),
                "sha256": sha256_bytes(static_frame),
                "bits_per_trit": static_rate,
                "payload_bits": int(metadata["payload_bits"]),
                "header_bytes": int(metadata["header_bytes"]),
                "frequencies": metadata["frequencies"],
            },
            "static_over_entropy_bits_per_trit": static_rate - entropy,
            "saving_vs_uniform_bits_per_trit": uniform_rate - static_rate,
            "encode_seconds": encode_seconds,
            "static_native_decode_seconds": static_samples,
            "uniform_native_decode_seconds": uniform_samples,
            "static_native_decode_mtrits_per_second_median": trits
            / statistics.median(static_samples)
            / 1e6,
            "uniform_native_decode_mtrits_per_second_median": trits
            / statistics.median(uniform_samples)
            / 1e6,
            "static_over_uniform_native_decode_throughput": statistics.median(uniform_samples)
            / statistics.median(static_samples),
            "exact_roundtrip": exact,
        }
        records.append(record)
        total_trits += trits
        total_static_bytes += len(static_frame)
        total_uniform_bytes += len(uniform_frame)
        total_entropy_bits += entropy * trits
        static_decode_seconds.extend(static_samples)
        uniform_decode_seconds.extend(uniform_samples)

    aggregate_entropy = total_entropy_bits / total_trits
    static_rate = total_static_bytes * 8 / total_trits
    uniform_rate = total_uniform_bytes * 8 / total_trits
    static_mtrits = total_trits * decode_repetitions / sum(static_decode_seconds) / 1e6
    uniform_mtrits = total_trits * decode_repetitions / sum(uniform_decode_seconds) / 1e6
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_chunks": len(records),
        "observed_trits": total_trits,
        "observed_static_bits_per_trit_including_headers": static_rate,
        "observed_uniform_bits_per_trit_including_headers": uniform_rate,
        "observed_saving_vs_uniform_bits_per_trit": uniform_rate - static_rate,
        "observed_static_over_entropy_bits_per_trit": static_rate - aggregate_entropy,
        "observed_static_native_decode_mtrits_per_second": static_mtrits,
        "observed_uniform_native_decode_mtrits_per_second": uniform_mtrits,
        "observed_static_over_uniform_native_decode_throughput": static_mtrits / uniform_mtrits,
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_chunks"])
        and total_trits == int(gate_spec["required_trits"])
        and all(record["exact_roundtrip"] for record in records)
        and uniform_rate - static_rate
        >= float(gate_spec["minimum_saving_vs_uniform_bits_per_trit"])
        and static_rate - aggregate_entropy
        <= float(gate_spec["maximum_static_over_entropy_bits_per_trit"])
        and static_mtrits
        >= float(gate_spec["minimum_static_native_decode_mtrits_per_second"])
        and static_mtrits / uniform_mtrits
        >= float(gate_spec["minimum_static_over_uniform_native_decode_throughput"])
    )
    receipt = {
        "schema": "qwen38_27b_static_entropy_codec_sample_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "source_manifest": {"path": str(manifest_path), "sha256": MANIFEST_SHA256},
        "records": records,
        "aggregate": {
            "trits": total_trits,
            "empirical_entropy_bits_per_trit": aggregate_entropy,
            "uniform_frame_bytes": total_uniform_bytes,
            "uniform_bits_per_trit_including_headers": uniform_rate,
            "static_frame_bytes": total_static_bytes,
            "static_bits_per_trit_including_headers": static_rate,
            "static_over_entropy_bits_per_trit": static_rate - aggregate_entropy,
            "saving_vs_uniform_bits_per_trit": uniform_rate - static_rate,
            "static_native_decode_mtrits_per_second": static_mtrits,
            "uniform_native_decode_mtrits_per_second": uniform_mtrits,
            "static_over_uniform_native_decode_throughput": static_mtrits / uniform_mtrits,
        },
        "gate": gate,
        "claim_boundary": {
            "real_qwen38_27b_chunks_authenticated": True,
            "sampled_chunks_only": True,
            "exact_static_artifacts_exported": True,
            "native_decoder_measured": True,
            "full_model_static_artifact_exported": False,
            "full_model_complete_rate_measured": False,
            "hot_path_direct_decode_claimed": False,
            "capability_changed": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
