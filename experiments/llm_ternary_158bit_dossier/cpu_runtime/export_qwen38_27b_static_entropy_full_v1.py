#!/usr/bin/env python3
"""Restartably export every authenticated mapped Qwen3.8-27B code as CQ3S1."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize import base3
from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_static_chunk_store import StaticChunkStore, sha256_bytes
from clark_quantize.qwen38_static_ternary_codec import (
    frame_metadata as static_metadata,
    native_decoder_available,
    native_encoder_available,
    pack as static_pack,
    unpack as static_unpack,
)
from clark_quantize.qwen38_ternary_entropy import empirical_entropy_bits_per_trit


EXPECTED_MANIFEST_SHA256 = "c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4"
EXPECTED_CHUNKS = 34_344
EXPECTED_TRITS = 26_869_760_000
EXPECTED_TENSORS = 402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(preregistration: dict) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen full static-export source changed: {label}")


def encode_batch(arguments: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    chunks_root_text, records = arguments
    chunks_root = Path(chunks_root_text)
    if not (base3.native_decoder_available() and native_encoder_available() and native_decoder_available()):
        raise RuntimeError("native uniform decode and static encode/decode are required")
    output = []
    for source in records:
        index = int(source["index"])
        source_path = chunks_root / f"{index:08d}.cq3a"
        source_frame = source_path.read_bytes()
        if sha256_bytes(source_frame) != source["frame_sha256"]:
            raise RuntimeError(f"Qwen3.8 source frame {index} digest changed")
        codes = base3.unpack_base3(source_frame)
        raw_sha256 = sha256_bytes(codes.tobytes())
        if codes.size != int(source["trits"]) or raw_sha256 != source["raw_sha256"]:
            raise RuntimeError(f"Qwen3.8 source frame {index} decoded codes changed")
        encode_start = time.perf_counter()
        static_frame = static_pack(codes)
        encode_seconds = time.perf_counter() - encode_start
        decode_start = time.perf_counter()
        decoded = static_unpack(static_frame)
        decode_seconds = time.perf_counter() - decode_start
        if not np.array_equal(decoded, codes):
            raise RuntimeError(f"Qwen3.8 static frame {index} round-trip changed codes")
        metadata = static_metadata(static_frame)
        counts = np.bincount(codes.astype(np.int16) + 1, minlength=3)
        output.append(
            {
                "index": index,
                "tensor": source["descriptor"]["tensor"],
                "trits": int(codes.size),
                "counts": [int(value) for value in counts],
                "raw_sha256": raw_sha256,
                "source_frame_bytes": len(source_frame),
                "source_frame_sha256": source["frame_sha256"],
                "static_frame_sha256": sha256_bytes(static_frame),
                "static_payload_bits": int(metadata["payload_bits"]),
                "frequencies": metadata["frequencies"],
                "encode_seconds": encode_seconds,
                "decode_seconds": decode_seconds,
                "frame": static_frame,
            }
        )
    return output


def load_receipt(path: Path, expected_schema: str) -> dict[str, Any]:
    content = path.read_bytes()
    value = json.loads(content)
    if value.get("schema") != expected_schema or value.get("gate", {}).get("passed") is not True:
        raise RuntimeError(f"required passing parent receipt is invalid: {path}")
    value["_sha256"] = sha256_bytes(content)
    return value


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def package_tensor_section(
    *,
    section_index: int,
    tensor: str,
    indices: list[int],
    store: StaticChunkStore,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    section_path = output_dir / f"{section_index:04d}.cq3s"
    receipt_path = output_dir / f"{section_index:04d}.json"
    cores = [json.loads(store.receipt_path(index).read_text())["core"] for index in indices]
    frame_bytes = sum(int(core["static_frame_bytes"]) for core in cores)
    section_bytes = align_up(frame_bytes, 64)
    if section_path.exists() or receipt_path.exists():
        if not section_path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"incomplete static tensor section: {tensor}")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("tensor") != tensor
            or receipt.get("chunk_indices") != indices
            or receipt.get("section_bytes") != section_bytes
            or section_path.stat().st_size != section_bytes
            or sha256_file(section_path) != receipt.get("sha256")
        ):
            raise RuntimeError(f"static tensor section identity changed: {tensor}")
        return receipt

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{section_path.name}.", dir=output_dir)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as destination:
            for index in indices:
                with store.chunk_path(index).open("rb") as source:
                    while block := source.read(8 * 1024 * 1024):
                        destination.write(block)
                        digest.update(block)
            padding = section_bytes - frame_bytes
            if padding:
                zeros = bytes(padding)
                destination.write(zeros)
                digest.update(zeros)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, section_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    counts = [sum(int(core["counts"][symbol]) for core in cores) for symbol in range(3)]
    receipt = {
        "schema": "qwen38_static_tensor_section_v1",
        "section_index": section_index,
        "tensor": tensor,
        "chunk_indices": indices,
        "chunks": len(indices),
        "trits": sum(int(core["trits"]) for core in cores),
        "counts": counts,
        "frame_bytes": frame_bytes,
        "padding_bytes": section_bytes - frame_bytes,
        "section_bytes": section_bytes,
        "sha256": digest.hexdigest(),
        "path": str(section_path),
    }
    atomic_bytes(receipt_path, canonical_json(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-store",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-base3-full/artifacts/"
            "qwen38-optimal.cq3q/codes.cq3d"
        ),
    )
    parser.add_argument(
        "--sample-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-static-entropy-codec-sample-v1.json"),
    )
    parser.add_argument(
        "--rate-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-ternary-type-class-rate-v1.json"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-static-entropy-full-v1"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-static-entropy-full-v1.json"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-chunks", type=int, default=16)
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_chunks <= 0:
        raise ValueError("positive worker and batch counts are required")
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_static_entropy_full_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_sources(prereg)
    sample = load_receipt(args.sample_receipt, "qwen38_27b_static_entropy_codec_sample_receipt_v1")
    rate = load_receipt(args.rate_receipt, "qwen38_27b_ternary_type_class_rate_receipt_v1")
    if not (base3.native_decoder_available() and native_encoder_available() and native_decoder_available()):
        raise RuntimeError("native static export codec is unavailable")

    source_manifest_path = args.source_store / "manifest.json"
    source_content = source_manifest_path.read_bytes()
    if sha256_bytes(source_content) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Qwen3.8 source manifest changed")
    source_manifest = json.loads(source_content)
    records = source_manifest.get("chunks", [])
    tensors = [record["descriptor"]["tensor"] for record in records]
    if (
        source_manifest.get("complete") is not True
        or len(records) != EXPECTED_CHUNKS
        or sum(int(record["trits"]) for record in records) != EXPECTED_TRITS
        or len(dict.fromkeys(tensors)) != EXPECTED_TENSORS
    ):
        raise RuntimeError("Qwen3.8 mapped-code coverage changed")

    identity = {
        "model": "Qwen/Qwen3.8-27B",
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "preregistration_sha256": sha256_file(prereg_path),
        "sample_receipt_sha256": sample["_sha256"],
        "rate_receipt_sha256": rate["_sha256"],
        "codec": "CQ3S1",
    }
    store = StaticChunkStore(args.artifact / "staging", identity)
    store.validate_prefix(hash_frames=True)
    remaining = records[store.next_index :]
    tasks = [
        (str(args.source_store / "chunks"), remaining[start : start + args.batch_chunks])
        for start in range(0, len(remaining), args.batch_chunks)
    ]
    export_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for batch in executor.map(encode_batch, tasks, chunksize=1):
            for encoded in batch:
                store.commit(encoded)
    store_manifest = store.finalize(expected_chunks=EXPECTED_CHUNKS, expected_trits=EXPECTED_TRITS)
    export_wall_seconds = time.perf_counter() - export_start

    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[record["descriptor"]["tensor"]].append(int(record["index"]))
    sections = []
    section_dir = args.artifact / "tensor-sections"
    for section_index, (tensor, indices) in enumerate(grouped.items()):
        sections.append(
            package_tensor_section(
                section_index=section_index,
                tensor=tensor,
                indices=indices,
                store=store,
                output_dir=section_dir,
            )
        )
    if len(sections) != EXPECTED_TENSORS or sum(item["trits"] for item in sections) != EXPECTED_TRITS:
        raise RuntimeError("static tensor-section coverage changed")

    artifact_manifest = {
        "schema": "qwen38_27b_static_entropy_mapped_artifact_v1",
        "model": "Qwen/Qwen3.8-27B",
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "codec": "CQ3S1",
        "sections": sections,
        "totals": {
            "tensors": len(sections),
            "chunks": EXPECTED_CHUNKS,
            "trits": EXPECTED_TRITS,
            "section_bytes": sum(item["section_bytes"] for item in sections),
        },
    }
    artifact_manifest_path = args.artifact / "artifact-manifest.json"
    atomic_bytes(artifact_manifest_path, canonical_json(artifact_manifest))
    artifact_manifest_bytes = artifact_manifest_path.stat().st_size

    cores = [json.loads(store.receipt_path(index).read_text())["core"] for index in range(EXPECTED_CHUNKS)]
    static_section_bytes = sum(item["section_bytes"] for item in sections)
    uniform_by_tensor: dict[str, int] = defaultdict(int)
    for core in cores:
        uniform_by_tensor[core["tensor"]] += int(core["source_frame_bytes"])
    uniform_section_bytes = sum(align_up(value, 64) for value in uniform_by_tensor.values())
    deployable_static_bytes = static_section_bytes + artifact_manifest_bytes
    deployable_uniform_bytes = uniform_section_bytes + artifact_manifest_bytes
    counts = store_manifest["totals"]["counts"]
    entropy = empirical_entropy_bits_per_trit(counts)
    static_bpt = deployable_static_bytes * 8 / EXPECTED_TRITS
    uniform_bpt = deployable_uniform_bytes * 8 / EXPECTED_TRITS
    encode_cpu_seconds = sum(float(core["encode_seconds"]) for core in cores)
    decode_cpu_seconds = sum(float(core["decode_seconds"]) for core in cores)
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_chunks": len(cores),
        "observed_tensors": len(sections),
        "observed_trits": EXPECTED_TRITS,
        "observed_static_mapped_bits_per_trit": static_bpt,
        "observed_uniform_mapped_bits_per_trit": uniform_bpt,
        "observed_saving_vs_uniform_bits_per_trit": uniform_bpt - static_bpt,
        "observed_static_over_entropy_bits_per_trit": static_bpt - entropy,
        "observed_native_encode_mtrits_per_cpu_second": EXPECTED_TRITS / encode_cpu_seconds / 1e6,
        "observed_native_decode_mtrits_per_cpu_second": EXPECTED_TRITS / decode_cpu_seconds / 1e6,
    }
    gate["passed"] = bool(
        len(cores) == int(gate_spec["required_chunks"])
        and len(sections) == int(gate_spec["required_tensors"])
        and EXPECTED_TRITS == int(gate_spec["required_trits"])
        and static_bpt <= float(gate_spec["maximum_static_mapped_bits_per_trit"])
        and uniform_bpt - static_bpt >= float(gate_spec["minimum_saving_vs_uniform_bits_per_trit"])
        and static_bpt - entropy <= float(gate_spec["maximum_static_over_entropy_bits_per_trit"])
        and EXPECTED_TRITS / encode_cpu_seconds / 1e6
        >= float(gate_spec["minimum_native_encode_mtrits_per_cpu_second"])
        and EXPECTED_TRITS / decode_cpu_seconds / 1e6
        >= float(gate_spec["minimum_native_decode_mtrits_per_cpu_second"])
    )
    receipt = {
        "schema": "qwen38_27b_static_entropy_full_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parents": {
            "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "sample_receipt_sha256": sample["_sha256"],
            "rate_receipt_sha256": rate["_sha256"],
        },
        "artifact": {
            "root": str(args.artifact),
            "manifest": str(artifact_manifest_path),
            "manifest_sha256": sha256_file(artifact_manifest_path),
            "manifest_bytes": artifact_manifest_bytes,
            "tensor_section_bytes": static_section_bytes,
            "deployable_mapped_bytes": deployable_static_bytes,
            "staging_frame_bytes": int(store_manifest["totals"]["static_frame_bytes"]),
            "staging_is_redundant_with_tensor_sections": True,
        },
        "rates": {
            "counts": counts,
            "empirical_entropy_bits_per_trit": entropy,
            "static_mapped_bits_per_trit_including_headers_alignment_and_manifest": static_bpt,
            "uniform_mapped_bits_per_trit_with_same_alignment_and_manifest": uniform_bpt,
            "saving_vs_uniform_bits_per_trit": uniform_bpt - static_bpt,
            "static_over_entropy_bits_per_trit": static_bpt - entropy,
        },
        "performance": {
            "export_wall_seconds_this_invocation": export_wall_seconds,
            "native_encode_cpu_seconds": encode_cpu_seconds,
            "native_decode_cpu_seconds": decode_cpu_seconds,
            "native_encode_mtrits_per_cpu_second": EXPECTED_TRITS / encode_cpu_seconds / 1e6,
            "native_decode_mtrits_per_cpu_second": EXPECTED_TRITS / decode_cpu_seconds / 1e6,
        },
        "gate": gate,
        "claim_boundary": {
            "all_34344_source_frames_authenticated": True,
            "all_26869760000_mapped_trits_roundtrip_exact": True,
            "actual_402_tensor_section_bytes_measured": True,
            "supplemental_264_tensor_codes_exported": False,
            "row_normalized_scales_bundled": False,
            "complete_model_artifact_proven": False,
            "hot_inference_direct_decode_claimed": False,
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
