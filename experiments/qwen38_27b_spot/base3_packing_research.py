#!/usr/bin/env python3
"""Run the first exact ideal-base-3 packing gate on real Qwen3.8 weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clark_quantize.base3 import frame_metadata, pack_base3, unpack_base3
from clark_quantize.ternary import ternarize


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def load_codes(
    model_dir: Path,
    preflight: dict,
    sample_rows: int,
    max_trits: int,
    device: torch.device,
):
    codes = []
    sources = []
    remaining = max_trits
    for probe in preflight["probes"]:
        if remaining <= 0:
            break
        name = probe["name"]
        index = preflight["download_receipt"]
        del index  # The shard map below is authoritative for tensor placement.
        weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(model_dir / weight_map[name], framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            source = view[: min(sample_rows, view.get_shape()[0]), :].contiguous()
        weight = source.to(device, dtype=torch.float32)
        dequantized, stats = ternarize(weight, group=128, method="optimal")
        tensor_codes = torch.sign(dequantized).to(torch.int8).cpu().numpy().reshape(-1)
        take = min(remaining, tensor_codes.size)
        codes.append(tensor_codes[:take])
        sources.append({"name": name, "available_trits": int(tensor_codes.size), "used_trits": take, "stats": stats})
        remaining -= take
        del weight, dequantized
    if remaining:
        raise RuntimeError(f"representative tensors supplied only {max_trits - remaining} of {max_trits} trits")
    return np.concatenate(codes), sources


def pack_aligned2(codes: np.ndarray) -> bytes:
    digits = (codes.astype(np.uint8) + 1).reshape(-1)
    padding = (-digits.size) % 4
    if padding:
        digits = np.pad(digits, (0, padding))
    groups = digits.reshape(-1, 4)
    return bytes(groups[:, 0] | (groups[:, 1] << 2) | (groups[:, 2] << 4) | (groups[:, 3] << 6))


def pack_radix5(codes: np.ndarray) -> bytes:
    digits = (codes.astype(np.uint16) + 1).reshape(-1)
    padding = (-digits.size) % 5
    if padding:
        digits = np.pad(digits, (0, padding))
    groups = digits.reshape(-1, 5)
    powers = np.asarray([1, 3, 9, 27, 81], dtype=np.uint16)
    return bytes((groups * powers).sum(axis=1).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--preflight", type=Path, default=Path("/workspace/quant-state/qwen38-preflight.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/quant-runs/qwen38-base3-reference/artifacts"))
    parser.add_argument("--receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-reference.json"))
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--max-trits", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    hypothesis = (
        "A streaming uniform arithmetic codec can losslessly pack real optimal-MSE Qwen ternary codes "
        "within 0.001 payload bits/trit of log2(3)."
    )
    preflight = json.loads(args.preflight.read_text())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    codes, sources = load_codes(
        args.model_dir,
        preflight,
        args.sample_rows,
        args.max_trits,
        device,
    )
    histogram = Counter(map(int, codes))
    raw_sha = hashlib.sha256(codes.tobytes()).hexdigest()

    started = time.perf_counter()
    frame = pack_base3(codes)
    encode_seconds = time.perf_counter() - started
    metadata = frame_metadata(frame)
    started = time.perf_counter()
    decoded = unpack_base3(frame)
    decode_seconds = time.perf_counter() - started
    exact = bool(np.array_equal(decoded, codes))
    if not exact:
        raise AssertionError("base-3 round-trip changed Qwen ternary codes")

    aligned = pack_aligned2(codes)
    radix5 = pack_radix5(codes)
    artifact = args.out_dir / "qwen38-optimal-sample.cq3a"
    atomic_bytes(artifact, frame)
    artifact_sha = hashlib.sha256(frame).hexdigest()

    ideal = math.log2(3)
    payload_bpt = metadata["payload_bits"] / codes.size
    ternary_params = preflight["parameter_ledger"]["ternary_payload"]
    throughput = codes.size / encode_seconds
    receipt = {
        "schema": "clark_qwen38_base3_reference_v1",
        "hypothesis": hypothesis,
        "model_revision": preflight["download_receipt"]["revision"],
        "quantizer": {"method": "optimal", "group_size": 128},
        "sample": {
            "trits": int(codes.size),
            "histogram": {str(key): value for key, value in sorted(histogram.items())},
            "raw_codes_sha256": raw_sha,
            "sources": sources,
        },
        "codecs": {
            "ideal_log2_3": {"bits_per_trit": ideal},
            "arithmetic_base3": {
                **metadata,
                "payload_bits_per_trit": payload_bpt,
                "delta_from_log2_3": payload_bpt - ideal,
                "frame_bits_per_trit": len(frame) * 8 / codes.size,
                "encode_seconds": encode_seconds,
                "decode_seconds": decode_seconds,
                "encode_mtrits_per_second": throughput / 1e6,
                "decode_mtrits_per_second": codes.size / decode_seconds / 1e6,
                "artifact": str(artifact),
                "artifact_sha256": artifact_sha,
                "exact_roundtrip": exact,
            },
            "radix5": {"payload_bytes": len(radix5), "payload_bits_per_trit": len(radix5) * 8 / codes.size},
            "aligned2": {"payload_bytes": len(aligned), "payload_bits_per_trit": len(aligned) * 8 / codes.size},
        },
        "full_checkpoint_projection": {
            "ternary_parameters": ternary_params,
            "ideal_ternary_payload_bytes": math.ceil(ternary_params * ideal / 8),
            "reference_encode_hours_at_measured_rate": ternary_params / throughput / 3600,
            "complete_model_base3_bytes_including_scales_and_islands": preflight["parameter_ledger"]["base3_payload_bytes"],
            "complete_model_effective_bits_per_parameter": preflight["parameter_ledger"]["effective_bits_per_total_parameter_base3"],
        },
        "resources": {
            "device": device_name,
            "peak_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        },
        "promotion": {
            "exact_roundtrip_required": True,
            "max_delta_bits_per_trit": 0.001,
            "passed": exact and payload_bpt <= ideal + 0.001,
            "scope": "reference codec only; native parallel throughput is a separate gate",
        },
        "protected_metrics": ["exact code equality", "CRC corruption detection", "full effective-bit accounting"],
        "stop_condition": "One million real codes or first correctness/rate failure; no full-checkpoint export.",
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
