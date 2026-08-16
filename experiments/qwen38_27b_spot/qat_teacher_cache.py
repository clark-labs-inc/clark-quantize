#!/usr/bin/env python3
"""Cache fixed local BF16 teacher targets for the one-layer QAT rung."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from behavioral_quality_gate import atomic_json, atomic_torch_save


CALIBRATION_TEXTS = (
    "Summarize the difference between lossless storage encoding and lossy model quantization in two sentences.",
    "Compute 48 times 17, then verify the result using a second arithmetic decomposition.",
    "Write a Python function that returns the first repeated item in a list, or None when every item is unique.",
    "Explain why a validation set must not be reused as the optimization set during model compression.",
    "En español, describe brevemente por qué una medición reproducible necesita condiciones explícitas.",
    "Given a tool named weather with a city argument, describe the JSON arguments for requesting Seattle weather.",
    "A process completes 3/8 of a task in 15 minutes at a constant rate. Determine the total completion time.",
    "State one reason a low training loss can coexist with poor generated text and one test that would reveal it.",
)


def validate_preregistration(prereg: dict, *, top_k: int, max_length: int) -> dict:
    first = prereg.get("first_rung", {})
    if (
        prereg.get("schema") != "clark_qwen38_one_layer_qat_preregistration_v1"
        or tuple(first.get("calibration_texts", ())) != CALIBRATION_TEXTS
        or first.get("calibration_sequences") != len(CALIBRATION_TEXTS)
        or first.get("teacher_top_k") != top_k
        or first.get("maximum_sequence_tokens") != max_length
    ):
        raise ValueError("teacher-cache arguments or texts differ from preregistration")
    return first


def validate_cache_rows(rows: list[dict], *, expected_rows: int, top_k: int) -> bool:
    if len(rows) != expected_rows:
        return False
    for row in rows:
        values = row.get("top_values")
        indices = row.get("top_indices")
        lse = row.get("teacher_logsumexp")
        ids = row.get("input_ids")
        if not all(torch.is_tensor(value) for value in (values, indices, lse, ids)):
            return False
        if values.ndim != 2 or indices.shape != values.shape or values.shape[1] != top_k:
            return False
        if lse.shape != values.shape[:1] or ids.ndim != 1 or ids.numel() != values.shape[0]:
            return False
        if not torch.isfinite(values.float()).all() or not torch.isfinite(lse.float()).all():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--sensitivity", type=Path, default=Path("/workspace/quant-state/qwen38-layer-sensitivity.json"))
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-runs/qwen38-qat/teacher-cache.pt"))
    parser.add_argument("--receipt", type=Path, default=Path("/workspace/quant-state/qwen38-qat-teacher-cache.json"))
    parser.add_argument("--preregistration", type=Path, default=HERE / "qat_preregistration.json")
    args = parser.parse_args()
    preregistration_content = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_content)
    validate_preregistration(preregistration, top_k=args.top_k, max_length=args.max_length)
    sensitivity = json.loads(args.sensitivity.read_text())
    if not sensitivity.get("promotion", {}).get("passed"):
        raise ValueError("layer sensitivity gate did not pass")
    if float(sensitivity.get("selected_teacher_kl_recovery", 0.0)) <= 0:
        raise ValueError("selected layer has no positive teacher-KL recovery")
    if sensitivity.get("model_revision") != preregistration.get("model", {}).get("revision"):
        raise ValueError("teacher-cache sensitivity revision differs from preregistration")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval().requires_grad_(False)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for index, text in enumerate(CALIBRATION_TEXTS):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_length,
        ).to("cuda")
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0]
        top_values, top_indices = torch.topk(logits.float(), k=args.top_k, dim=-1)
        rows.append({
            "index": index,
            "text": text,
            "input_ids": inputs["input_ids"][0].cpu().to(torch.long),
            "top_values": top_values.cpu().to(torch.bfloat16),
            "top_indices": top_indices.cpu().to(torch.int32),
            "teacher_logsumexp": torch.logsumexp(logits.float(), dim=-1).cpu().to(torch.float32),
        })
        del logits, top_values, top_indices
    valid = validate_cache_rows(rows, expected_rows=len(CALIBRATION_TEXTS), top_k=args.top_k)
    payload = {
        "schema": "clark_qwen38_qat_teacher_cache_v1",
        "model_revision": sensitivity["model_revision"],
        "selected_layer": sensitivity["selected_layer"],
        "top_k": args.top_k,
        "max_length": args.max_length,
        "rows": rows,
    }
    cache_sha256 = atomic_torch_save(args.out, payload)
    receipt = {
        "schema": "clark_qwen38_qat_teacher_cache_receipt_v1",
        "hypothesis": "Eight fixed local calibration sequences produce finite BF16 top-64 targets for bounded one-layer QAT.",
        "model_revision": payload["model_revision"],
        "selected_layer": payload["selected_layer"],
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(preregistration_content).hexdigest(),
        },
        "cache": str(args.out),
        "cache_sha256": cache_sha256,
        "rows": [
            {"index": row["index"], "tokens": int(row["input_ids"].numel()), "top_k": int(row["top_values"].shape[1])}
            for row in rows
        ],
        "all_finite_and_shape_valid": valid,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "promotion": {"passed": valid, "scope": "local teacher cache only; no student training"},
        "stop_condition": "Eight sequences or first identity, shape, non-finite, OOM, or runtime failure.",
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not valid or not all(math.isfinite(value) for value in (receipt["elapsed_seconds"],)):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
