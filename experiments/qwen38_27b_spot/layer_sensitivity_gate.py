#!/usr/bin/env python3
"""Select the first QAT layer by one-at-a-time BF16 restoration sensitivity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from behavioral_quality_gate import (
    HELD_OUT_TEXTS,
    apply_artifact,
    atomic_json,
    evaluate_likelihoods,
    resolve_parameter_map,
    validate_terminal_quality_receipt,
)
from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader


SELECTION_RULE = "Maximum held-out teacher-KL recovery per GiB restored, then maximum absolute teacher-KL recovery."


def validate_sensitivity_preregistration(prereg: dict, layers: list[int]) -> dict[int, dict]:
    candidates = prereg.get("sensitivity_candidates", [])
    expected_layers = [row.get("layer") for row in candidates if isinstance(row, dict)]
    if (
        prereg.get("schema") != "clark_qwen38_one_layer_qat_preregistration_v1"
        or expected_layers != layers
        or len(set(expected_layers)) != len(expected_layers)
        or prereg.get("selection_rule") != SELECTION_RULE
    ):
        raise ValueError("sensitivity candidates or selection rule differ from preregistration")
    return {int(row["layer"]): row for row in candidates}


def layer_tensor_names(reader: QwenBase3ArtifactReader, layer: int) -> tuple[str, ...]:
    prefix = f"model.language_model.layers.{layer}."
    names = tuple(name for name in reader.tensor_names if name.startswith(prefix))
    if not names:
        raise ValueError(f"artifact has no mapped tensors for layer {layer}")
    return names


@torch.inference_mode()
def restore_source_tensors(model, reader, model_dir: Path, tensor_names) -> dict[str, int]:
    try:
        named_parameters = dict(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover
        named_parameters = dict(model.named_parameters())
    mapping = resolve_parameter_map(tensor_names, named_parameters)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name in tensor_names:
        by_shard.setdefault(weight_map[name], []).append(name)
    parameters = tensors = 0
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                source = handle.get_tensor(name)
                target = named_parameters[mapping[name]]
                if list(source.shape) != list(target.shape):
                    raise ValueError(f"source/runtime shape mismatch for {name}")
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                parameters += source.numel()
                tensors += 1
    return {"tensors": tensors, "parameters": parameters, "bf16_bytes": parameters * 2}


def select_layer(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no layer sensitivity rows")
    return max(rows, key=lambda row: (row["teacher_kl_recovery_per_gib"], row["teacher_kl_recovery"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"),
    )
    parser.add_argument("--pack-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-full.json"))
    parser.add_argument("--quality-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-quality.json"))
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-quality/teacher-logits.pt"),
    )
    parser.add_argument("--layers", default="0,3,31,63")
    parser.add_argument("--preregistration", type=Path, default=HERE / "qat_preregistration.json")
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/qwen38-layer-sensitivity.json"))
    args = parser.parse_args()
    layers = [int(value.strip()) for value in args.layers.split(",") if value.strip()]
    if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise ValueError("layers must be unique non-negative integers")
    preregistration_content = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_content)
    expected_candidates = validate_sensitivity_preregistration(preregistration, layers)

    pack_receipt = json.loads(args.pack_receipt.read_text())
    if not pack_receipt.get("promotion", {}).get("passed"):
        raise ValueError("full pack did not pass promotion")
    quality_receipt = json.loads(args.quality_receipt.read_text())
    quality_entry = validate_terminal_quality_receipt(
        quality_receipt,
        pack_receipt,
        teacher_cache=args.teacher_cache,
    )
    teacher_cache = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
    expected_revision = pack_receipt["identity"]["model_revision"]
    if (
        preregistration.get("model", {}).get("revision") != expected_revision
        or
        teacher_cache.get("schema") != "clark_qwen38_bf16_teacher_logits_v1"
        or teacher_cache.get("model_revision") != expected_revision
        or tuple(teacher_cache.get("texts", ())) != HELD_OUT_TEXTS
        or len(teacher_cache.get("logits", ())) != len(HELD_OUT_TEXTS)
    ):
        raise ValueError("BF16 teacher cache identity mismatch")
    teachers = teacher_cache["logits"]
    reader = QwenBase3ArtifactReader(args.artifact)
    for layer in layers:
        names = layer_tensor_names(reader, layer)
        parameters = sum(int(reader.sources[name]["trits"]) for name in names)
        expected = expected_candidates[layer]
        if (
            len(names) != expected.get("mapped_tensors")
            or parameters != expected.get("mapped_parameters")
            or parameters * 2 != expected.get("bf16_restore_bytes")
        ):
            raise ValueError(f"layer {layer} mapped inventory differs from preregistration")

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
    model.eval()
    applied = apply_artifact(model, reader)
    baseline, _ = evaluate_likelihoods(model, tokenizer, HELD_OUT_TEXTS, "cuda", teachers=teachers)
    rows = []
    for layer in layers:
        names = layer_tensor_names(reader, layer)
        restored = restore_source_tensors(model, reader, args.model_dir, names)
        metrics, _ = evaluate_likelihoods(model, tokenizer, HELD_OUT_TEXTS, "cuda", teachers=teachers)
        recovery = baseline["mean_teacher_kl"] - metrics["mean_teacher_kl"]
        rows.append({
            "layer": layer,
            "mapped_tensor_names": names,
            "restored": restored,
            "metrics": metrics,
            "teacher_kl_recovery": recovery,
            "teacher_kl_recovery_per_gib": recovery / max(restored["bf16_bytes"] / 2**30, 1e-12),
        })
        apply_artifact(model, reader, tensor_names=names)
    reverted, _ = evaluate_likelihoods(model, tokenizer, HELD_OUT_TEXTS, "cuda", teachers=teachers)
    reversion_error = abs(reverted["mean_teacher_kl"] - baseline["mean_teacher_kl"])
    selected = select_layer(rows)
    finite = all(
        math.isfinite(row["metrics"]["mean_teacher_kl"])
        and math.isfinite(row["teacher_kl_recovery_per_gib"])
        for row in rows
    )
    receipt = {
        "schema": "clark_qwen38_layer_sensitivity_v1",
        "hypothesis": (
            "One-at-a-time BF16 layer restoration identifies a finite, reproducible teacher-KL sensitivity "
            "ranking that can select the first bounded one-layer QAT target."
        ),
        "model_revision": expected_revision,
        "artifact": str(args.artifact),
        "quality_gate_passed": quality_receipt.get("promotion", {}).get("passed"),
        "quality_entry_validation": quality_entry,
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(preregistration_content).hexdigest(),
        },
        "all_ternary_baseline": baseline,
        "layers": rows,
        "selection_rule": SELECTION_RULE,
        "selected_layer": selected["layer"],
        "selected_teacher_kl_recovery": selected["teacher_kl_recovery"],
        "reverted_baseline": reverted,
        "reversion_mean_teacher_kl_error": reversion_error,
        "applied": applied,
        "elapsed_seconds": time.time() - started,
        "promotion": {
            "passed": finite and reversion_error <= 1e-4 and selected["teacher_kl_recovery"] > 0,
            "scope": "layer selection for a bounded QAT probe; not a trained-model quality result",
        },
        "stop_condition": "Four representative layers or first identity, mapping, OOM, non-finite, or reversion failure.",
    }
    atomic_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["promotion"]["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
