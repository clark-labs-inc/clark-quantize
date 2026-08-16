#!/usr/bin/env python3
"""Preregistered ten-step, one-layer hard-ternary QAT rung for Qwen3.8-27B."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPOT_SCRIPTS = REPO / "scripts/spot"
for path in (REPO, HERE, SPOT_SCRIPTS):
    sys.path.insert(0, str(path))

from behavioral_quality_gate import (
    GENERATION_PROMPTS,
    HELD_OUT_TEXTS,
    apply_artifact,
    atomic_json,
    evaluate_likelihoods,
    generate_controls,
    resolve_parameter_map,
    validate_terminal_quality_receipt,
)
from clark_quantize.bonsai_qat8 import BonsaiQAT8
from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader
from clark_quantize.qwen_qat import (
    HardTernaryLinear,
    checkpoint_rng_state_valid,
    hard_ternary_weight,
    load_qat_checkpoint,
    save_qat_checkpoint,
    state_tree_equal,
    topk_tail_teacher_kl,
)
from clark_quantize.ternary import ternary_codes_scales
from layer_sensitivity_gate import layer_tensor_names
from qat_teacher_cache import validate_cache_rows
from prune_qat_checkpoints import prune


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def clip_and_measure_gradient_norm(
    parameters: list[torch.nn.Parameter],
    maximum_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip gradients and return the distinct pre/post-clip total norms."""
    before = torch.nn.utils.clip_grad_norm_(parameters, maximum_norm)
    component_norms = [
        torch.linalg.vector_norm(parameter.grad.detach()).float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    after = (
        torch.linalg.vector_norm(torch.stack(component_norms))
        if component_norms
        else torch.zeros((), device=parameters[0].device if parameters else "cpu")
    )
    return before, after


def validate_step10_checkpoint_state(
    state: dict[str, Any],
    *,
    revision: str,
    selected_layer: int,
    wrapper_names: set[str],
) -> list[dict[str, Any]]:
    history = state.get("history")
    valid_history = (
        isinstance(history, list)
        and len(history) == 10
        and [row.get("step") for row in history] == list(range(1, 11))
        and all(
            isinstance(row.get("gradient_norm_after_clip"), (int, float))
            and math.isfinite(row["gradient_norm_after_clip"])
            and row["gradient_norm_after_clip"] <= 10.0
            for row in history
        )
    )
    if (
        state.get("schema") != "clark_qwen38_one_layer_qat_state_v1"
        or state.get("step") != 10
        or state.get("model_revision") != revision
        or state.get("selected_layer") != selected_layer
        or set(state.get("masters", {})) != wrapper_names
        or not valid_history
    ):
        raise ValueError("step-10 checkpoint trainer identity, mapping, or history is invalid")
    return history


def parent_and_attr(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parent_name, _, attr = module_name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), attr


@torch.no_grad()
def replace_selected_layer(
    model,
    reader: QwenBase3ArtifactReader,
    model_dir: Path,
    layer: int,
    *,
    group_size: int,
    row_batch: int,
) -> tuple[dict[str, HardTernaryLinear], dict[str, Any]]:
    names = layer_tensor_names(reader, layer)
    try:
        named_parameters = dict(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover
        named_parameters = dict(model.named_parameters())
    mapping = resolve_parameter_map(names, named_parameters)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wrappers: dict[str, HardTernaryLinear] = {}
    exact_step_zero = True
    parameters = 0
    for source_name in names:
        runtime_weight_name = mapping[source_name]
        if not runtime_weight_name.endswith(".weight"):
            raise ValueError(f"runtime tensor is not a weight: {runtime_weight_name}")
        module_name = runtime_weight_name[: -len(".weight")]
        source_module = model.get_submodule(module_name)
        if not isinstance(source_module, torch.nn.Linear):
            raise TypeError(f"mapped tensor {source_name} targets non-Linear {type(source_module).__name__}")
        artifact_weight = source_module.weight.detach().clone()
        with safe_open(model_dir / weight_map[source_name], framework="pt", device="cpu") as handle:
            source_weight = handle.get_tensor(source_name)
        wrapper = HardTernaryLinear(
            source_weight,
            source_module.bias,
            group_size=group_size,
            row_batch=row_batch,
        ).to(device=source_module.weight.device)
        hard = wrapper.hard_weight(dtype=artifact_weight.dtype)
        exact_step_zero = exact_step_zero and bool(torch.equal(hard, artifact_weight))
        parent, attr = parent_and_attr(model, module_name)
        setattr(parent, attr, wrapper)
        wrappers[source_name] = wrapper
        parameters += wrapper.master.numel()
        del source_weight, artifact_weight, hard, source_module
        torch.cuda.empty_cache()
    return wrappers, {
        "layer": layer,
        "tensor_names": names,
        "tensors": len(wrappers),
        "parameters": parameters,
        "exact_step_zero": exact_step_zero,
    }


def optimizer_for(wrappers: dict[str, HardTernaryLinear], learning_rate: float) -> BonsaiQAT8:
    return BonsaiQAT8([{
        "params": [module.master for module in wrappers.values()],
        "role": "matrix_qat8",
        "lr": learning_rate,
        "betas": (0.9, 0.95),
        "use_muon": False,
        "threshold_boost": 0.0,
    }])


def calibration_loss(model, row: dict, *, kl_weight: float, ce_weight: float) -> tuple[torch.Tensor, dict]:
    device = next(model.parameters()).device
    input_ids = row["input_ids"].unsqueeze(0).to(device)
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    logits = outputs.logits
    kl, stats = topk_tail_teacher_kl(
        logits,
        row["top_values"],
        row["top_indices"],
        row["teacher_logsumexp"],
    )
    ce = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )
    loss = kl_weight * kl + ce_weight * ce
    return loss, {**stats, "causal_ce": float(ce.detach().cpu()), "loss": float(loss.detach().cpu())}


def tensor_metadata(masters: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in masters.items()
    }


@torch.no_grad()
def projection_metrics(
    wrappers: dict[str, HardTernaryLinear],
    reader: QwenBase3ArtifactReader,
) -> dict[str, float | int]:
    """Measure the final hard projection against the exact step-zero artifact."""
    parameters = zeros = changed = 0
    squared_error = squared_master = 0.0
    for name, module in wrappers.items():
        for record in reader.records_by_tensor[name]:
            descriptor = record["descriptor"]
            start = int(descriptor["row_start"])
            stop = int(descriptor["row_stop"])
            baseline_codes, _ = reader.raw_batch(record)
            master = module.master[start:stop]
            final_codes, _ = ternary_codes_scales(
                master.float(), group=module.group_size, method="optimal"
            )
            baseline = torch.from_numpy(baseline_codes).to(device=final_codes.device)
            if final_codes.shape != baseline.shape:
                raise ValueError(f"projection code shape mismatch for {name}")
            hard = hard_ternary_weight(
                master, group_size=module.group_size, row_batch=module.row_batch
            ).to(device=master.device, dtype=torch.float32)
            delta = master.float() - hard
            parameters += master.numel()
            zeros += int((final_codes == 0).sum().item())
            changed += int((final_codes != baseline).sum().item())
            squared_error += float(delta.square().sum().item())
            squared_master += float(master.float().square().sum().item())
    if parameters <= 0:
        raise ValueError("no QAT projection parameters")
    return {
        "parameters": parameters,
        "ternary_zero_fraction": zeros / parameters,
        "code_churn_fraction_from_step_zero": changed / parameters,
        "changed_codes_from_step_zero": changed,
        "master_to_hard_relative_l2": math.sqrt(squared_error / max(squared_master, 1e-30)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--artifact", type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"),
    )
    parser.add_argument("--pack-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-full.json"))
    parser.add_argument("--quality-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-quality.json"))
    parser.add_argument("--sensitivity", type=Path, default=Path("/workspace/quant-state/qwen38-layer-sensitivity.json"))
    parser.add_argument("--calibration-cache", type=Path, default=Path("/workspace/quant-runs/qwen38-qat/teacher-cache.pt"))
    parser.add_argument("--calibration-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-qat-teacher-cache.json"))
    parser.add_argument("--heldout-cache", type=Path, default=Path("/workspace/quant-runs/qwen38-base3-quality/teacher-logits.pt"))
    parser.add_argument("--preregistration", type=Path, default=HERE / "qat_preregistration.json")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/workspace/quant-runs/qwen38-qat/checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/qwen38-one-layer-qat.json"))
    parser.add_argument("--row-batch", type=int, default=128)
    parser.add_argument(
        "--finalize-checkpoint",
        action="store_true",
        help="Finalize an authenticated checkpoint-10 after a post-save receipt failure; do not retrain.",
    )
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    first = prereg["first_rung"]
    if first["optimizer_steps"] != 10 or first["maximum_complete_checkpoints"] != 2:
        raise ValueError("first-rung step/checkpoint contract changed")
    pack = json.loads(args.pack_receipt.read_text())
    quality = json.loads(args.quality_receipt.read_text())
    sensitivity = json.loads(args.sensitivity.read_text())
    calibration_receipt = json.loads(args.calibration_receipt.read_text())
    if not all(
        receipt.get("promotion", {}).get("passed")
        for receipt in (pack, sensitivity, calibration_receipt)
    ):
        raise ValueError("pack, sensitivity, and teacher-cache promotions are required")
    quality_entry = validate_terminal_quality_receipt(
        quality,
        pack,
        teacher_cache=args.heldout_cache,
    )
    if float(sensitivity.get("selected_teacher_kl_recovery", 0.0)) <= 0:
        raise ValueError("selected layer has no positive teacher-KL recovery")
    revision = pack["identity"]["model_revision"]
    if (
        prereg["model"]["revision"] != revision
        or quality.get("model_revision") != revision
        or sensitivity.get("model_revision") != revision
        or calibration_receipt.get("model_revision") != revision
    ):
        raise ValueError("preregistered receipt model revisions do not match")
    artifact_manifest = args.artifact / "codes.cq3d" / "manifest.json"
    if file_sha256(artifact_manifest) != pack.get("code_manifest_sha256"):
        raise ValueError("artifact manifest SHA-256 does not match pack receipt")
    quality_cache = quality.get("teacher_cache", {})
    if (
        Path(quality_cache.get("path", "")) != args.heldout_cache
        or file_sha256(args.heldout_cache) != quality_cache.get("sha256")
        or Path(calibration_receipt.get("cache", "")) != args.calibration_cache
        or file_sha256(args.calibration_cache) != calibration_receipt.get("cache_sha256")
    ):
        raise ValueError("teacher cache path or SHA-256 does not match its receipt")
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    heldout = torch.load(args.heldout_cache, map_location="cpu", weights_only=False)
    if (
        calibration.get("schema") != "clark_qwen38_qat_teacher_cache_v1"
        or calibration.get("model_revision") != revision
        or calibration.get("selected_layer") != sensitivity["selected_layer"]
        or tuple(row.get("text") for row in calibration.get("rows", ()))
        != tuple(first["calibration_texts"])
        or len(calibration.get("rows", ())) != first["calibration_sequences"]
        or calibration.get("top_k") != first["teacher_top_k"]
        or calibration.get("max_length") != first["maximum_sequence_tokens"]
        or not validate_cache_rows(
            calibration.get("rows", []),
            expected_rows=first["calibration_sequences"],
            top_k=first["teacher_top_k"],
        )
        or heldout.get("schema") != "clark_qwen38_bf16_teacher_logits_v1"
        or heldout.get("model_revision") != revision
        or tuple(heldout.get("texts", ())) != HELD_OUT_TEXTS
    ):
        raise ValueError("teacher cache identity mismatch")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(20260815)
    torch.cuda.manual_seed_all(20260815)
    torch.cuda.reset_peak_memory_stats()
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
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader = QwenBase3ArtifactReader(args.artifact)
    applied = apply_artifact(model, reader)
    wrappers, replacement = replace_selected_layer(
        model,
        reader,
        args.model_dir,
        int(sensitivity["selected_layer"]),
        group_size=int(prereg["representation"]["group_size"]),
        row_batch=args.row_batch,
    )
    if not replacement["exact_step_zero"]:
        raise ValueError("active layer hard forward differs from packed artifact at step zero")
    learning_rate = float(first["learning_rate"])
    kl_weight = float(first["teacher_kl_weight"])
    ce_weight = float(first["causal_ce_weight"])
    grad_clip = float(first["gradient_clip_norm"])
    optimizer = optimizer_for(wrappers, learning_rate)
    trainable_parameters = [module.master for module in wrappers.values()]
    baseline_kl = float(sensitivity["all_ternary_baseline"]["mean_teacher_kl"])
    checkpoint = args.checkpoint_root / "checkpoint-10"
    if args.finalize_checkpoint:
        loaded_checkpoint, existing_manifest = load_qat_checkpoint(checkpoint, map_location="cpu")
        history = validate_step10_checkpoint_state(
            loaded_checkpoint,
            revision=revision,
            selected_layer=int(sensitivity["selected_layer"]),
            wrapper_names=set(wrappers),
        )
        with torch.no_grad():
            for name, module in wrappers.items():
                module.master.copy_(loaded_checkpoint["masters"][name].to(module.master.device))
        optimizer.load_state_dict(loaded_checkpoint["optimizer"])
        if optimizer.global_step != 10:
            raise ValueError("checkpoint finalization optimizer is not at step 10")
        torch.random.set_rng_state(loaded_checkpoint["cpu_rng_state"])
        torch.cuda.set_rng_state_all(loaded_checkpoint["cuda_rng_state"])
        initial_heldout = sensitivity["all_ternary_baseline"]
        checkpoint_manifest = existing_manifest
        del loaded_checkpoint
    else:
        if checkpoint.exists():
            raise ValueError("checkpoint-10 already exists; use --finalize-checkpoint after inspection")
        initial_heldout, _ = evaluate_likelihoods(
            model, tokenizer, HELD_OUT_TEXTS, "cuda", teachers=heldout["logits"]
        )
        if abs(initial_heldout["mean_teacher_kl"] - baseline_kl) > 1e-4:
            raise ValueError("QAT step-zero held-out KL does not reproduce sensitivity baseline")

        history = []
        for step in range(1, int(first["optimizer_steps"]) + 1):
            row = calibration["rows"][(step - 1) % len(calibration["rows"])]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = calibration_loss(model, row, kl_weight=kl_weight, ce_weight=ce_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            gradient_norm_before, gradient_norm_after = clip_and_measure_gradient_norm(
                trainable_parameters,
                grad_clip,
            )
            if (
                not torch.isfinite(gradient_norm_before)
                or not torch.isfinite(gradient_norm_after)
                or float(gradient_norm_after) > 10.0
            ):
                raise FloatingPointError(
                    "invalid post-clip gradient norm at step "
                    f"{step}: before={float(gradient_norm_before)}, after={float(gradient_norm_after)}"
                )
            optimizer.step()
            record = {
                "step": step,
                "calibration_row": int(row["index"]),
                **metrics,
                "gradient_norm_before_clip": float(gradient_norm_before.detach().cpu()),
                "gradient_norm_after_clip": float(gradient_norm_after.detach().cpu()),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            del loss
            gc.collect()
            torch.cuda.empty_cache()

    final_heldout, _ = evaluate_likelihoods(
        model, tokenizer, HELD_OUT_TEXTS, "cuda", teachers=heldout["logits"]
    )
    generations = generate_controls(model, tokenizer, GENERATION_PROMPTS, "cuda", max_new_tokens=24)
    projection = projection_metrics(wrappers, reader)
    if not args.finalize_checkpoint:
        masters = {name: module.master.detach().cpu().contiguous() for name, module in wrappers.items()}
        checkpoint_state = {
            "schema": "clark_qwen38_one_layer_qat_state_v1",
            "step": 10,
            "model_revision": revision,
            "selected_layer": sensitivity["selected_layer"],
            "masters": masters,
            "optimizer": optimizer.state_dict(),
            "cpu_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "history": history,
        }
        checkpoint = save_qat_checkpoint(
            args.checkpoint_root,
            step=10,
            state=checkpoint_state,
            tensor_metadata=tensor_metadata(masters),
            metadata={
                "model_revision": revision,
                "selected_layer": sensitivity["selected_layer"],
                "artifact_manifest_sha256": pack["code_manifest_sha256"],
            },
        )
        checkpoint_manifest = json.loads((checkpoint / "checkpoint.json").read_text())
        del checkpoint_state, masters
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    loaded, checkpoint_manifest = load_qat_checkpoint(checkpoint, map_location="cpu")
    validate_step10_checkpoint_state(
        loaded,
        revision=revision,
        selected_layer=int(sensitivity["selected_layer"]),
        wrapper_names=set(wrappers),
    )
    with torch.no_grad():
        for name, module in wrappers.items():
            module.master.copy_(loaded["masters"][name].to(module.master.device))
    resumed_optimizer = optimizer_for(wrappers, learning_rate)
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    optimizer_state_exact = state_tree_equal(
        resumed_optimizer.state_dict(), loaded["optimizer"]
    )
    rng_state_complete = checkpoint_rng_state_valid(
        loaded, cuda_devices=torch.cuda.device_count()
    )
    resume_loss, _ = calibration_loss(
        model, calibration["rows"][0], kl_weight=kl_weight, ce_weight=ce_weight
    )
    exact_resume = (
        resumed_optimizer.global_step == 10
        and optimizer_state_exact
        and rng_state_complete
        and bool(torch.isfinite(resume_loss))
        and all(
            torch.equal(module.master.detach().cpu(), loaded["masters"][name])
            for name, module in wrappers.items()
        )
    )
    del loaded, resumed_optimizer, resume_loss
    retention = prune(args.checkpoint_root, keep=2)
    completed_checkpoints = sorted(path.name for path in args.checkpoint_root.glob("checkpoint-*") if path.is_dir())

    initial_kl = initial_heldout["mean_teacher_kl"]
    final_kl = final_heldout["mean_teacher_kl"]
    relative_kl_improvement = (initial_kl - final_kl) / max(abs(initial_kl), 1e-12)
    relative_nll_regression = (
        final_heldout["mean_nll"] - initial_heldout["mean_nll"]
    ) / max(abs(initial_heldout["mean_nll"]), 1e-12)
    nonempty = all(row["completion"].strip() for row in generations)
    promotion_rule = prereg["promotion_to_100_steps"]
    passed = (
        exact_resume
        and len(completed_checkpoints) <= 2
        and relative_kl_improvement >= float(promotion_rule["minimum_relative_teacher_kl_improvement"])
        and relative_nll_regression <= float(promotion_rule["maximum_relative_nll_regression"])
        and nonempty
    )
    receipt = {
        "schema": "clark_qwen38_one_layer_qat_v1",
        "hypothesis": "Ten hard-forward BonsaiQAT8 steps on the most sensitive layer improve held-out teacher KL without material NLL regression.",
        "model_revision": revision,
        "selected_layer": sensitivity["selected_layer"],
        "artifact": str(args.artifact),
        "quality_entry_validation": quality_entry,
        "resumed_finalization": args.finalize_checkpoint,
        "applied": applied,
        "replacement": replacement,
        "hyperparameters": first,
        "history": history,
        "initial_heldout": initial_heldout,
        "final_heldout": final_heldout,
        "generations": generations,
        "projection_metrics": projection,
        "comparison": {
            "relative_teacher_kl_improvement": relative_kl_improvement,
            "relative_nll_regression": relative_nll_regression,
            "all_completions_nonempty": nonempty,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "state_sha256": checkpoint_manifest["state_sha256"],
            "exact_resume_smoke": exact_resume,
            "optimizer_state_exact": optimizer_state_exact,
            "rng_state_complete": rng_state_complete,
            "completed_checkpoints": completed_checkpoints,
            "retention": retention,
        },
        "resources": {
            "elapsed_seconds": time.time() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device": torch.cuda.get_device_name(0),
        },
        "promotion": {
            "passed": passed,
            "scope": "ten-step one-layer trainability rung; no global QAT or quality-parity claim",
        },
        "stop_condition": "Ten steps or first identity, step-zero equivalence, non-finite, gradient, OOM, checkpoint, or retention failure.",
    }
    atomic_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
