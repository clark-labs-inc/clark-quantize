#!/usr/bin/env python3
"""Authenticated step-10 to step-100 continuation of the selected-layer QAT rung."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

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
    validate_terminal_quality_receipt,
)
from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader
from clark_quantize.qwen_qat import (
    checkpoint_rng_state_valid,
    load_qat_checkpoint,
    save_qat_checkpoint,
    state_tree_equal,
)
from one_layer_qat import (
    calibration_loss,
    clip_and_measure_gradient_norm,
    file_sha256,
    optimizer_for,
    projection_metrics,
    replace_selected_layer,
    tensor_metadata,
)
from qat_teacher_cache import validate_cache_rows
from prune_qat_checkpoints import prune


SCHEMA = "clark_qwen38_one_layer_qat_100_v1"


def update_regression_streak(previous_kl: float, current_kl: float, streak: int) -> int:
    if not all(math.isfinite(value) for value in (previous_kl, current_kl)) or streak < 0:
        raise ValueError("invalid held-out KL regression state")
    return streak + 1 if current_kl > previous_kl else 0


def validate_second_rung(prereg: dict) -> dict:
    second = prereg.get("second_rung", {})
    if (
        second.get("resume_checkpoint_step") != 10
        or second.get("optimizer_steps_total") != 100
        or second.get("evaluation_steps") != list(range(20, 101, 10))
        or second.get("maximum_complete_checkpoints") != 2
        or second.get("stop_after_consecutive_heldout_kl_regressions") != 2
    ):
        raise ValueError("second-rung step, evaluation, or retention contract changed")
    return second


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--artifact", type=Path, default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"))
    parser.add_argument("--pack-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-full.json"))
    parser.add_argument("--quality-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-quality.json"))
    parser.add_argument("--sensitivity", type=Path, default=Path("/workspace/quant-state/qwen38-layer-sensitivity.json"))
    parser.add_argument("--calibration-cache", type=Path, default=Path("/workspace/quant-runs/qwen38-qat/teacher-cache.pt"))
    parser.add_argument("--calibration-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-qat-teacher-cache.json"))
    parser.add_argument("--heldout-cache", type=Path, default=Path("/workspace/quant-runs/qwen38-base3-quality/teacher-logits.pt"))
    parser.add_argument("--first-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-one-layer-qat.json"))
    parser.add_argument("--preregistration", type=Path, default=HERE / "qat_preregistration.json")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/workspace/quant-runs/qwen38-qat/checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/qwen38-one-layer-qat-100.json"))
    parser.add_argument("--row-batch", type=int, default=128)
    args = parser.parse_args()

    prereg_content = args.preregistration.read_bytes()
    prereg = json.loads(prereg_content)
    first_config = prereg["first_rung"]
    second = validate_second_rung(prereg)
    pack = json.loads(args.pack_receipt.read_text())
    quality = json.loads(args.quality_receipt.read_text())
    sensitivity = json.loads(args.sensitivity.read_text())
    calibration_receipt = json.loads(args.calibration_receipt.read_text())
    first_receipt = json.loads(args.first_receipt.read_text())
    if not all(
        receipt.get("promotion", {}).get("passed") is True
        for receipt in (pack, sensitivity, calibration_receipt, first_receipt)
    ):
        raise ValueError("pack, sensitivity, teacher-cache, and ten-step promotions are required")
    quality_entry = validate_terminal_quality_receipt(
        quality,
        pack,
        teacher_cache=args.heldout_cache,
    )
    revision = pack["identity"]["model_revision"]
    if (
        prereg["model"]["revision"] != revision
        or quality.get("model_revision") != revision
        or sensitivity.get("model_revision") != revision
        or calibration_receipt.get("model_revision") != revision
        or first_receipt.get("model_revision") != revision
        or first_receipt.get("selected_layer") != sensitivity.get("selected_layer")
    ):
        raise ValueError("100-step receipt revision or selected-layer chain mismatch")
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
        raise ValueError("teacher cache path or SHA-256 differs from anchor receipts")
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    heldout = torch.load(args.heldout_cache, map_location="cpu", weights_only=False)
    if (
        calibration.get("schema") != "clark_qwen38_qat_teacher_cache_v1"
        or calibration.get("model_revision") != revision
        or calibration.get("selected_layer") != sensitivity["selected_layer"]
        or tuple(row.get("text") for row in calibration.get("rows", ()))
        != tuple(first_config["calibration_texts"])
        or len(calibration.get("rows", ())) != first_config["calibration_sequences"]
        or calibration.get("top_k") != first_config["teacher_top_k"]
        or calibration.get("max_length") != first_config["maximum_sequence_tokens"]
        or not validate_cache_rows(calibration["rows"], expected_rows=first_config["calibration_sequences"], top_k=first_config["teacher_top_k"])
        or heldout.get("schema") != "clark_qwen38_bf16_teacher_logits_v1"
        or heldout.get("model_revision") != revision
        or tuple(heldout.get("texts", ())) != HELD_OUT_TEXTS
    ):
        raise ValueError("100-step teacher cache identity mismatch")

    checkpoint_value = first_receipt.get("checkpoint", {}).get("path")
    checkpoint10 = Path(checkpoint_value) if isinstance(checkpoint_value, str) else Path("/")
    if checkpoint10 != args.checkpoint_root / "checkpoint-10":
        raise ValueError("ten-step receipt does not identify checkpoint-10 in the fixed root")
    loaded, checkpoint10_manifest = load_qat_checkpoint(checkpoint10, map_location="cpu")
    if (
        checkpoint10_manifest.get("state_sha256") != first_receipt.get("checkpoint", {}).get("state_sha256")
        or loaded.get("schema") != "clark_qwen38_one_layer_qat_state_v1"
        or loaded.get("step") != 10
        or loaded.get("model_revision") != revision
        or loaded.get("selected_layer") != sensitivity["selected_layer"]
    ):
        raise ValueError("checkpoint-10 identity differs from its first-rung receipt")
    complete_before = sorted(path.name for path in args.checkpoint_root.glob("checkpoint-*") if path.is_dir())
    if complete_before != ["checkpoint-10"]:
        raise ValueError(f"100-step rung requires only checkpoint-10, found {complete_before}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    if not replacement["exact_step_zero"] or set(loaded.get("masters", {})) != set(wrappers):
        raise ValueError("100-step wrapper mapping does not reproduce the authenticated step-zero artifact")
    for name, module in wrappers.items():
        module.master.copy_(loaded["masters"][name].to(module.master.device))
    learning_rate = float(first_config["learning_rate"])
    kl_weight = float(first_config["teacher_kl_weight"])
    ce_weight = float(first_config["causal_ce_weight"])
    grad_clip = float(first_config["gradient_clip_norm"])
    optimizer = optimizer_for(wrappers, learning_rate)
    trainable_parameters = [module.master for module in wrappers.values()]
    optimizer.load_state_dict(loaded["optimizer"])
    if optimizer.global_step != 10:
        raise ValueError(f"resumed optimizer global step is {optimizer.global_step}, expected 10")
    # Model construction may consume RNG. Restore the authenticated trainer
    # states only after model/wrapper/optimizer reconstruction is complete.
    torch.random.set_rng_state(loaded["cpu_rng_state"])
    torch.cuda.set_rng_state_all(loaded["cuda_rng_state"])
    rng_state_restored = (
        checkpoint_rng_state_valid(loaded, cuda_devices=torch.cuda.device_count())
        and torch.equal(torch.random.get_rng_state(), loaded["cpu_rng_state"])
        and all(
            torch.equal(actual.cpu(), expected.cpu())
            for actual, expected in zip(torch.cuda.get_rng_state_all(), loaded["cuda_rng_state"])
        )
    )
    if not rng_state_restored:
        raise ValueError("checkpoint-10 RNG state did not restore exactly")

    resumed_heldout, _ = evaluate_likelihoods(model, tokenizer, HELD_OUT_TEXTS, "cuda", teachers=heldout["logits"])
    first_final = first_receipt["final_heldout"]
    if any(abs(resumed_heldout[key] - first_final[key]) > 1e-4 for key in ("mean_nll", "mean_teacher_kl")):
        raise ValueError("checkpoint-10 does not reproduce the first-rung final held-out metrics")
    history = list(loaded.get("history", []))
    evaluations = [{"step": 10, "likelihood": resumed_heldout}]
    previous_kl = float(resumed_heldout["mean_teacher_kl"])
    regression_streak = 0
    stop_reason = None
    completed_step = 10
    evaluation_steps = set(second["evaluation_steps"])
    for step in range(11, int(second["optimizer_steps_total"]) + 1):
        row = calibration["rows"][(step - 1) % len(calibration["rows"])]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = calibration_loss(model, row, kl_weight=kl_weight, ce_weight=ce_weight)
        if not torch.isfinite(loss):
            stop_reason = f"non-finite loss at step {step}"
            break
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
            stop_reason = (
                f"invalid post-clip gradient norm at step {step}: "
                f"before={float(gradient_norm_before)}, after={float(gradient_norm_after)}"
            )
            break
        optimizer.step()
        completed_step = step
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
        if step in evaluation_steps:
            likelihood, _ = evaluate_likelihoods(model, tokenizer, HELD_OUT_TEXTS, "cuda", teachers=heldout["logits"])
            regression_streak = update_regression_streak(previous_kl, float(likelihood["mean_teacher_kl"]), regression_streak)
            evaluations.append({"step": step, "likelihood": likelihood, "regression_streak": regression_streak})
            previous_kl = float(likelihood["mean_teacher_kl"])
            if regression_streak >= int(second["stop_after_consecutive_heldout_kl_regressions"]):
                stop_reason = f"held-out teacher KL regressed at two consecutive evaluations through step {step}"
                break

    final_heldout = evaluations[-1]["likelihood"] if evaluations[-1]["step"] == completed_step else evaluate_likelihoods(
        model, tokenizer, HELD_OUT_TEXTS, "cuda", teachers=heldout["logits"]
    )[0]
    generations = generate_controls(model, tokenizer, GENERATION_PROMPTS, "cuda", max_new_tokens=24)
    projection = projection_metrics(wrappers, reader)
    masters = {name: module.master.detach().cpu().contiguous() for name, module in wrappers.items()}
    checkpoint_state = {
        "schema": "clark_qwen38_one_layer_qat_state_v1",
        "step": completed_step,
        "model_revision": revision,
        "selected_layer": sensitivity["selected_layer"],
        "masters": masters,
        "optimizer": optimizer.state_dict(),
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "history": history,
    }
    if completed_step == 10:
        terminal_checkpoint = checkpoint10
    else:
        terminal_checkpoint = save_qat_checkpoint(
            args.checkpoint_root,
            step=completed_step,
            state=checkpoint_state,
            tensor_metadata=tensor_metadata(masters),
            metadata={
                "model_revision": revision,
                "selected_layer": sensitivity["selected_layer"],
                "artifact_manifest_sha256": pack["code_manifest_sha256"],
                "resumed_from_state_sha256": checkpoint10_manifest["state_sha256"],
            },
        )
    del checkpoint_state, masters, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, terminal_manifest = load_qat_checkpoint(terminal_checkpoint, map_location="cpu")
    resumed_optimizer = optimizer_for(wrappers, learning_rate)
    resumed_optimizer.load_state_dict(reloaded["optimizer"])
    optimizer_state_exact = state_tree_equal(
        resumed_optimizer.state_dict(), reloaded["optimizer"]
    )
    rng_state_complete = checkpoint_rng_state_valid(
        reloaded, cuda_devices=torch.cuda.device_count()
    )
    exact_resume = (
        reloaded.get("step") == completed_step
        and reloaded.get("model_revision") == revision
        and reloaded.get("selected_layer") == sensitivity["selected_layer"]
        and set(reloaded.get("masters", {})) == set(wrappers)
        and resumed_optimizer.global_step == completed_step
        and optimizer_state_exact
        and rng_state_complete
        and all(torch.equal(module.master.detach().cpu(), reloaded["masters"][name]) for name, module in wrappers.items())
    )
    del reloaded, resumed_optimizer
    retention = prune(args.checkpoint_root, keep=2)
    completed_checkpoints = sorted(path.name for path in args.checkpoint_root.glob("checkpoint-*") if path.is_dir())
    initial = first_receipt["initial_heldout"]
    relative_kl_improvement = (initial["mean_teacher_kl"] - final_heldout["mean_teacher_kl"]) / max(abs(initial["mean_teacher_kl"]), 1e-12)
    relative_nll_regression = (final_heldout["mean_nll"] - initial["mean_nll"]) / max(abs(initial["mean_nll"]), 1e-12)
    nonempty = all(row["completion"].strip() for row in generations)
    rule = prereg["promotion_after_100_steps"]
    passed = (
        completed_step == 100
        and stop_reason is None
        and exact_resume
        and len(completed_checkpoints) <= 2
        and relative_kl_improvement >= float(rule["minimum_relative_teacher_kl_improvement_from_step_zero"])
        and relative_nll_regression <= float(rule["maximum_relative_nll_regression_from_step_zero"])
        and nonempty
    )
    receipt = {
        "schema": SCHEMA,
        "hypothesis": "Authenticated continuation to 100 hard-forward selected-layer QAT steps preserves the ten-step gain without material held-out NLL regression.",
        "model_revision": revision,
        "selected_layer": sensitivity["selected_layer"],
        "quality_entry_validation": quality_entry,
        "preregistration": {"path": str(args.preregistration), "sha256": hashlib.sha256(prereg_content).hexdigest()},
        "resumed_from": {"checkpoint": str(checkpoint10), "state_sha256": checkpoint10_manifest["state_sha256"], "exact_heldout_reproduction": True},
        "applied": applied,
        "replacement": replacement,
        "completed_step": completed_step,
        "stop_reason": stop_reason,
        "history": history,
        "evaluations": evaluations,
        "initial_heldout": initial,
        "step10_heldout": resumed_heldout,
        "final_heldout": final_heldout,
        "generations": generations,
        "projection_metrics": projection,
        "comparison": {
            "relative_teacher_kl_improvement_from_step_zero": relative_kl_improvement,
            "relative_nll_regression_from_step_zero": relative_nll_regression,
            "all_completions_nonempty": nonempty,
        },
        "checkpoint": {
            "path": str(terminal_checkpoint),
            "state_sha256": terminal_manifest["state_sha256"],
            "exact_resume_smoke": exact_resume,
            "optimizer_state_exact": optimizer_state_exact,
            "rng_state_complete": rng_state_complete,
            "checkpoint10_rng_state_restored": rng_state_restored,
            "completed_checkpoints": completed_checkpoints,
            "retention": retention,
        },
        "resources": {
            "elapsed_seconds": time.time() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device": torch.cuda.get_device_name(0),
        },
        "promotion": {"passed": passed, "scope": "100-step selected-layer QAT result; no global-QAT or complete-model quality claim"},
        "stop_condition": "100 steps or the first preregistered numerical, OOM, held-out regression, checkpoint, or retention failure.",
    }
    atomic_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
