#!/usr/bin/env python3
"""Localize behavioral collapse in a dense Qwen hard-ternary weight patch.

The script keeps the original BF16 model as the reference for excluded tensor
families, applies the retained patch everywhere else, and measures three
independent signals on a small frozen teacher cache:

* teacher top-k KL / probability mass / top-1 agreement;
* projected hidden-state drift at the cache's sealed layer boundaries; and
* deterministic open-loop generations on fixed smoke prompts.

Run one arm per process so every arm begins from the exact BF16 parent.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import transformers

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "qwen_agentworld"))

from eval_teacher_behavior_cache import topk_teacher_metrics  # noqa: E402
from label_teacher_cache_hidden_proj import hidden_projection  # noqa: E402
from train_mlp_triplet_survival_qat import (  # noqa: E402
    first_parameter_device,
    load_model,
    register_hidden_hooks,
    row_to_inputs,
)
from weight_patch import apply_weight_patch  # noqa: E402


DEFAULT_PROMPTS = (
    "Write one coherent sentence explaining why quantization-aware training can preserve model quality.",
    "What is 17 multiplied by 23? Give only the number.",
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function named square that returns the square of its numeric argument.",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if sidecar.exists():
        cached = json.loads(sidecar.read_text())
        if int(cached.get("size_bytes") or -1) == path.stat().st_size:
            value = str(cached.get("sha256") or "")
            if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
                return value
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def select_balanced_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows_per_task: int,
    seed: int,
    max_tokens: int | None,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if max_tokens is None or int(row.get("tokens") or 0) <= max_tokens]
    random.Random(seed).shuffle(candidates)
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in candidates:
        task = str(row.get("task"))
        if counts[task] >= max_rows_per_task:
            continue
        selected.append(row)
        counts[task] += 1
    return selected


def projected_hidden_metrics(student: torch.Tensor, teacher: torch.Tensor) -> dict[str, float]:
    student = student.float()
    teacher = teacher.to(device=student.device, dtype=torch.float32)
    if student.shape != teacher.shape:
        raise ValueError(f"projected hidden shape mismatch: {tuple(student.shape)} != {tuple(teacher.shape)}")
    relative_mse = (student - teacher).square().mean() / teacher.square().mean().clamp_min(1e-8)
    cosine = F.cosine_similarity(student, teacher, dim=-1).mean()
    return {"relative_mse": float(relative_mse.item()), "cosine": float(cosine.item())}


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    positions = sum(int(row["positions"]) for row in rows)
    if not rows:
        return {"rows": 0, "positions": 0}
    keys = (
        "teacher_topk_kl_full_mean",
        "teacher_topk_kl_topk_mean",
        "student_mass_on_teacher_topk_mean",
        "top1_agreement",
    )
    return {
        "rows": len(rows),
        "positions": positions,
        **{key: sum(float(row[key]) for row in rows) / len(rows) for key in keys},
    }


def summarize_hidden(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        for layer, metrics in row["hidden"].items():
            grouped[int(layer)].append(metrics)
    return {
        str(layer): {
            "rows": len(values),
            "relative_mse": sum(value["relative_mse"] for value in values) / len(values),
            "cosine": sum(value["cosine"] for value in values) / len(values),
        }
        for layer, values in sorted(grouped.items())
    }


def chat_prompt(tokenizer: Any, text: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def run_generations(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: tuple[str, ...],
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    device = first_parameter_device(model)
    results = []
    for prompt in prompts:
        encoded = tokenizer(chat_prompt(tokenizer, prompt), return_tensors="pt").to(device)
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = output[0, encoded.input_ids.shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(
            {
                "prompt": prompt,
                "text": text,
                "tokens": int(generated.numel()),
                "empty": not bool(text.strip()),
                "unique_whitespace_tokens": len(set(text.split())),
            }
        )
        print(json.dumps({"event": "generation", **results[-1]}, ensure_ascii=False), flush=True)
    return results


def release_non_text_modules(model: torch.nn.Module) -> None:
    core = getattr(model, "model", model)
    for attribute in ("visual", "vision_model", "vision_tower"):
        if getattr(core, attribute, None) is not None:
            setattr(core, attribute, None)
    if getattr(model, "mtp", None) is not None:
        model.mtp = None
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--exclude-regex")
    parser.add_argument("--include-regex")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rows-per-task", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--row-seed", type=int, default=20260816)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("format") != "qwen_agentworld_teacher_behavior_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    hidden_layers = [int(value) for value in cache.get("hidden_layers") or []]
    projection_dim = int(cache.get("hidden_projection_dim") or 0)
    projection_seed = int(cache.get("hidden_projection_seed") or 0)
    if not hidden_layers or projection_dim <= 0:
        raise ValueError("cache does not contain projected hidden labels")
    selected = select_balanced_rows(
        cache["rows"],
        max_rows_per_task=args.max_rows_per_task,
        seed=args.row_seed,
        max_tokens=args.max_tokens,
    )
    if not selected:
        raise RuntimeError("no cache rows selected")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()
    started = datetime.now(timezone.utc)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = load_model(args.model_dir)
    release_non_text_modules(model)
    patch_result = None
    if args.patch is not None:
        patch_result = apply_weight_patch(
            model,
            args.patch,
            exclude_regex=args.exclude_regex,
            include_regex=args.include_regex,
        )
        if patch_result["missing_count"] or not patch_result["applied_count"]:
            raise RuntimeError(f"patch application failed: {patch_result}")
    elif args.exclude_regex is not None:
        parser.error("--exclude-regex requires --patch")
    elif args.include_regex is not None:
        parser.error("--include-regex requires --patch")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.config.use_cache = True

    device = first_parameter_device(model)
    keep_ref = {"keep": 1}
    handles, captured = register_hidden_hooks(model, hidden_layers, keep_ref)
    projection_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    metric_rows: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected, start=1):
            keep = int(row["keep"])
            keep_ref["keep"] = keep
            captured.clear()
            with torch.inference_mode():
                logits = model(
                    **row_to_inputs(row, device),
                    use_cache=False,
                    logits_to_keep=keep,
                ).logits[0]
            metrics = topk_teacher_metrics(
                logits,
                row["top_values"],
                row["top_indices"],
                temperature=args.temperature,
            )
            hidden: dict[int, dict[str, float]] = {}
            for layer in hidden_layers:
                student = captured[layer].float()
                projection = hidden_projection(
                    layer_id=layer,
                    hidden_dim=student.shape[-1],
                    projected_dim=projection_dim,
                    seed=projection_seed,
                    device=student.device,
                    cache=projection_cache,
                )
                projected = F.layer_norm(student, student.shape[-1:]) @ projection
                hidden[layer] = projected_hidden_metrics(projected, row["teacher_hidden_proj"][layer])
            metric_row = {
                "task": row.get("task"),
                "id": row.get("id"),
                "tokens": int(row["tokens"]),
                **metrics,
                "hidden": hidden,
            }
            metric_rows.append(metric_row)
            print(
                json.dumps(
                    {
                        "event": "cache_row",
                        "index": index,
                        "rows": len(selected),
                        "task": metric_row["task"],
                        "id": metric_row["id"],
                        "kl": metrics["teacher_topk_kl_full_mean"],
                        "top1": metrics["top1_agreement"],
                    }
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    generations = [] if args.skip_generation else run_generations(
        model,
        tokenizer,
        DEFAULT_PROMPTS,
        max_new_tokens=args.max_new_tokens,
    )
    finished = datetime.now(timezone.utc)
    summary = summarize_metric_rows(metric_rows)
    result = {
        "schema": "qwen38_hard_patch_forensic_ablation_v1",
        "arm": args.arm,
        "model_dir": str(args.model_dir),
        "cache": str(args.cache),
        "patch": str(args.patch) if args.patch is not None else None,
        "patch_sha256": sha256_file(args.patch) if args.patch is not None else None,
        "exclude_regex": args.exclude_regex,
        "include_regex": args.include_regex,
        "patch_application": (
            {
                "applied_count": patch_result["applied_count"],
                "skipped_count": patch_result["skipped_count"],
                "missing_count": patch_result["missing_count"],
                "applied_parameters": sum(int(row["params"]) for row in patch_result["applied"]),
                "skipped_parameters": sum(int(row.get("params") or 0) for row in patch_result["skipped"]),
            }
            if patch_result is not None
            else None
        ),
        "selection": {
            "max_rows_per_task": args.max_rows_per_task,
            "max_tokens": args.max_tokens,
            "row_seed": args.row_seed,
            "row_ids": [row.get("id") for row in selected],
        },
        "summary": summary,
        "hidden_summary_by_layer": summarize_hidden(metric_rows),
        "generations": generations,
        "rows": metric_rows,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "peak_accelerator_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"event": "complete", "out": str(args.out), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
