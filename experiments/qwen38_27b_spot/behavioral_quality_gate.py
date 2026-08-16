#!/usr/bin/env python3
"""Compare the BF16 Qwen3.8-27B control with the packed ternary artifact.

The model is loaded once.  We first collect deterministic BF16 likelihood and
generation controls, then replace only the matrices present in the completed
base-3 artifact in bounded row batches.  This is a PTQ quality gate, not a
claim that packing efficiency implies useful model behavior.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader, decode_artifact_record


HELD_OUT_TEXTS = (
    "A reliable compression experiment reports both the storage rate and the quality of the reconstructed model.",
    "If a machine processes 18 batches per minute for 7 minutes, it processes 126 batches in total.",
    "The Pacific Ocean is larger than the Atlantic Ocean, while both cover substantial portions of Earth.",
    "When evidence contradicts a hypothesis, the correct scientific response is to revise or reject the hypothesis.",
)

GENERATION_PROMPTS = (
    "In one sentence, explain why a lower reconstruction error does not guarantee better language-model quality.",
    "A researcher should stop a quantization run early when",
    "Calculate 37 multiplied by 24 and explain the arithmetic briefly.",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_terminal_quality_receipt(
    receipt: dict[str, Any],
    pack_receipt: dict[str, Any],
    *,
    teacher_cache: Path | None = None,
) -> dict[str, Any]:
    """Validate a completed quality observation independently of promotion.

    A failed PTQ hypothesis is still valid input to the preregistered
    sensitivity/QAT rescue rung when it has complete, finite metrics,
    deterministic controls, exact artifact identity, and an authenticated
    local teacher cache.  Promotion remains a reported scientific outcome; it
    is deliberately not rewritten here.
    """
    comparison = receipt.get("comparison", {})
    integrity = receipt.get("artifact_integrity", {})
    applied = receipt.get("applied", {})
    expected_identity = pack_receipt.get("identity", {})
    expected_trits = pack_receipt.get("code_ledger", {}).get("trits")
    cache = receipt.get("teacher_cache", {})
    cache_path_value = cache.get("path")
    cache_path = Path(cache_path_value) if isinstance(cache_path_value, str) else None
    finite_values = (
        comparison.get("perplexity_ratio"),
        comparison.get("mean_teacher_kl"),
        receipt.get("bf16", {}).get("likelihood", {}).get("mean_nll"),
        receipt.get("bf16", {}).get("likelihood", {}).get("perplexity"),
        receipt.get("ternary", {}).get("likelihood", {}).get("mean_nll"),
        receipt.get("ternary", {}).get("likelihood", {}).get("perplexity"),
    )
    metrics_finite = all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in finite_values
    )
    artifact_valid = (
        integrity.get("valid") is True
        and integrity.get("artifact") == pack_receipt.get("artifact")
        and integrity.get("manifest_sha256") == pack_receipt.get("code_manifest_sha256")
        and integrity.get("code_source_sha256") == pack_receipt.get("code_source_sha256")
        and integrity.get("scale_source_sha256") == pack_receipt.get("scale_source_sha256")
        and integrity.get("model_revision") == expected_identity.get("model_revision")
        and integrity.get("index_sha256") == expected_identity.get("index_sha256")
        and integrity.get("trits") == expected_trits
        and applied.get("parameters") == expected_trits
        and isinstance(applied.get("tensors"), int)
        and applied.get("tensors", 0) > 0
        and isinstance(applied.get("batches"), int)
        and applied.get("batches", 0) > 0
    )
    deterministic = (
        comparison.get("all_bf16_and_ternary_completions_nonempty") is True
        and comparison.get("bf16_generations_deterministic") is True
        and comparison.get("ternary_generations_deterministic") is True
        and comparison.get("bf16_generation_sha256")
        == comparison.get("bf16_generation_repeat_sha256")
        and comparison.get("ternary_generation_sha256")
        == comparison.get("ternary_generation_repeat_sha256")
    )
    cache_valid = (
        cache_path is not None
        and cache_path.is_file()
        and isinstance(cache.get("sha256"), str)
        and file_sha256(cache_path) == cache.get("sha256")
        and (teacher_cache is None or cache_path.resolve() == teacher_cache.resolve())
    )
    valid = (
        receipt.get("schema") == "clark_qwen38_base3_behavioral_quality_v1"
        and receipt.get("model_revision") == expected_identity.get("model_revision")
        and metrics_finite
        and artifact_valid
        and deterministic
        and cache_valid
    )
    result = {
        "valid": valid,
        "promotion_passed": receipt.get("promotion", {}).get("passed") is True,
        "metrics_finite": metrics_finite,
        "artifact_valid": artifact_valid,
        "deterministic_controls": deterministic,
        "teacher_cache_valid": cache_valid,
    }
    if not valid:
        raise ValueError(f"quality receipt is not valid terminal science: {result}")
    return result


def validate_artifact_identity(
    reader: QwenBase3ArtifactReader,
    artifact: Path,
    pack_receipt: dict[str, Any],
) -> dict[str, Any]:
    expected_path = Path(pack_receipt.get("artifact", ""))
    identity = reader.manifest.get("identity", {})
    extra = reader.manifest.get("extra", {})
    ledger = reader.manifest.get("ledger", {})
    actual_manifest_sha256 = file_sha256(reader.manifest_path)
    valid = (
        artifact.resolve() == expected_path.resolve()
        and actual_manifest_sha256 == pack_receipt.get("code_manifest_sha256")
        and identity.get("model_revision") == pack_receipt.get("identity", {}).get("model_revision")
        and identity.get("index_sha256") == pack_receipt.get("identity", {}).get("index_sha256")
        and reader.manifest.get("source_sha256") == pack_receipt.get("code_source_sha256")
        and extra.get("scale_source_sha256") == pack_receipt.get("scale_source_sha256")
        and ledger.get("trits") == pack_receipt.get("code_ledger", {}).get("trits")
    )
    result = {
        "valid": valid,
        "artifact": str(artifact),
        "manifest_sha256": actual_manifest_sha256,
        "code_source_sha256": reader.manifest.get("source_sha256"),
        "scale_source_sha256": extra.get("scale_source_sha256"),
        "model_revision": identity.get("model_revision"),
        "index_sha256": identity.get("index_sha256"),
        "trits": ledger.get("trits"),
    }
    if not valid:
        raise ValueError(f"quality artifact identity differs from full-pack receipt: {result}")
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[float, int]:
    """Return summed next-token NLL and token count."""
    shifted_logits = logits[:, :-1].float()
    shifted_labels = input_ids[:, 1:].to(shifted_logits.device)
    losses = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="sum",
    )
    return float(losses.item()), int(shifted_labels.numel())


def teacher_student_kl(teacher: torch.Tensor, student: torch.Tensor) -> tuple[float, int]:
    """Return summed forward KL over causal positions and position count."""
    teacher_logits = teacher[:, :-1].float()
    student_logits = student[:, :-1].float()
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher/student logit shapes differ")
    teacher_logp = torch.log_softmax(teacher_logits, dim=-1)
    student_logp = torch.log_softmax(student_logits, dim=-1)
    divergence = torch.sum(teacher_logp.exp() * (teacher_logp - student_logp), dim=-1)
    return float(divergence.sum().item()), int(divergence.numel())


@torch.inference_mode()
def evaluate_likelihoods(model, tokenizer, texts: tuple[str, ...], device: str, teachers=None):
    rows = []
    cached = []
    nll_sum = 0.0
    token_count = 0
    kl_sum = 0.0
    position_count = 0
    for index, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        logits = model(**inputs, use_cache=False).logits
        nll, tokens = causal_nll(logits, inputs["input_ids"])
        row = {"text": text, "tokens": tokens, "nll": nll, "mean_nll": nll / max(tokens, 1)}
        nll_sum += nll
        token_count += tokens
        if teachers is None:
            cached.append(logits.to(device="cpu", dtype=torch.bfloat16))
        else:
            teacher = teachers[index].to(device=logits.device, dtype=logits.dtype)
            kl, positions = teacher_student_kl(teacher, logits)
            row.update({"teacher_kl": kl, "mean_teacher_kl": kl / max(positions, 1)})
            kl_sum += kl
            position_count += positions
        rows.append(row)
        del logits
    result = {
        "rows": rows,
        "tokens": token_count,
        "nll": nll_sum,
        "mean_nll": nll_sum / max(token_count, 1),
        "perplexity": math.exp(min(nll_sum / max(token_count, 1), 80.0)),
    }
    if teachers is not None:
        result.update({
            "teacher_kl": kl_sum,
            "teacher_kl_positions": position_count,
            "mean_teacher_kl": kl_sum / max(position_count, 1),
        })
    return result, cached


@torch.inference_mode()
def generate_controls(model, tokenizer, prompts: tuple[str, ...], device: str, max_new_tokens: int):
    rows = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        completion_ids = generated[0, inputs["input_ids"].shape[1] :]
        rows.append({
            "prompt": prompt,
            "new_tokens": int(completion_ids.numel()),
            "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
        })
    return rows


def resolve_parameter_map(source_names, named_parameters) -> dict[str, str]:
    """Resolve checkpoint names to runtime names without fuzzy matching."""
    resolved: dict[str, str] = {}
    claimed: dict[str, str] = {}
    checkpoint_prefix = "model.language_model."
    runtime_prefix = "model."
    for source_name in source_names:
        candidates = [source_name]
        if source_name.startswith(checkpoint_prefix):
            candidates.append(runtime_prefix + source_name[len(checkpoint_prefix) :])
        matches = [candidate for candidate in candidates if candidate in named_parameters]
        if len(matches) != 1:
            raise ValueError(
                f"artifact tensor {source_name!r} resolved to {matches}; expected exactly one runtime parameter"
            )
        runtime_name = matches[0]
        if runtime_name in claimed:
            raise ValueError(
                f"artifact tensors {claimed[runtime_name]!r} and {source_name!r} map to {runtime_name!r}"
            )
        resolved[source_name] = runtime_name
        claimed[runtime_name] = source_name
    return resolved


def validate_preregistration(preregistration: dict[str, Any], args) -> None:
    quality = preregistration.get("behavioral_quality", {})
    if (
        preregistration.get("schema") != "clark_qwen38_evaluation_preregistration_v1"
        or tuple(quality.get("held_out_texts", ())) != HELD_OUT_TEXTS
        or tuple(quality.get("generation_prompts", ())) != GENERATION_PROMPTS
        or quality.get("maximum_new_tokens") != args.max_new_tokens
        or quality.get("maximum_perplexity_ratio") != args.max_ppl_ratio
        or quality.get("maximum_mean_teacher_kl") != args.max_mean_teacher_kl
        or quality.get("require_finite_metrics") is not True
        or quality.get("require_nonempty_deterministic_generations") is not True
    ):
        raise ValueError("behavioral quality arguments differ from preregistration")


def reconstruct_codes(codes, scales, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    code_tensor = torch.from_numpy(codes)
    scale_tensor = torch.from_numpy(scales)
    columns = code_tensor.shape[1]
    padding = (-columns) % group_size
    if padding:
        code_tensor = torch.nn.functional.pad(code_tensor, (0, padding))
    return (
        code_tensor.reshape(code_tensor.shape[0], -1, group_size).to(dtype)
        * scale_tensor.to(dtype).unsqueeze(-1)
    ).reshape(code_tensor.shape[0], -1)[:, :columns].contiguous()


@torch.inference_mode()
def apply_artifact(
    model,
    reader: QwenBase3ArtifactReader,
    tensor_names=None,
    *,
    workers: int = 14,
    max_in_flight: int = 28,
) -> dict[str, int]:
    if workers <= 0 or max_in_flight < workers:
        raise ValueError("decode workers must be positive and max-in-flight must be at least workers")
    try:
        named_parameters = dict(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - compatibility with older torch
        named_parameters = dict(model.named_parameters())
    selected_names = tuple(reader.tensor_names if tensor_names is None else tensor_names)
    unknown = sorted(set(selected_names) - set(reader.tensor_names))
    if unknown:
        raise ValueError(f"requested tensors absent from artifact: {unknown[:8]}")
    parameter_map = resolve_parameter_map(selected_names, named_parameters)

    jobs = []
    for name in selected_names:
        parameter = named_parameters[parameter_map[name]]
        source = reader.sources[name]
        if list(parameter.shape) != source["shape"]:
            raise ValueError(f"shape mismatch for {name}: model={list(parameter.shape)} artifact={source['shape']}")
        jobs.extend((name, parameter, record) for record in reader.records_by_tensor[name])

    batches = parameters = 0
    pending: dict[int, tuple[Future, str, torch.nn.Parameter, dict]] = {}

    def commit_oldest() -> None:
        nonlocal batches, parameters
        sequence = min(pending)
        future, name, parameter, record = pending.pop(sequence)
        decoded_index, codes, scales = future.result()
        if decoded_index != int(record["index"]):
            raise ValueError(f"parallel decoder returned chunk {decoded_index} for {record['index']}")
        descriptor = record["descriptor"]
        row_start = int(descriptor["row_start"])
        row_stop = int(descriptor["row_stop"])
        block = reconstruct_codes(codes, scales, reader.group_size, torch.float16)
        if list(block.shape) != [row_stop - row_start, parameter.shape[1]]:
            raise ValueError(f"reconstructed batch shape mismatch for {name}")
        parameter[row_start:row_stop].copy_(block.to(device=parameter.device, dtype=parameter.dtype))
        batches += 1
        parameters += block.numel()

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        for sequence, (name, parameter, record) in enumerate(jobs):
            pending[sequence] = (
                executor.submit(decode_artifact_record, reader.root, reader.group_size, record),
                name,
                parameter,
                record,
            )
            while len(pending) >= max_in_flight:
                commit_oldest()
        while pending:
            commit_oldest()

    return {"tensors": len(selected_names), "batches": batches, "parameters": parameters}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"),
    )
    parser.add_argument("--pack-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-full.json"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/qwen38-base3-quality.json"))
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-quality/teacher-logits.pt"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.5)
    parser.add_argument("--max-mean-teacher-kl", type=float, default=1.0)
    parser.add_argument("--decode-workers", type=int, default=14)
    parser.add_argument("--max-in-flight", type=int, default=28)
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path(__file__).resolve().parent / "evaluation_preregistration.json",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_ppl_ratio <= 1 or args.max_mean_teacher_kl <= 0:
        raise ValueError("quality thresholds and generation length must be positive")

    preregistration_content = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_content)
    validate_preregistration(preregistration, args)
    pack_receipt = json.loads(args.pack_receipt.read_text())
    if not pack_receipt.get("promotion", {}).get("passed"):
        raise ValueError("full-pack receipt has not passed promotion")
    if preregistration.get("model_revision") != pack_receipt["identity"]["model_revision"]:
        raise ValueError("behavioral preregistration model revision mismatch")
    reader = QwenBase3ArtifactReader(args.artifact)
    artifact_integrity = validate_artifact_identity(reader, args.artifact, pack_receipt)

    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    torch.manual_seed(20260815)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
        "local_files_only": True,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    model_class = "AutoModelForCausalLM"
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    except ValueError:
        model_class = "AutoModelForImageTextToText"
        model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **load_kwargs)
    model.eval()
    load_seconds = time.time() - started

    baseline_started = time.time()
    bf16_likelihood, teacher_logits = evaluate_likelihoods(model, tokenizer, HELD_OUT_TEXTS, "cuda")
    bf16_generations = generate_controls(model, tokenizer, GENERATION_PROMPTS, "cuda", args.max_new_tokens)
    bf16_generations_repeat = generate_controls(model, tokenizer, GENERATION_PROMPTS, "cuda", args.max_new_tokens)
    bf16_generations_deterministic = bf16_generations_repeat == bf16_generations
    teacher_cache_sha256 = atomic_torch_save(args.teacher_cache, {
        "schema": "clark_qwen38_bf16_teacher_logits_v1",
        "model_revision": pack_receipt["identity"]["model_revision"],
        "texts": HELD_OUT_TEXTS,
        "logits": teacher_logits,
    })
    baseline_seconds = time.time() - baseline_started

    apply_started = time.time()
    applied = apply_artifact(
        model,
        reader,
        workers=args.decode_workers,
        max_in_flight=args.max_in_flight,
    )
    apply_seconds = time.time() - apply_started
    expected_parameters = int(pack_receipt["code_ledger"]["trits"])
    if applied["parameters"] != expected_parameters:
        raise ValueError(f"applied {applied['parameters']} parameters, expected {expected_parameters}")

    ternary_started = time.time()
    ternary_likelihood, _ = evaluate_likelihoods(
        model, tokenizer, HELD_OUT_TEXTS, "cuda", teachers=teacher_logits
    )
    ternary_generations = generate_controls(model, tokenizer, GENERATION_PROMPTS, "cuda", args.max_new_tokens)
    ternary_generations_repeat = generate_controls(model, tokenizer, GENERATION_PROMPTS, "cuda", args.max_new_tokens)
    ternary_generations_deterministic = ternary_generations_repeat == ternary_generations
    ternary_seconds = time.time() - ternary_started
    ppl_ratio = ternary_likelihood["perplexity"] / max(bf16_likelihood["perplexity"], 1e-12)
    finite = all(math.isfinite(value) for value in (
        bf16_likelihood["mean_nll"],
        bf16_likelihood["perplexity"],
        ternary_likelihood["mean_nll"],
        ternary_likelihood["perplexity"],
        ternary_likelihood["mean_teacher_kl"],
        ppl_ratio,
    ))
    nonempty = all(
        row["completion"].strip()
        for row in bf16_generations + ternary_generations
    )
    passed = (
        finite
        and nonempty
        and bf16_generations_deterministic
        and ternary_generations_deterministic
        and ppl_ratio <= args.max_ppl_ratio
        and ternary_likelihood["mean_teacher_kl"] <= args.max_mean_teacher_kl
    )
    receipt = {
        "schema": "clark_qwen38_base3_behavioral_quality_v1",
        "hypothesis": (
            "Data-free optimal-MSE group-128 ternarization preserves bounded held-out likelihood and "
            "teacher-distribution drift relative to the exact BF16 parent."
        ),
        "model_dir": str(args.model_dir),
        "model_revision": pack_receipt["identity"]["model_revision"],
        "artifact": str(args.artifact),
        "code_manifest_sha256": pack_receipt["code_manifest_sha256"],
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(preregistration_content).hexdigest(),
        },
        "model_class": model_class,
        "artifact_integrity": artifact_integrity,
        "teacher_cache": {"path": str(args.teacher_cache), "sha256": teacher_cache_sha256},
        "bf16": {"likelihood": bf16_likelihood, "generations": bf16_generations},
        "ternary": {"likelihood": ternary_likelihood, "generations": ternary_generations},
        "comparison": {
            "perplexity_ratio": ppl_ratio,
            "mean_teacher_kl": ternary_likelihood["mean_teacher_kl"],
            "all_bf16_and_ternary_completions_nonempty": nonempty,
            "bf16_generations_deterministic": bf16_generations_deterministic,
            "ternary_generations_deterministic": ternary_generations_deterministic,
            "bf16_generation_sha256": generation_sha256(bf16_generations),
            "bf16_generation_repeat_sha256": generation_sha256(bf16_generations_repeat),
            "ternary_generation_sha256": generation_sha256(ternary_generations),
            "ternary_generation_repeat_sha256": generation_sha256(ternary_generations_repeat),
        },
        "applied": applied,
        "thresholds": {
            "max_perplexity_ratio": args.max_ppl_ratio,
            "max_mean_teacher_kl": args.max_mean_teacher_kl,
            "require_finite_metrics": True,
            "require_nonempty_completions": True,
        },
        "timing_seconds": {
            "model_load": load_seconds,
            "bf16_control": baseline_seconds,
            "artifact_apply": apply_seconds,
            "ternary_evaluation": ternary_seconds,
            "total": time.time() - started,
        },
        "parallel_decode": {
            "workers": args.decode_workers,
            "max_in_flight": args.max_in_flight,
            "ordered_parent_application": True,
        },
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "device": torch.cuda.get_device_name(0),
        "promotion": {
            "passed": passed,
            "scope": "small preregistered PTQ diagnostic; downstream benchmark promotion remains separate",
        },
        "stop_condition": "Complete both arms or stop on integrity, mapping, OOM, non-finite, or runtime failure.",
    }
    atomic_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
