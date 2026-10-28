#!/usr/bin/env python3
"""Evaluate frozen capability rows for BF16, base-3 PTQ, or a QAT patch."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "large_scale_ternary"))
sys.path.insert(0, str(REPO / "experiments" / "qwen_agentworld"))

from gate_b_protocol import (  # noqa: E402
    GSM8K_SHOTS,
    digest_json,
    environment_provenance,
    select_and_verify_rows,
    select_and_verify_support_rows,
    sha256_file,
    verify_manifest,
    wilson_interval,
)
from weight_patch import apply_weight_patch  # noqa: E402


def extract_last_boxed(text: str) -> str | None:
    starts = [match.start() for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    open_brace = text.find("{", start)
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index]
    return None


def normalize_math_answer(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip("$").strip()
    replacements = (
        (r"\left", ""), (r"\right", ""), (r"\!", ""), (r"\,", ""),
        (r"\;", ""), (r"\ ", ""), (" ", ""), ("\n", ""),
        ("−", "-"), (r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"),
    )
    for source, target in replacements:
        value = value.replace(source, target)
    value = value.replace(",", "") if re.fullmatch(r"-?[0-9,]+(?:\.[0-9]+)?", value) else value
    if value.endswith(".0") and re.fullmatch(r"-?\d+\.0", value):
        value = value[:-2]
    return value


def chat_prompt(tokenizer: Any, text: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def summarize(task: str, shots: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(record["correct"] for record in records)
    return {
        "task": task,
        "n": len(records),
        "shots": shots,
        "correct": correct,
        "accuracy": correct / len(records),
        "accuracy_wilson_95_ci": wilson_interval(correct, len(records)),
        "samples": records,
    }


def load_model(args: argparse.Namespace):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    errors = []
    model = None
    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                args.model,
                local_files_only=True,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
    if model is None:
        raise RuntimeError("could not load model: " + "; ".join(errors))

    core = getattr(model, "model", model)
    for attribute in ("visual", "vision_model", "vision_tower"):
        if getattr(core, attribute, None) is not None:
            setattr(core, attribute, None)
    if getattr(model, "mtp", None) is not None:
        model.mtp = None
    gc.collect()
    torch.cuda.empty_cache()

    transform: dict[str, Any] = {"method": "exact_bf16_parent"}
    if args.arm == "base3_ptq":
        from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader
        from experiments.qwen38_27b_spot.behavioral_quality_gate import apply_artifact

        reader = QwenBase3ArtifactReader(args.artifact)
        applied = apply_artifact(model, reader, workers=args.decode_workers, max_in_flight=args.max_in_flight)
        transform = {"method": "completed_base3_artifact", "artifact": str(args.artifact), **applied}
    elif args.arm == "qat_patch":
        applied = apply_weight_patch(model, args.patch)
        if applied["missing_count"] or not applied["applied_count"]:
            raise RuntimeError(f"QAT patch did not apply exactly: {applied}")
        transform = {
            "method": "hard_ternary_qat_weight_patch",
            "patch": str(args.patch),
            "patch_sha256": sha256_file(args.patch),
            "applied_count": applied["applied_count"],
            "applied_parameters": sum(row["params"] for row in applied["applied"]),
        }
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, transform


def generate_batch(model: Any, tokenizer: Any, texts: list[str], max_new: int) -> list[tuple[str, int, int, float]]:
    prompts = [chat_prompt(tokenizer, text) for text in texts]
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(next(model.parameters()).device)
    tokenizer.padding_side = previous_padding_side
    started = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = (time.perf_counter() - started) * 1000
    prompt_width = int(inputs.input_ids.shape[1])
    prompt_lengths = inputs.attention_mask.sum(dim=1).tolist()
    return [
        (
            tokenizer.decode(output[index, prompt_width:], skip_special_tokens=True),
            int(prompt_lengths[index]),
            int(output.shape[1] - prompt_width),
            elapsed,
        )
        for index in range(len(texts))
    ]


@torch.inference_mode()
def eval_math(model: Any, tokenizer: Any, selected: list[Any], max_new: int, batch_size: int) -> dict[str, Any]:
    records = []
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        outputs = generate_batch(model, tokenizer, [row["problem"] + "\nReturn only the final answer enclosed in \\boxed{}; do not show work." for row, _ in batch], max_new)
        for (row, sample), (output, input_tokens, output_tokens, latency) in zip(batch, outputs):
            prediction = normalize_math_answer(extract_last_boxed(output))
            gold = normalize_math_answer(row["answer"])
            records.append({"sample_id": sample["sample_id"], "correct": prediction == gold, "prediction": prediction, "gold": gold, "output": output, "input_tokens": input_tokens, "output_tokens": output_tokens, "batch_latency_ms": latency})
        print(f"math500 {len(records)}/{len(selected)} acc={sum(r['correct'] for r in records)/len(records):.3f}", flush=True)
        print(json.dumps({key: records[-1][key] for key in ("sample_id", "prediction", "gold", "correct", "output_tokens") } | {"output_tail": records[-1]["output"][-400:]}, ensure_ascii=False), flush=True)
    return summarize("math500", 0, records)


@torch.inference_mode()
def eval_gsm8k(model: Any, tokenizer: Any, selected: list[Any], shots: int, max_new: int, batch_size: int) -> dict[str, Any]:
    preamble = "".join(f"Question: {question}\nAnswer: {answer}\n\n" for question, answer in GSM8K_SHOTS[:shots])
    records = []
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        outputs = generate_batch(model, tokenizer, [preamble + f"Question: {row['question']}\nAnswer: Return only the final numeric answer." for row, _ in batch], max_new)
        for (row, sample), (output, input_tokens, output_tokens, latency) in zip(batch, outputs):
            gold = row["answer"].split("####")[-1].strip().replace(",", "")
            numbers = re.findall(r"-?\$?\d[\d,]*\.?\d*", output)
            prediction = numbers[-1].replace(",", "").replace("$", "").rstrip(".") if numbers else None
            try:
                correct = prediction is not None and abs(float(prediction) - float(gold)) < 1e-4
            except ValueError:
                correct = prediction == gold
            records.append({"sample_id": sample["sample_id"], "correct": bool(correct), "prediction": prediction, "gold": gold, "output": output, "input_tokens": input_tokens, "output_tokens": output_tokens, "batch_latency_ms": latency})
        print(f"gsm8k {len(records)}/{len(selected)} acc={sum(r['correct'] for r in records)/len(records):.3f}", flush=True)
        print(json.dumps({key: records[-1][key] for key in ("sample_id", "prediction", "gold", "correct", "output_tokens") } | {"output_tail": records[-1]["output"][-400:]}, ensure_ascii=False), flush=True)
    return summarize("gsm8k", shots, records)


@torch.inference_mode()
def eval_mmlu(model: Any, tokenizer: Any, selected: list[Any], support: dict[str, list[Any]], shots: int) -> dict[str, Any]:
    letters = ["A", "B", "C", "D"]
    letter_ids = [tokenizer(" " + letter, add_special_tokens=False).input_ids[-1] for letter in letters]

    def format_row(row: Any, answer: bool) -> str:
        choices = "\n".join(f"{letters[index]}. {choice}" for index, choice in enumerate(row["choices"]))
        result = f"{row['question'].strip()}\n{choices}\nAnswer:"
        return result + (f" {letters[row['answer']]}\n\n" if answer else "")

    records = []
    for position, (row, sample) in enumerate(selected, 1):
        subject = row["subject"]
        text = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        text += "".join(format_row(example, True) for example in support[subject][:shots]) + format_row(row, False)
        prompt = chat_prompt(tokenizer, text)
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        started = time.perf_counter()
        logits = model(**inputs).logits[0, -1]
        choice_logits = torch.stack([logits[token] for token in letter_ids])
        prediction = int(choice_logits.argmax())
        records.append({"sample_id": sample["sample_id"], "correct": prediction == row["answer"], "prediction": letters[prediction], "gold": letters[row["answer"]], "choice_logits": [float(v) for v in choice_logits.float().cpu()], "input_tokens": int(inputs.input_ids.shape[1]), "latency_ms": (time.perf_counter() - started) * 1000})
        print(f"mmlu {position}/{len(selected)} acc={sum(r['correct'] for r in records)/position:.3f}", flush=True)
    return summarize("mmlu", shots, records)


def load_rows(spec: dict[str, Any]):
    from datasets import load_dataset

    dataset = load_dataset(spec["dataset"], spec.get("config"), split=spec["split"], revision=spec["revision"])
    if dataset._fingerprint != spec["fingerprint"]:
        raise RuntimeError(f"{spec['task']} fingerprint drift")
    selected = select_and_verify_rows(dataset, spec)
    support = None
    if "support" in spec:
        support_spec = spec["support"]
        support_dataset = load_dataset(support_spec["dataset"], support_spec.get("config"), split=support_spec["split"], revision=support_spec["revision"])
        if support_dataset._fingerprint != support_spec["fingerprint"]:
            raise RuntimeError("MMLU support fingerprint drift")
        support = select_and_verify_support_rows(support_dataset, support_spec)
    return selected, support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--arm", choices=("bf16", "base3_ptq", "qat_patch"), required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=12)
    parser.add_argument("--max-in-flight", type=int, default=24)
    args = parser.parse_args()
    if args.arm == "base3_ptq" and args.artifact is None:
        parser.error("--base3-ptq requires --artifact")
    if args.arm == "qat_patch" and args.patch is None:
        parser.error("--qat-patch requires --patch")

    manifest = json.loads(args.manifest.read_text())
    verify_manifest(manifest)
    if digest_json(manifest["prompt_contract"]) != manifest["prompt_contract_sha256"]:
        raise RuntimeError("prompt contract hash mismatch")
    generation_contract = manifest["prompt_contract"]["generation"]
    if args.max_new_tokens != generation_contract["max_new_tokens"] or args.generation_batch_size != generation_contract["batch_size"]:
        raise RuntimeError("generation geometry differs from frozen prompt contract")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()
    started = datetime.now(timezone.utc)
    model, tokenizer, transform = load_model(args)
    report = {
        "schema": "qwen38_capability_forgetting_eval_v1",
        "arm": args.arm,
        "transform": transform,
        "model": {"path": str(args.model), "revision": args.model_revision, "config_sha256": digest_json(model.config.to_dict()), "tokenizer_vocab_sha256": digest_json(tokenizer.get_vocab())},
        "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest), "content_sha256": manifest["content_sha256"]},
        "environment": environment_provenance(REPO),
        "started_at": started.isoformat(),
        "results": [],
    }
    for spec in manifest["tasks"]:
        selected, support = load_rows(spec)
        if spec["task"] == "math500":
            result = eval_math(model, tokenizer, selected, args.max_new_tokens, args.generation_batch_size)
        elif spec["task"] == "gsm8k":
            result = eval_gsm8k(model, tokenizer, selected, spec["shots"], args.max_new_tokens, args.generation_batch_size)
        elif spec["task"] == "mmlu":
            result = eval_mmlu(model, tokenizer, selected, support, spec["shots"])
        else:
            raise RuntimeError(f"unknown task {spec['task']}")
        report["results"].append(result)
        print(json.dumps({"task": result["task"], "n": result["n"], "accuracy": result["accuracy"]}), flush=True)
    finished = datetime.now(timezone.utc)
    report["finished_at"] = finished.isoformat()
    report["elapsed_seconds"] = (finished - started).total_seconds()
    report["peak_accelerator_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    report["report_content_sha256"] = digest_json(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"out": str(args.out), "elapsed_seconds": report["elapsed_seconds"], "peak_accelerator_memory_bytes": report["peak_accelerator_memory_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
