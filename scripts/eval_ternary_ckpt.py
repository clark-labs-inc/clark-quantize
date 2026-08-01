"""Run a preregistered, fail-closed GSM8K/MMLU Gate B evaluation.

The model, immutable revision, evaluation arm, and sealed sample manifest are
required. A QAT checkpoint must replace every base-model tensor exactly; this
script never leaves missing weights at the base model and reports a score.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "large_scale_ternary"))

from clark_quantize.ternary_kd import fused_absmean_dequant
from gate_b_protocol import (
    GSM8K_SHOTS,
    checkpoint_assignment_plan,
    digest_json,
    environment_provenance,
    prompt_contract_sha256,
    select_and_verify_rows,
    select_and_verify_support_rows,
    sha256_file,
    validate_arm,
    verify_manifest,
    wilson_interval,
)
from gate_b_model_arms import apply_ptq_ternary_dense


def preflight_gpu(min_free_gb: float, device: str) -> None:
    if device == "cpu":
        return
    if not device.startswith("cuda"):
        raise SystemExit("Gate B currently supports cpu or cuda devices")
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 2**30, total / 2**30
    print(f"GPU free {free_gb:.1f} / {total_gb:.1f} GB")
    if free_gb < min_free_gb:
        raise SystemExit(
            f"ABORT: only {free_gb:.1f} GB free, need >= {min_free_gb:.1f}. "
            "A training run is likely using the card; not touching its PID."
        )
    fraction = min((free_gb - 0.75) / total_gb, 0.99)
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f"capped eval process to {fraction * total_gb:.1f} GB")


def load_deployed_model(
    base: str,
    revision: str,
    checkpoint: Path | None,
    device: str,
    *,
    arm: str,
    dtype=torch.bfloat16,
):
    """Construct the exact dense form of a QAT checkpoint or the BF16 control."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load_plain():
        common = {
            "revision": revision,
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        try:
            return AutoModelForCausalLM.from_pretrained(base, **common)
        except ValueError:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(base, **common)

    tokenizer = AutoTokenizer.from_pretrained(
        base, revision=revision, trust_remote_code=True
    )
    model = load_plain()
    arm_metadata = {"method": "exact_bf16_parent"}
    if arm == "ptq_ternary":
        arm_metadata = apply_ptq_ternary_dense(model, dtype=dtype)
    if checkpoint is not None:
        print(f"loading checkpoint {checkpoint} (CPU) ...")
        payload = torch.load(checkpoint, map_location="cpu")
        raw_state = payload.get("model", payload)
        state = {
            key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key: value
            for key, value in raw_state.items()
        }
        own = model.state_dict()
        plan = checkpoint_assignment_plan(state, own)
        assignment, dequantized = {}, 0
        for target, source in plan.items():
            value = state[source]
            if source.endswith("weight_master"):
                assignment[target] = fused_absmean_dequant(
                    value.float(), out_dtype=own[target].dtype
                ).reshape(own[target].shape)
                dequantized += 1
            else:
                assignment[target] = value.to(dtype=own[target].dtype)
        model.load_state_dict(assignment, strict=True)
        print(
            f"checkpoint step {payload.get('step', '?')}: dequantized "
            f"{dequantized} ternary tensors and applied all "
            f"{len(assignment)} model tensors"
        )
        arm_metadata = {
            "method": "qat_ternary_checkpoint_dense_dequant_v1",
            "dequantized_tensors": dequantized,
            "checkpoint_step": payload.get("step", None),
        }
        del payload, raw_state, state, assignment, plan

    core = getattr(model, "model", model)
    for attribute in ("visual", "vision_model", "vision_tower"):
        if getattr(core, attribute, None) is not None:
            setattr(core, attribute, None)
            print(f"dropped {attribute} for text-only eval")
    if getattr(model, "mtp", None) is not None:
        model.mtp = None
        print("dropped mtp head for text-only eval")
    model.config.use_cache = True
    model.eval().to(device)
    return model, tokenizer, arm_metadata


def summarize_task(task: str, shots: int, records: list[dict]) -> dict:
    correct = sum(record["correct"] for record in records)
    latencies = sorted(record["latency_ms"] for record in records)
    return {
        "task": task,
        "n": len(records),
        "shots": shots,
        "accuracy": correct / len(records),
        "accuracy_wilson_95_ci": wilson_interval(correct, len(records)),
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[
                min(len(latencies) - 1, int(len(latencies) * 0.95))
            ],
        },
        "samples": records,
    }


@torch.no_grad()
def eval_gsm8k(model, tokenizer, device, selected, shots, max_new):
    preamble = "".join(
        f"Question: {question}\nAnswer: {answer}\n\n"
        for question, answer in GSM8K_SHOTS[:shots]
    )
    records, started = [], time.time()
    for index, (row, sample) in enumerate(selected):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        prompt = preamble + f"Question: {row['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        item_started = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        latency_ms = (time.perf_counter() - item_started) * 1_000
        text = tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).split("Question:")[0]
        numbers = re.findall(r"-?\$?\d[\d,]*\.?\d*", text)
        prediction = (
            numbers[-1].replace(",", "").replace("$", "").rstrip(".")
            if numbers
            else None
        )
        try:
            correct = (
                prediction is not None
                and abs(float(prediction) - float(gold)) < 1e-4
            )
        except ValueError:
            correct = prediction == gold
        records.append(
            {
                "sample_id": sample["sample_id"],
                "correct": bool(correct),
                "prediction": prediction,
                "gold": gold,
                "output": text,
                "latency_ms": latency_ms,
                "input_tokens": int(inputs.input_ids.shape[1]),
                "output_tokens": int(
                    output.shape[1] - inputs.input_ids.shape[1]
                ),
            }
        )
        if (index + 1) % 25 == 0:
            running = sum(record["correct"] for record in records)
            print(
                f"  gsm8k {index + 1}/{len(selected)} "
                f"acc={running / (index + 1):.3f} "
                f"({(index + 1) / (time.time() - started):.2f} q/s)",
                flush=True,
            )
    return summarize_task("gsm8k", shots, records)


@torch.no_grad()
def eval_mmlu(model, tokenizer, device, selected, support, shots):
    letters = ["A", "B", "C", "D"]
    letter_ids = [
        tokenizer(" " + letter, add_special_tokens=False).input_ids[-1]
        for letter in letters
    ]

    def format_row(row, with_answer):
        options = "\n".join(
            f"{letters[index]}. {choice}"
            for index, choice in enumerate(row["choices"])
        )
        text = f"{row['question'].strip()}\n{options}\nAnswer:"
        return (
            text + f" {letters[row['answer']]}\n\n" if with_answer else text
        )

    records, started = [], time.time()
    for index, (row, sample) in enumerate(selected):
        subject = row["subject"]
        examples = support[subject][:shots]
        heading = (
            "The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n"
        )
        prompt = (
            heading
            + "".join(format_row(example, True) for example in examples)
            + format_row(row, False)
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        item_started = time.perf_counter()
        logits = model(**inputs).logits[0, -1]
        choices = torch.stack([logits[token_id] for token_id in letter_ids])
        latency_ms = (time.perf_counter() - item_started) * 1_000
        prediction = int(choices.argmax())
        records.append(
            {
                "sample_id": sample["sample_id"],
                "correct": prediction == row["answer"],
                "prediction": letters[prediction],
                "gold": letters[row["answer"]],
                "choice_logits": [
                    float(value) for value in choices.float().cpu()
                ],
                "latency_ms": latency_ms,
                "input_tokens": int(inputs.input_ids.shape[1]),
            }
        )
        if (index + 1) % 50 == 0:
            running = sum(record["correct"] for record in records)
            print(
                f"  mmlu {index + 1}/{len(selected)} "
                f"acc={running / (index + 1):.3f} "
                f"({(index + 1) / (time.time() - started):.1f} q/s)",
                flush=True,
            )
    return summarize_task("mmlu", shots, records)


def load_manifest_task(task_spec):
    from datasets import load_dataset

    dataset = load_dataset(
        task_spec["dataset"],
        task_spec["config"],
        split=task_spec["split"],
        revision=task_spec["revision"],
    )
    if dataset._fingerprint != task_spec["fingerprint"]:
        raise ValueError(
            f"{task_spec['task']} fingerprint drift: "
            f"{dataset._fingerprint} != {task_spec['fingerprint']}"
        )
    selected = select_and_verify_rows(dataset, task_spec)
    support = None
    if "support" in task_spec:
        spec = task_spec["support"]
        support_dataset = load_dataset(
            spec["dataset"],
            spec["config"],
            split=spec["split"],
            revision=spec["revision"],
        )
        if support_dataset._fingerprint != spec["fingerprint"]:
            raise ValueError("support dataset fingerprint drift")
        support = select_and_verify_support_rows(support_dataset, spec)
    return selected, support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=["bf16", "qat_ternary", "ptq_ternary", "int4"],
    )
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-free-gb", type=float, default=10.5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    validate_arm(args.arm, args.ckpt)
    if args.arm == "int4":
        raise SystemExit(
            "int4 is preregistered but its loader is not implemented; "
            "refusing to score a mislabeled arm"
        )
    manifest = json.loads(args.manifest.read_text())
    verify_manifest(manifest)
    if manifest.get("prompt_contract_sha256") != prompt_contract_sha256():
        raise SystemExit("prompt contract does not match the sealed manifest")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    preflight_gpu(args.min_free_gb, args.device)
    started = datetime.now(timezone.utc)
    if args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer, arm_metadata = load_deployed_model(
        args.model,
        args.model_revision,
        args.ckpt,
        args.device,
        arm=args.arm,
    )

    report = {
        "schema_version": 1,
        "arm": args.arm,
        "arm_transform": arm_metadata,
        "model": {
            "id": args.model,
            "revision": args.model_revision,
            "config_sha256": digest_json(model.config.to_dict()),
            "tokenizer_vocab_sha256": digest_json(tokenizer.get_vocab()),
        },
        "checkpoint": (
            {"path": str(args.ckpt), "sha256": sha256_file(args.ckpt)}
            if args.ckpt
            else None
        ),
        "manifest": {
            "path": str(args.manifest),
            "file_sha256": sha256_file(args.manifest),
            "content_sha256": manifest["content_sha256"],
        },
        "environment": environment_provenance(REPO),
        "started_at": started.isoformat(),
        "results": [],
    }
    for task_spec in manifest["tasks"]:
        task = task_spec["task"]
        print(f"\n=== {task} ===", flush=True)
        selected, support = load_manifest_task(task_spec)
        if task == "gsm8k":
            result = eval_gsm8k(
                model,
                tokenizer,
                args.device,
                selected,
                task_spec["shots"],
                args.max_new_tokens,
            )
        elif task == "mmlu":
            result = eval_mmlu(
                model,
                tokenizer,
                args.device,
                selected,
                support,
                task_spec["shots"],
            )
        else:
            raise SystemExit(f"unknown manifest task {task!r}")
        print(
            f"  -> {task} accuracy = {result['accuracy']:.4f} "
            f"(n={result['n']})"
        )
        report["results"].append(result)

    finished = datetime.now(timezone.utc)
    report["finished_at"] = finished.isoformat()
    report["elapsed_seconds"] = (finished - started).total_seconds()
    report["resource_measurements"] = {
        "peak_accelerator_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if args.device != "cpu"
            else None
        ),
        "energy_joules": None,
        "energy_status": "requires external power meter or NVML sampler",
    }
    report["report_content_sha256"] = digest_json(report)
    print(
        "\n"
        + json.dumps(
            [
                {key: value for key, value in result.items() if key != "samples"}
                for result in report["results"]
            ],
            indent=2,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
