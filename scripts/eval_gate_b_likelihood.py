"""Evaluate causal likelihood and exact teacher KL on a sealed text manifest."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
GATE_DIR = REPO / "experiments" / "large_scale_ternary"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(GATE_DIR))

from eval_ternary_ckpt import load_deployed_model, preflight_gpu
from gate_b_likelihood import aggregate_likelihood, causal_likelihood
from gate_b_protocol import (
    digest_json,
    environment_provenance,
    row_digest,
    sha256_file,
    validate_arm,
    verify_manifest,
)
from prepare_fineweb2_likelihood_manifest import LIKELIHOOD_CONTRACT


def fetch_pinned_rows(task_spec) -> list[tuple[dict, dict]]:
    from datasets import load_dataset

    samples_by_config = defaultdict(list)
    for sample in task_spec["samples"]:
        samples_by_config[sample["config"]].append(sample)
    selected = []
    for position, config in enumerate(sorted(samples_by_config), start=1):
        wanted = {
            int(sample["index"]): sample for sample in samples_by_config[config]
        }
        dataset = load_dataset(
            task_spec["dataset"],
            name=config,
            split=task_spec["split"],
            revision=task_spec["revision"],
            streaming=True,
        )
        found = {}
        for index, row in enumerate(dataset):
            if index in wanted:
                sample = wanted[index]
                actual = row_digest(row)
                if actual != sample["row_sha256"]:
                    raise ValueError(
                        f"dataset drift for {sample['sample_id']}: "
                        f"{actual} != {sample['row_sha256']}"
                    )
                found[index] = (row, sample)
            if len(found) == len(wanted):
                break
        if found.keys() != wanted.keys():
            missing = sorted(wanted.keys() - found.keys())
            raise ValueError(f"{config} is missing frozen rows {missing}")
        selected.extend(found[index] for index in sorted(found))
        if position % 25 == 0:
            print(
                f"verified {position}/{len(samples_by_config)} language configs",
                flush=True,
            )
    return selected


@torch.no_grad()
def evaluate(model, teacher, tokenizer, device, selected, contract):
    records = []
    for index, (row, sample) in enumerate(selected):
        inputs = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=contract["max_tokens"],
            add_special_tokens=True,
        ).to(device)
        if inputs.input_ids.shape[1] < contract["minimum_tokens"]:
            raise ValueError(
                f"{sample['sample_id']} has only "
                f"{inputs.input_ids.shape[1]} tokens"
            )
        started = time.perf_counter()
        student_logits = model(**inputs).logits
        teacher_logits = (
            teacher(**inputs).logits if teacher is not None else student_logits
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000
        metrics = causal_likelihood(
            student_logits,
            inputs.input_ids,
            teacher_logits=teacher_logits,
        )
        records.append(
            {
                "sample_id": sample["sample_id"],
                "stratum": sample["stratum"],
                "latency_ms": elapsed_ms,
                **metrics,
            }
        )
        del student_logits, teacher_logits
        if (index + 1) % 25 == 0:
            aggregate = aggregate_likelihood(records)
            print(
                f"{index + 1}/{len(selected)} "
                f"ppl={aggregate['micro_perplexity']:.3f} "
                f"kl={aggregate['micro_mean_teacher_kl']:.5f}",
                flush=True,
            )
    return records, aggregate_likelihood(records)


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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-free-gb", type=float, default=24.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    validate_arm(args.arm, args.ckpt)
    if args.arm == "int4":
        raise SystemExit(
            "int4 loader is not implemented; refusing a mislabeled arm"
        )
    manifest = json.loads(args.manifest.read_text())
    verify_manifest(manifest)
    if len(manifest["tasks"]) != 1:
        raise SystemExit("likelihood manifest must contain exactly one task")
    task = manifest["tasks"][0]
    if task["task"] != "fineweb2_likelihood":
        raise SystemExit("manifest is not a FineWeb2 likelihood gate")
    contract_hash = digest_json(LIKELIHOOD_CONTRACT)
    if manifest.get("prompt_contract_sha256") != contract_hash:
        raise SystemExit("likelihood contract hash mismatch")
    if task["likelihood_contract"] != LIKELIHOOD_CONTRACT:
        raise SystemExit("likelihood contract body mismatch")

    preflight_gpu(args.min_free_gb, args.device)
    if args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    selected = fetch_pinned_rows(task)
    started = datetime.now(timezone.utc)
    teacher = None
    if args.arm != "bf16":
        print("loading exact BF16 parent for teacher KL")
        teacher, teacher_tokenizer, _ = load_deployed_model(
            args.model,
            args.model_revision,
            None,
            args.device,
            arm="bf16",
        )
    else:
        teacher_tokenizer = None
    model, tokenizer, arm_metadata = load_deployed_model(
        args.model,
        args.model_revision,
        args.ckpt,
        args.device,
        arm=args.arm,
    )
    if teacher_tokenizer is not None:
        if digest_json(teacher_tokenizer.get_vocab()) != digest_json(
            tokenizer.get_vocab()
        ):
            raise ValueError("student and teacher tokenizers differ")

    records, aggregate = evaluate(
        model, teacher, tokenizer, args.device, selected, LIKELIHOOD_CONTRACT
    )
    finished = datetime.now(timezone.utc)
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
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "aggregate": aggregate,
        "samples": records,
        "resource_measurements": {
            "peak_accelerator_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if args.device != "cpu"
                else None
            ),
            "energy_joules": None,
            "energy_status": "requires external power meter or NVML sampler",
        },
    }
    report["report_content_sha256"] = digest_json(report)
    if not math.isfinite(report["aggregate"]["micro_perplexity"]):
        raise ValueError("non-finite likelihood result")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(
        f"wrote {args.out}: ppl={aggregate['micro_perplexity']:.4f}, "
        f"teacher KL={aggregate['micro_mean_teacher_kl']:.6f}"
    )


if __name__ == "__main__":
    main()
