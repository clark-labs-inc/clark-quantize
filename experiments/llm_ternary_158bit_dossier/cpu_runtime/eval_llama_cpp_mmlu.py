#!/usr/bin/env python3
"""Evaluate a frozen five-shot MMLU slice through llama.cpp exact choice logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_DIR = REPO_ROOT / "experiments" / "large_scale_ternary"
if str(GATE_DIR) not in sys.path:
    sys.path.insert(0, str(GATE_DIR))

from gate_b_protocol import (  # noqa: E402
    prompt_contract_sha256,
    select_and_verify_rows,
    select_and_verify_support_rows,
    verify_manifest,
    wilson_interval,
)

LETTERS = ("A", "B", "C", "D")
GRAMMAR = 'root ::= " A" | " B" | " C" | " D"'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json_write(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def request_json(
    server: str,
    endpoint: str,
    body: dict[str, Any],
    *,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        server.rstrip("/") + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"server returned HTTP {error.code} for {endpoint}: {detail}"
        ) from error
    if not isinstance(result, dict):
        raise ValueError(f"server returned non-object JSON for {endpoint}")
    return result, (time.perf_counter() - started) * 1000.0


def tokenizer_contract(server: str, timeout: float) -> dict[str, Any]:
    pieces: dict[str, dict[str, Any]] = {}
    for text in (*[f" {letter}" for letter in LETTERS], " "):
        response, _ = request_json(
            server,
            "/tokenize",
            {"content": text, "add_special": False, "with_pieces": True},
            timeout=timeout,
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or len(tokens) != 1:
            raise ValueError(f"{text!r} is not exactly one token: {tokens!r}")
        token = tokens[0]
        if token.get("piece") != text:
            raise ValueError(
                f"tokenizer piece mismatch for {text!r}: {token.get('piece')!r}"
            )
        pieces[text] = {"id": int(token["id"]), "piece": token["piece"]}
    label_ids = {letter: pieces[f" {letter}"]["id"] for letter in LETTERS}
    if len(set(label_ids.values())) != len(LETTERS):
        raise ValueError(f"choice labels do not have unique token IDs: {label_ids}")
    if pieces[" "]["id"] in set(label_ids.values()):
        raise ValueError("standalone-space token aliases a choice token")
    return {
        "labels": label_ids,
        "standalone_space": pieces[" "],
        "pieces": pieces,
    }


def format_row(row: dict[str, Any], *, include_answer: bool) -> str:
    choices = "\n".join(
        f"{LETTERS[index]}. {choice}" for index, choice in enumerate(row["choices"])
    )
    result = f"{row['question'].strip()}\n{choices}\nAnswer:"
    if include_answer:
        result += f" {LETTERS[int(row['answer'])]}\n\n"
    return result


def format_prompt(
    row: dict[str, Any], support_rows: list[dict[str, Any]], shots: int
) -> str:
    subject = row["subject"].replace("_", " ")
    header = (
        "The following are multiple choice questions (with answers) about "
        f"{subject}.\n\n"
    )
    examples = "".join(
        format_row(example, include_answer=True) for example in support_rows[:shots]
    )
    return header + examples + format_row(row, include_answer=False)


def extract_choice_probabilities(
    response: dict[str, Any], label_ids: dict[str, int]
) -> tuple[str, dict[str, float], dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("chat response must contain exactly one choice")
    content_rows = choices[0].get("logprobs", {}).get("content")
    if not isinstance(content_rows, list) or len(content_rows) != 1:
        raise ValueError("chat response must contain one generated-token probability row")
    token_row = content_rows[0]
    top = token_row.get("top_probs")
    if not isinstance(top, list) or len(top) != len(LETTERS):
        raise ValueError(f"expected exactly four post-sampling probabilities: {top!r}")
    by_id: dict[int, float] = {}
    for item in top:
        token_id = int(item["id"])
        probability = float(item["prob"])
        if token_id in by_id or not math.isfinite(probability) or probability <= 0:
            raise ValueError("invalid or duplicate choice probability")
        by_id[token_id] = probability
    if set(by_id) != set(label_ids.values()):
        raise ValueError(
            f"post-sampling token set differs from A/B/C/D: {sorted(by_id)}"
        )
    probabilities = {letter: by_id[label_ids[letter]] for letter in LETTERS}
    if abs(sum(probabilities.values()) - 1.0) > 1e-5:
        raise ValueError(f"choice probabilities do not sum to one: {probabilities}")
    prediction = max(LETTERS, key=lambda letter: probabilities[letter])
    sampled = {
        "id": int(token_row["id"]),
        "token": token_row["token"],
        "prob": float(token_row["prob"]),
        "message_content": choices[0].get("message", {}).get("content"),
    }
    return prediction, probabilities, sampled


def load_frozen_rows(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Any], dict[str, list[Any]]]:
    from datasets import load_dataset

    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest)
    if manifest.get("prompt_contract_sha256") != prompt_contract_sha256():
        raise ValueError("MMLU prompt contract hash mismatch")
    tasks = manifest.get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("task") != "mmlu":
        raise ValueError("manifest must contain exactly one MMLU task")
    spec = tasks[0]
    if int(spec.get("shots", -1)) != 5 or len(spec.get("samples", [])) != 64:
        raise ValueError("expected a frozen 64-row, five-shot MMLU manifest")
    dataset = load_dataset(
        spec["dataset"],
        spec.get("config"),
        split=spec["split"],
        revision=spec["revision"],
    )
    if dataset._fingerprint != spec["fingerprint"]:
        raise ValueError("MMLU test fingerprint drift")
    selected = select_and_verify_rows(dataset, spec)
    support_spec = spec["support"]
    support_dataset = load_dataset(
        support_spec["dataset"],
        support_spec.get("config"),
        split=support_spec["split"],
        revision=support_spec["revision"],
    )
    if support_dataset._fingerprint != support_spec["fingerprint"]:
        raise ValueError("MMLU support fingerprint drift")
    support = select_and_verify_support_rows(support_dataset, support_spec)
    return manifest, spec, selected, support


def validate_completed_prefix(records: list[dict[str, Any]], selected: list[Any]) -> None:
    if len(records) > len(selected):
        raise ValueError("partial receipt contains too many records")
    for index, record in enumerate(records):
        expected = selected[index][1]["sample_id"]
        if record.get("position") != index or record.get("sample_id") != expected:
            raise ValueError(f"partial record {index} does not match frozen row order")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.workers <= 0 or args.checkpoint_every <= 0 or args.timeout <= 0:
        parser.error("workers, checkpoint-every, and timeout must be positive")
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    if sha256_file(args.model) != args.model_sha256:
        raise ValueError("model SHA-256 does not match --model-sha256")

    manifest, spec, selected, support = load_frozen_rows(args.manifest)
    token_contract = tokenizer_contract(args.server, args.timeout)
    label_ids = token_contract["labels"]
    space_id = int(token_contract["standalone_space"]["id"])
    evaluator_sha = sha256_file(Path(__file__))
    contract = {
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_content_sha256": manifest["content_sha256"],
        "model": str(args.model.resolve()),
        "model_label": args.model_label,
        "model_sha256": args.model_sha256,
        "server": args.server,
        "workers": args.workers,
        "timeout": args.timeout,
        "checkpoint_every": args.checkpoint_every,
        "evaluator_sha256": evaluator_sha,
        "tokenizer_contract": token_contract,
        "completion": {
            "endpoint": "/v1/chat/completions",
            "max_tokens": 1,
            "temperature": 1.0,
            "samplers": ["temperature"],
            "grammar": GRAMMAR,
            "post_sampling_probs": True,
            "top_logprobs": 4,
            "standalone_space_logit_bias": False,
            "enable_thinking": False,
            "seed_formula": "manifest_seed_plus_dataset_index",
        },
    }
    partial_path = args.out.with_suffix(args.out.suffix + ".partial")
    records: list[dict[str, Any]] = []
    resumed_from = 0
    if partial_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"partial receipt exists; inspect it and pass --resume: {partial_path}"
            )
        partial = json.loads(partial_path.read_text())
        if partial.get("schema") != "llama_cpp_mmlu_checkpoint_v1":
            raise ValueError("unsupported MMLU partial receipt schema")
        if partial.get("contract") != contract:
            raise ValueError("MMLU partial receipt contract mismatch")
        records = list(partial.get("records", []))
        validate_completed_prefix(records, selected)
        resumed_from = len(records)
        print(
            json.dumps({"event": "resume", "completed": resumed_from}),
            flush=True,
        )
    elif args.resume:
        raise FileNotFoundError(f"resume requested but partial is absent: {partial_path}")

    def save_partial() -> None:
        atomic_json_write(
            {
                "schema": "llama_cpp_mmlu_checkpoint_v1",
                "contract": contract,
                "records": records,
            },
            partial_path,
        )

    def evaluate_one(position: int) -> dict[str, Any]:
        row, sample = selected[position]
        subject = row["subject"]
        prompt = format_prompt(row, support[subject], int(spec["shots"]))
        body = {
            "model": args.model_label,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 1.0,
            "seed": int(manifest["seed"]) + int(sample["index"]),
            "logprobs": True,
            "top_logprobs": 4,
            "grammar": GRAMMAR,
            "post_sampling_probs": True,
            "samplers": ["temperature"],
            "logit_bias": [[space_id, False]],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response, latency_ms = request_json(
            args.server,
            "/v1/chat/completions",
            body,
            timeout=args.timeout,
        )
        response_model = response.get("model")
        if not isinstance(response_model, str) or Path(response_model).resolve() != args.model.resolve():
            raise ValueError(
                f"server response model differs from bound model: {response_model!r}"
            )
        prediction, probabilities, sampled = extract_choice_probabilities(
            response, label_ids
        )
        gold = LETTERS[int(row["answer"])]
        return {
            "position": position,
            "sample_id": sample["sample_id"],
            "dataset_index": int(sample["index"]),
            "subject": subject,
            "row_sha256": sample["row_sha256"],
            "correct": prediction == gold,
            "prediction": prediction,
            "gold": gold,
            "choice_probabilities": probabilities,
            "sampled": sampled,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "usage": response.get("usage"),
            "timings": response.get("timings"),
            "system_fingerprint": response.get("system_fingerprint"),
            "latency_ms": latency_ms,
        }

    started = datetime.now(timezone.utc)
    for start in range(resumed_from, len(selected), args.workers):
        positions = list(range(start, min(start + args.workers, len(selected))))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            batch = list(executor.map(evaluate_one, positions))
        records.extend(batch)
        validate_completed_prefix(records, selected)
        if len(records) % args.checkpoint_every == 0 or len(records) == len(selected):
            save_partial()
        correct = sum(bool(record["correct"]) for record in records)
        print(
            json.dumps(
                {
                    "event": "mmlu_batch",
                    "completed": len(records),
                    "total": len(selected),
                    "correct": correct,
                    "accuracy": correct / len(records),
                    "last_sample_id": records[-1]["sample_id"],
                }
            ),
            flush=True,
        )

    if len(records) != 64:
        raise RuntimeError(f"MMLU denominator mismatch: {len(records)} != 64")
    finished = datetime.now(timezone.utc)
    correct = sum(bool(record["correct"]) for record in records)
    subject_counts = Counter(record["subject"] for record in records)
    result = {
        "schema": "llama_cpp_mmlu_exact_choice_v1",
        "status": "completed",
        "valid": True,
        "created_at": finished.isoformat(),
        "started_at": started.isoformat(),
        "elapsed_seconds_this_invocation": (finished - started).total_seconds(),
        "resumed_from_records": resumed_from,
        "contract": contract,
        "dataset": {
            "id": spec["dataset"],
            "config": spec.get("config"),
            "split": spec["split"],
            "revision": spec["revision"],
            "fingerprint": spec["fingerprint"],
            "support_fingerprint": spec["support"]["fingerprint"],
            "shots": int(spec["shots"]),
            "sampling": spec["sampling"],
            "subjects": len(subject_counts),
            "subject_counts": dict(sorted(subject_counts.items())),
        },
        "summary": {
            "task": "mmlu",
            "correct": correct,
            "n": len(records),
            "accuracy": correct / len(records),
            "accuracy_wilson_95_ci": wilson_interval(correct, len(records)),
        },
        "records": records,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "workers": args.workers,
        },
    }
    result["report_content_sha256"] = canonical_digest(result)
    atomic_json_write(result, args.out)
    partial_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "correct": correct,
                "n": len(records),
                "report_content_sha256": result["report_content_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
