#!/usr/bin/env python3
"""Build a reproducible teacher-prompt set without frozen evaluation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k-revision", required=True)
    parser.add_argument("--mmlu-revision", required=True)
    parser.add_argument("--gsm8k-rows", type=int, default=96)
    parser.add_argument("--mmlu-rows", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="train", revision=args.gsm8k_revision)
    mmlu = load_dataset("cais/mmlu", "all", split="auxiliary_train", revision=args.mmlu_revision)
    rng = random.Random(args.seed)
    gsm_indices = rng.sample(range(len(gsm)), args.gsm8k_rows)
    mmlu_indices = rng.sample(range(len(mmlu)), args.mmlu_rows)
    records: list[dict[str, Any]] = []
    for index in gsm_indices:
        row = gsm[index]
        records.append({
            "id": f"gsm8k_train:{index}",
            "source": "gsm8k_train",
            "source_row_sha256": digest(row),
            "text": "Solve this arithmetic word problem carefully and give the final numeric answer:\n" + row["question"],
        })
    letters = ["A", "B", "C", "D"]
    for index in mmlu_indices:
        row = mmlu[index]
        choices = "\n".join(f"{letters[i]}. {choice}" for i, choice in enumerate(row["choices"]))
        records.append({
            "id": f"mmlu_auxiliary_train:{index}",
            "source": "mmlu_auxiliary_train",
            "source_row_sha256": digest(row),
            "text": f"Answer this multiple-choice question about {row['subject'].replace('_', ' ')} and explain briefly.\n{row['question']}\n{choices}",
        })
    skills = [
        "write a Python function that merges overlapping integer intervals",
        "explain why a held-out evaluation set must not be used for training",
        "derive the area of a circle from an integral in polar coordinates",
        "summarize the difference between precision and recall with an example",
        "translate a short weather forecast from English to Spanish",
        "produce valid JSON describing three dependent engineering tasks",
        "reason about whether a graph with an odd cycle is bipartite",
        "explain the bias-variance tradeoff to a new machine-learning engineer",
        "debug a binary search that fails when the target is the last element",
        "compare TCP flow control with congestion control",
        "calculate compound interest and show each intermediate quantity",
        "explain conservation of momentum in a two-body collision",
        "write a SQL query using a window function to rank sales per region",
        "state the contrapositive of a quantified implication and prove equivalence",
        "give a concise safety checklist for rotating cloud credentials",
        "classify a chemical reaction as endothermic or exothermic from evidence",
    ]
    for cycle in range(2):
        for index, skill in enumerate(skills):
            records.append({
                "id": f"synthetic_general:{cycle}:{index}",
                "source": "synthetic_general",
                "source_row_sha256": None,
                "text": f"Task: {skill}. Provide a correct, self-contained response. Variation {cycle + 1}.",
            })
    rng.shuffle(records)
    if len({record["id"] for record in records}) != len(records):
        raise RuntimeError("duplicate teacher prompt IDs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = b"".join(canonical(record) + b"\n" for record in records)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(args.out)
    receipt = {
        "schema": "qwen38_teacher_prompts_v1",
        "path": str(args.out),
        "sha256": hashlib.sha256(body).hexdigest(),
        "rows": len(records),
        "seed": args.seed,
        "sources": {
            "gsm8k_train": {"revision": args.gsm8k_revision, "fingerprint": gsm._fingerprint, "rows": args.gsm8k_rows},
            "mmlu_auxiliary_train": {"revision": args.mmlu_revision, "fingerprint": mmlu._fingerprint, "rows": args.mmlu_rows},
            "synthetic_general": {"rows": len(skills) * 2},
        },
        "frozen_eval_rows_included": False,
    }
    receipt["content_sha256"] = digest(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    receipt_tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt_tmp.replace(args.receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
