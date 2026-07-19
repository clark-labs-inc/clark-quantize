"""Collect BF16 teacher behavior for generic/OOD text prompts.

The output intentionally matches qwen_agentworld_teacher_behavior_v1 so the
existing all-2D QAT trainer can consume it. This is local teacher distillation:
no external judge/model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_onpolicy_survival_cache import load_model, token_ids_for_strings


def extract_text(row: dict[str, Any]) -> str:
    for key in ("text", "context", "prompt", "question", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                role = msg.get("role", "user")
                parts.append(f"{role}: {msg['content']}")
        if parts:
            return "\n".join(parts)
    raise ValueError(f"row has no usable text field: keys={sorted(row)}")


def make_chat_text(tokenizer: Any, text: str, *, as_chat: bool) -> str:
    if not as_chat:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--last-n", type=int, default=64)
    parser.add_argument("--dense-positions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--as-chat", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.truncation_side = args.truncation_side
    model = load_model(args.model_dir)
    for param in model.parameters():
        param.requires_grad_(False)
    device = next(model.parameters()).device

    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= args.max_rows:
                break

    out_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        text = make_chat_text(tokenizer, extract_text(source), as_chat=args.as_chat)
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=args.max_length)
        model_inputs = {key: value.to(device) for key, value in inputs.items()}
        sequence_length = int(inputs["input_ids"].shape[-1])
        keep = sequence_length if args.dense_positions or args.last_n <= 0 else min(args.last_n, sequence_length)
        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=False, logits_to_keep=keep)
        logits = outputs.logits[0, -keep:, :].detach().float()
        teacher_logsumexp = torch.logsumexp(logits, dim=-1)
        top_values, top_indices = torch.topk(logits, k=min(args.top_k, logits.shape[-1]), dim=-1)
        row = {
            "task": source.get("source", "ood_text"),
            "id": source.get("qid", source.get("id", index)),
            "index": index,
            "tokens": int(inputs["input_ids"].shape[-1]),
            "keep": keep,
            "input_ids": inputs["input_ids"][0].cpu().to(torch.long),
            "attention_mask": inputs["attention_mask"][0].cpu().to(torch.long),
            "top_values": top_values.cpu().to(torch.float16),
            "top_indices": top_indices.cpu().to(torch.int32),
            "teacher_logsumexp": teacher_logsumexp.cpu().to(torch.float32),
        }
        out_rows.append(row)
        print(
            json.dumps(
                {
                    "index": index,
                    "task": row["task"],
                    "id": row["id"],
                    "tokens": row["tokens"],
                    "positions": keep,
                    "top_k": int(top_values.shape[-1]),
                }
            ),
            flush=True,
        )

    payload = {
        "format": "qwen_agentworld_teacher_behavior_v1",
        "model_dir": str(args.model_dir),
        "data_dir": str(args.input),
        "max_length": args.max_length,
        "limit_jobs": args.max_rows,
        "per_domain_limit": None,
        "last_n": args.last_n,
        "dense_positions": args.dense_positions,
        "top_k": args.top_k,
        "format_token_ids": token_ids_for_strings(
            tokenizer,
            ["<think>", "</think>", "<predicted_observation>", "</predicted_observation>", "\n", "[", "]", "{", "}", '"'],
        ),
        "collapse_token_ids": token_ids_for_strings(tokenizer, ["<think>", "assistant", "\n\n", "\n"]),
        "rows": out_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "rows"
            }
            | {
                "rows": [
                    {
                        "task": row["task"],
                        "id": row["id"],
                        "tokens": row["tokens"],
                        "positions": int(row["top_values"].shape[0]),
                        "top_k": int(row["top_values"].shape[-1]),
                    }
                    for row in out_rows
                ]
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"out": str(args.out), "rows": len(out_rows)}, indent=2))


if __name__ == "__main__":
    main()
