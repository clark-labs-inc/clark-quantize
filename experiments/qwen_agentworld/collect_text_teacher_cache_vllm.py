"""Build dense text/reasoning teacher caches with vLLM prompt logprobs.

This is the fast path for bulk Qwen all-2D QAT cache generation. It only
handles text rows; multimodal image/video rows still use
collect_multimodal_teacher_cache.py because vLLM prompt-logprob extraction for
Qwen multimodal payloads is a separate integration.

Rows use the existing qwen_agentworld_teacher_behavior_v1 envelope, but store
normalized teacher log-probabilities in teacher_top_logprobs instead of raw
teacher top logits plus teacher_logsumexp.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_multimodal_teacher_cache import (  # noqa: E402
    aya_messages,
    generic_text_messages,
    openr1_messages,
    parse_csv,
    stream_aya,
    stream_fineweb,
    stream_openr1,
)
from collect_onpolicy_survival_cache import token_ids_for_strings  # noqa: E402


def apply_text_template(processor: Any, source_kind: str, sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if source_kind == "aya":
        messages = aya_messages(sample)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text, {"modality": "text", "language": sample.get("language_code") or sample.get("language")}
    if source_kind == "openr1":
        messages = openr1_messages(sample)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text, {"modality": "text", "reasoning": True, "think_tokens": "<think>" in text}
    if source_kind == "text":
        content = generic_text_messages(sample)
        if isinstance(content, list):
            text = processor.apply_chat_template(content, tokenize=False, add_generation_prompt=False)
        else:
            text = str(content)
        return text, {"modality": "text"}
    raise ValueError(f"vLLM text cache does not support source kind: {source_kind}")


def tokenized_text_payload(
    processor: Any,
    text: str,
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = processor.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"][0].to(torch.long).contiguous()
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def logprob_float(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    if isinstance(value, (tuple, list)) and value:
        return float(value[0])
    return float(value)


def top_logprobs_to_tensors(
    prompt_logprobs: list[Any],
    *,
    original_tokens: int,
    top_k: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [original_tokens, top_k] ids/logprobs aligned to model positions.

    The vLLM prompt logprob at prompt position i is the distribution after the
    previous token. We append one probe token to the prompt, then use positions
    1..L to supervise original model positions 0..L-1.
    """

    indices = torch.full((original_tokens, top_k), int(pad_token_id), dtype=torch.int32)
    values = torch.full((original_tokens, top_k), -float("inf"), dtype=torch.float16)
    if len(prompt_logprobs) < original_tokens + 1:
        raise RuntimeError(f"prompt_logprobs length {len(prompt_logprobs)} < required {original_tokens + 1}")
    for position in range(original_tokens):
        entry = prompt_logprobs[position + 1]
        if entry is None:
            raise RuntimeError(f"vLLM returned None prompt_logprobs at position {position + 1}")
        if not isinstance(entry, dict):
            raise RuntimeError(f"unsupported vLLM prompt_logprobs entry type: {type(entry)}")
        pairs = sorted(
            ((int(token_id), logprob_float(logprob)) for token_id, logprob in entry.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        if not pairs:
            raise RuntimeError(f"empty prompt_logprobs entry at position {position + 1}")
        for column, (token_id, logprob) in enumerate(pairs):
            indices[position, column] = token_id
            values[position, column] = logprob
    return indices, values


def build_sources(args: argparse.Namespace) -> Iterable[tuple[str, dict[str, Any], str]]:
    return chain(
        stream_aya(args.aya_samples, args.aya_offset),
        stream_openr1(args.openr1_samples, args.openr1_offset),
        stream_fineweb(parse_csv(args.fineweb_configs), args.fineweb_per_config, args.fineweb_offset),
    )


def flush_batch(
    *,
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[list[int]],
    pending: list[dict[str, Any]],
    top_k: int,
    pad_token_id: int,
) -> list[dict[str, Any]]:
    if not prompts:
        return []
    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    rows: list[dict[str, Any]] = []
    for item, output in zip(pending, outputs, strict=True):
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if prompt_logprobs is None:
            raise RuntimeError("vLLM output did not include prompt_logprobs")
        input_ids = item["input_ids"]
        top_indices, top_logprobs = top_logprobs_to_tensors(
            prompt_logprobs,
            original_tokens=int(input_ids.shape[0]),
            top_k=top_k,
            pad_token_id=pad_token_id,
        )
        rows.append(
            {
                "task": item["task"],
                "id": item["id"],
                "index": item["index"],
                "tokens": int(input_ids.shape[0]),
                "keep": int(input_ids.shape[0]),
                "input_ids": input_ids,
                "attention_mask": item["attention_mask"],
                "top_indices": top_indices,
                "teacher_top_logprobs": top_logprobs,
                "source_meta": item["source_meta"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--save-every-rows", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=1344)
    parser.add_argument("--aya-samples", type=int, default=256)
    parser.add_argument("--aya-offset", type=int, default=0)
    parser.add_argument("--openr1-samples", type=int, default=512)
    parser.add_argument("--openr1-offset", type=int, default=0)
    parser.add_argument(
        "--fineweb-configs",
        default="spa_Latn,fra_Latn,deu_Latn,arb_Arab,hin_Deva,cmn_Hani,rus_Cyrl,jpn_Jpan,kor_Hang",
    )
    parser.add_argument("--fineweb-per-config", type=int, default=64)
    parser.add_argument("--fineweb-offset", type=int, default=0)
    parser.add_argument("--fast-exit", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer = processor.tokenizer
    probe_token_id = tokenizer.eos_token_id
    if probe_token_id is None:
        probe_token_id = tokenizer.pad_token_id
    if probe_token_id is None:
        probe_token_id = 0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else probe_token_id

    llm = LLM(
        model=str(args.model_dir),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_length + 2,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_logprobs=max(args.top_k, 20),
    )
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=args.top_k,
        detokenize=False,
        skip_special_tokens=False,
    )

    out_rows: list[dict[str, Any]] = []
    failures = 0
    prompts: list[list[int]] = []
    pending: list[dict[str, Any]] = []
    pending_tokens = 0
    next_save_rows = args.save_every_rows if args.save_every_rows > 0 else 0

    def build_payload() -> dict[str, Any]:
        format_token_ids = token_ids_for_strings(
            tokenizer,
            ["<think>", "</think>", "<predicted_observation>", "</predicted_observation>", "\n", "[", "]", "{", "}", '"'],
        )
        collapse_token_ids = token_ids_for_strings(tokenizer, ["<think>", "assistant", "\n\n", "\n"])
        return {
            "format": "qwen_agentworld_teacher_behavior_v1",
            "model_dir": str(args.model_dir),
            "data_dir": "vllm_hf_stream:CohereLabs/aya_dataset,open-r1/OpenR1-Math-220k,HuggingFaceFW/fineweb-2",
            "max_length": args.max_length,
            "last_n": 0,
            "dense_positions": True,
            "top_k": args.top_k,
            "teacher_values_kind": "logprobs",
            "position_alignment": "vllm_prompt_logprobs_probe_token_shift_plus_one",
            "format_token_ids": format_token_ids,
            "collapse_token_ids": collapse_token_ids,
            "hidden_layers": [],
            "builder": "collect_text_teacher_cache_vllm.py",
            "builder_args": vars(args) | {"model_dir": str(args.model_dir), "out": str(args.out)},
            "failures": failures,
            "rows": out_rows,
        }

    def write_payload(path: Path) -> None:
        payload = build_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        meta_path = path.with_suffix(path.suffix + ".json")
        meta_path.write_text(
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
                            "positions": int(row["teacher_top_logprobs"].shape[0]),
                            "top_k": int(row["teacher_top_logprobs"].shape[-1]),
                            "meta": row.get("source_meta", {}),
                        }
                        for row in out_rows
                    ]
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"partial_out": str(path), "meta": str(meta_path), "rows": len(out_rows), "failures": failures}), flush=True)

    def flush_pending() -> None:
        nonlocal prompts, pending, pending_tokens, out_rows, next_save_rows
        batch_rows = flush_batch(
            llm=llm,
            sampling_params=sampling_params,
            prompts=prompts,
            pending=pending,
            top_k=args.top_k,
            pad_token_id=pad_token_id,
        )
        for row in batch_rows:
            row["index"] = len(out_rows)
            out_rows.append(row)
            print(
                json.dumps(
                    {
                        "index": row["index"],
                        "source": row["id"],
                        "tokens": row["tokens"],
                        "positions": row["keep"],
                        "top_k": int(row["teacher_top_logprobs"].shape[-1]),
                    }
                ),
                flush=True,
            )
        prompts = []
        pending = []
        pending_tokens = 0
        if next_save_rows > 0 and len(out_rows) >= next_save_rows:
            write_payload(args.out.with_suffix(args.out.suffix + ".partial"))
            while next_save_rows > 0 and len(out_rows) >= next_save_rows:
                next_save_rows += args.save_every_rows

    for _source_index, (source_kind, sample, source_id) in enumerate(build_sources(args)):
        if len(out_rows) >= args.max_rows:
            break
        try:
            text, meta = apply_text_template(processor, source_kind, sample)
            payload = tokenized_text_payload(
                processor,
                text,
                max_length=args.max_length,
            )
            input_ids = payload["input_ids"]
            if input_ids.numel() == 0:
                raise RuntimeError("empty tokenized row")
            row_tokens = int(input_ids.numel())
            if prompts and (
                len(prompts) >= args.batch_size
                or (args.max_batch_tokens > 0 and pending_tokens + row_tokens > args.max_batch_tokens)
            ):
                flush_pending()
            prompts.append([int(token) for token in input_ids.tolist()] + [int(probe_token_id)])
            pending_tokens += row_tokens
            pending.append(
                {
                    "task": source_kind,
                    "id": source_id,
                    "index": len(out_rows) + len(pending),
                    "input_ids": input_ids,
                    "attention_mask": payload["attention_mask"],
                    "source_meta": meta,
                }
            )
            if len(prompts) >= args.batch_size:
                flush_pending()
        except Exception as exc:  # noqa: BLE001 - external rows drift.
            failures += 1
            print(json.dumps({"skip": source_id, "error": type(exc).__name__, "message": str(exc)[:240]}), flush=True)

    if prompts:
        flush_pending()

    write_payload(args.out)
    meta_path = args.out.with_suffix(args.out.suffix + ".json")
    print(json.dumps({"out": str(args.out), "meta": str(meta_path), "rows": len(out_rows), "failures": failures}, indent=2))
    if args.fast_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
