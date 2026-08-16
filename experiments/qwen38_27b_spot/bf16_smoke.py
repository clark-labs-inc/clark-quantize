#!/usr/bin/env python3
"""Load the exact local Qwen3.8-27B checkpoint and run a bounded text smoke."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/qwen38-bf16-smoke.json"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    hypothesis = (
        "The official BF16 checkpoint loads and performs text generation "
        "within one supported high-memory GPU."
    )
    prompt = "In one sentence, explain why quantization needs held-out evaluation."
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
    load_seconds = time.time() - started
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    generation_started = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    generation_seconds = time.time() - generation_started
    completion_ids = generated[0, inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

    receipt = {
        "schema": "clark_qwen38_bf16_smoke_v1",
        "hypothesis": hypothesis,
        "model_dir": str(args.model_dir),
        "model_class": model_class,
        "dtype": str(next(model.parameters()).dtype),
        "prompt": prompt,
        "completion": completion,
        "input_tokens": int(inputs["input_ids"].numel()),
        "new_tokens": int(completion_ids.numel()),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "tokens_per_second": completion_ids.numel() / max(generation_seconds, 1e-9),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "stop_condition": "One deterministic 16-token-or-shorter completion; no training or network access.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
