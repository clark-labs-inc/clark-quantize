"""Collect BF16 teacher labels on the teacher's own text completions.

Unlike prompt-only behavior caches, each retained logit predicts an actual
teacher-generated completion token.  This gives blockwise hard-ternary QAT a
behavioral target across answer/reasoning trajectories rather than mostly
chat-template reconstruction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_onpolicy_survival_cache import parse_layers, token_ids_for_strings
from train_mlp_triplet_survival_qat import load_model, register_hidden_hooks


def extract_text(row: dict[str, Any]) -> str:
    for key in ("text", "context", "prompt", "question", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"row has no usable text field: keys={sorted(row)}")


def causal_label_sequence(prompt_ids: torch.Tensor, generated_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return inputs whose final positions predict every generated token.

    For completion tokens c0..cN, causal logits at positions ending with
    prompt[-1], c0..cN-1 predict c0..cN.  We therefore concatenate and remove
    the final generated token before the teacher-label forward pass.
    """
    if prompt_ids.ndim != 1 or generated_ids.ndim != 1:
        raise ValueError("prompt_ids and generated_ids must be rank-1")
    if prompt_ids.numel() < 1:
        raise ValueError("prompt must contain at least one token")
    if generated_ids.numel() < 1:
        raise ValueError("generation must contain at least one token")
    full = torch.cat((prompt_ids, generated_ids))
    return full[:-1].contiguous(), int(generated_ids.numel())


def alignment_counts(top_indices: torch.Tensor, target_ids: torch.Tensor) -> tuple[int, int]:
    if top_indices.ndim != 2 or target_ids.ndim != 1 or top_indices.shape[0] != target_ids.shape[0]:
        raise ValueError("top_indices and target_ids have incompatible shapes")
    matches = top_indices.eq(target_ids.reshape(-1, 1))
    return int(matches[:, 0].sum().item()), int(matches.any(dim=-1).sum().item())


def hidden_projection(
    *,
    layer_id: int,
    hidden_dim: int,
    projected_dim: int,
    seed: int,
    device: torch.device,
    cache: dict[tuple[int, int, int], torch.Tensor],
) -> torch.Tensor:
    key = (layer_id, hidden_dim, projected_dim)
    projection = cache.get(key)
    if projection is not None and projection.device == device:
        return projection
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer_id * 1009 + hidden_dim * 9176 + projected_dim)
    projection_cpu = torch.empty(hidden_dim, projected_dim, dtype=torch.float32)
    projection_cpu.bernoulli_(0.5, generator=generator)
    projection_cpu.mul_(2.0).sub_(1.0).div_(math.sqrt(projected_dim))
    projection = projection_cpu.to(device=device)
    cache[key] = projection
    return projection


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_write_json(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def release_non_text_modules(model: torch.nn.Module) -> None:
    core = getattr(model, "model", model)
    for attribute in ("visual", "vision_model", "vision_tower"):
        if getattr(core, attribute, None) is not None:
            setattr(core, attribute, None)
    if getattr(model, "mtp", None) is not None:
        model.mtp = None
    gc.collect()
    torch.cuda.empty_cache()


def load_jsonl(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=224)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--generate-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--hidden-layers", default="3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=20260701)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()

    if args.torch_threads < 0:
        raise ValueError("torch-threads must be non-negative")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    if args.generate_tokens < 1 or args.generate_tokens >= args.max_model_len:
        raise ValueError("generate-tokens must be positive and smaller than max-model-len")

    torch.manual_seed(args.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    model = load_model(args.model_dir, device=args.device)
    release_non_text_modules(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device

    hidden_layers = parse_layers(args.hidden_layers)
    keep_ref = {"keep": 1}
    handles, captured = register_hidden_hooks(model, hidden_layers, keep_ref)
    projection_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    sources = load_jsonl(args.input, args.max_rows)
    out_rows: list[dict[str, Any]] = []
    try:
        for index, source in enumerate(sources):
            chat_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": extract_text(source)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(
                [chat_text],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_model_len - args.generate_tokens,
            )
            prompt_ids = encoded["input_ids"][0].to(device)
            prompt_mask = encoded["attention_mask"].to(device)
            generated = model.generate(
                input_ids=prompt_ids.unsqueeze(0),
                attention_mask=prompt_mask,
                do_sample=False,
                max_new_tokens=args.generate_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
            completion_ids = generated[prompt_ids.numel():]
            label_ids, keep = causal_label_sequence(prompt_ids, completion_ids)
            attention_mask = torch.ones_like(label_ids, dtype=torch.long)
            keep_ref["keep"] = keep
            captured.clear()
            with torch.no_grad():
                outputs = model(
                    input_ids=label_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    use_cache=False,
                    logits_to_keep=keep,
                )
            logits = outputs.logits[0, -keep:, :].detach().float()
            teacher_logsumexp = torch.logsumexp(logits, dim=-1)
            top_values, top_indices = torch.topk(logits, k=min(args.top_k, logits.shape[-1]), dim=-1)
            greedy_matches, topk_matches = alignment_counts(top_indices, completion_ids)
            if topk_matches != keep:
                raise RuntimeError(
                    f"causal label alignment failed for row {index}: "
                    f"generated token present in teacher top-k at {topk_matches}/{keep} positions"
                )
            projected: dict[int, torch.Tensor] = {}
            for layer_id in hidden_layers:
                hidden = captured[layer_id].float()
                projection = hidden_projection(
                    layer_id=layer_id,
                    hidden_dim=hidden.shape[-1],
                    projected_dim=args.projection_dim,
                    seed=args.projection_seed,
                    device=device,
                    cache=projection_cache,
                )
                projected[layer_id] = (F.layer_norm(hidden, hidden.shape[-1:]) @ projection).cpu().to(torch.float16).contiguous()
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            row = {
                "task": source.get("source", "teacher_rollout"),
                "id": source.get("id", index),
                "index": index,
                "tokens": int(label_ids.numel()),
                "keep": keep,
                "prompt_tokens": int(prompt_ids.numel()),
                "generated_tokens": int(completion_ids.numel()),
                "greedy_alignment_matches": greedy_matches,
                "topk_alignment_matches": topk_matches,
                "target_ids": completion_ids.cpu().to(torch.long),
                "input_ids": label_ids.cpu().to(torch.long),
                "attention_mask": attention_mask.cpu(),
                "top_values": top_values.cpu().to(torch.float16),
                "top_indices": top_indices.cpu().to(torch.int32),
                "teacher_logsumexp": teacher_logsumexp.cpu().to(torch.float32),
                "teacher_hidden_proj": projected,
                "completion_text": completion_text,
            }
            out_rows.append(row)
            print(json.dumps({
                "event": "teacher_rollout_labeled",
                "index": index,
                "task": row["task"],
                "id": row["id"],
                "prompt_tokens": row["prompt_tokens"],
                "generated_tokens": row["generated_tokens"],
                "preview": completion_text[:160],
                "greedy_alignment": f"{greedy_matches}/{keep}",
                "topk_alignment": f"{topk_matches}/{keep}",
            }), flush=True)
            del outputs, logits, top_values, top_indices, generated
            captured.clear()
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    payload = {
        "format": "qwen_agentworld_teacher_behavior_v1",
        "label_semantics": "teacher_completion_next_token_v1",
        "model_dir": str(args.model_dir),
        "data_dir": str(args.input),
        "max_length": args.max_model_len,
        "limit_jobs": args.max_rows,
        "top_k": args.top_k,
        "hidden_layers": hidden_layers,
        "hidden_projection_dim": args.projection_dim,
        "hidden_projection_seed": args.projection_seed,
        "runtime_device": str(device),
        "torch_threads": torch.get_num_threads(),
        "format_token_ids": token_ids_for_strings(tokenizer, ["<think>", "</think>", "\n", "[", "]", "{", "}", '"']),
        "collapse_token_ids": token_ids_for_strings(tokenizer, ["<think>", "assistant", "\n\n", "\n"]),
        "rows": out_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, args.out)
    receipt = {
        "schema": "qwen_teacher_rollout_cache_v1",
        "path": str(args.out),
        "sha256": sha256_file(args.out),
        "size_bytes": args.out.stat().st_size,
        "rows": len(out_rows),
        "task_counts": {task: sum(row["task"] == task for row in out_rows) for task in sorted({row["task"] for row in out_rows})},
        "total_generated_tokens": sum(row["generated_tokens"] for row in out_rows),
        "greedy_alignment_matches": sum(row["greedy_alignment_matches"] for row in out_rows),
        "greedy_alignment_total": sum(row["generated_tokens"] for row in out_rows),
        "topk_alignment_matches": sum(row["topk_alignment_matches"] for row in out_rows),
        "topk_alignment_total": sum(row["generated_tokens"] for row in out_rows),
        "label_semantics": payload["label_semantics"],
        "hidden_layers": hidden_layers,
        "hidden_projection_dim": args.projection_dim,
        "runtime_device": str(device),
        "torch_threads": torch.get_num_threads(),
        "rows_summary": [
            {
                "index": row["index"],
                "task": row["task"],
                "id": row["id"],
                "tokens": row["tokens"],
                "generated_tokens": row["generated_tokens"],
                "greedy_alignment_matches": row["greedy_alignment_matches"],
                "topk_alignment_matches": row["topk_alignment_matches"],
                "completion_preview": row["completion_text"][:240],
            }
            for row in out_rows
        ],
    }
    atomic_write_json(receipt, args.out.with_suffix(args.out.suffix + ".json"))
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows_summary"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
