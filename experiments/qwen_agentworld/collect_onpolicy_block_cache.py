"""Collect student-state block targets for one-GPU Gauss-Seidel QAT.

The cache stores h_l^Q from the compressed model, then labels it with
B_l^T(h_l^Q) from the BF16 teacher. This is the block-local analogue of
on-policy distillation: the teacher evaluates states the low-bit model visits.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_onpolicy_survival_cache import (
    collapse_summary,
    load_agentworld_eval,
    load_model,
    messages_for_job,
    parse_layers,
    token_ids_for_strings,
)
from weight_patch import apply_weight_patch


def first_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def layer_module(model: torch.nn.Module, layer_id: int) -> torch.nn.Module:
    return model.get_submodule(f"model.language_model.layers.{layer_id}")


def rows_to_json(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": row["index"],
            "task": row.get("task"),
            "id": row.get("id"),
            "tokens": int(row["tokens"]),
            "keep": int(row["keep"]),
            "collapse": row["collapse"],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--eval-dir", default="/workspace/Qwen-AgentWorld/eval", type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weight-patch", action="append", type=Path)
    parser.add_argument("--layers", default="0,1,2,3,28,29,30,31")
    parser.add_argument("--max-rows", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--generate-tokens", type=int, default=96)
    parser.add_argument("--label-positions", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    layers = parse_layers(args.layers)

    agentworld_eval = load_agentworld_eval(args.eval_dir)
    jobs = agentworld_eval.load_data(str(args.data_dir))[: args.max_rows]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.truncation_side = args.truncation_side

    student = load_model(args.model_dir)
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)
    patch_reports = [apply_weight_patch(student, patch) for patch in args.weight_patch or []]
    device = first_device(student)

    rows: list[dict[str, Any]] = []
    captured: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)

    def make_hook(layer_id: int):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            row_index = len(rows)
            hidden = inputs[0].detach().cpu().to(torch.bfloat16).squeeze(0).contiguous()
            pos = kwargs.get("position_embeddings")
            if not (isinstance(pos, tuple) and len(pos) == 2):
                raise RuntimeError("expected tuple position_embeddings")
            position_ids = kwargs.get("position_ids")
            captured[row_index][layer_id] = {
                "h_in": hidden,
                "cos": pos[0].detach().cpu().to(torch.bfloat16).squeeze(0).contiguous(),
                "sin": pos[1].detach().cpu().to(torch.bfloat16).squeeze(0).contiguous(),
                "position_ids": position_ids.detach().cpu().squeeze(0).contiguous() if torch.is_tensor(position_ids) else None,
            }

        return hook

    handles = [layer_module(student, layer).register_forward_hook(make_hook(layer), with_kwargs=True) for layer in layers]
    try:
        for index, job in enumerate(jobs, start=1):
            text = tokenizer.apply_chat_template(
                messages_for_job(job),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=args.max_model_len)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            print(json.dumps({"event": "generate_start", "index": index, "task": job.get("task"), "id": job.get("id"), "prompt_tokens": int(inputs["input_ids"].shape[-1])}), flush=True)
            with torch.no_grad():
                generated = student.generate(
                    **inputs,
                    max_new_tokens=args.generate_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Run a full forward pass to trigger block hooks on the actual generated prefix.
                student(input_ids=generated, attention_mask=torch.ones_like(generated), use_cache=False, logits_to_keep=1)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            new_tokens = generated[0, prompt_tokens:]
            generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
            keep = max(1, min(args.label_positions, int(new_tokens.numel()), int(generated.shape[-1])))
            row = {
                "index": index,
                "task": job.get("task"),
                "id": job.get("id"),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(new_tokens.numel()),
                "tokens": int(generated.shape[-1]),
                "keep": keep,
                "input_ids": generated.squeeze(0).cpu().to(torch.long),
                "collapse": collapse_summary(generation),
                "generation": generation,
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("index", "task", "id", "tokens", "keep", "collapse")}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    del student
    torch.cuda.empty_cache()

    teacher = load_model(args.model_dir)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher_device = first_device(teacher)
    for row_offset, row in enumerate(rows):
        keep = int(row["keep"])
        input_ids = row["input_ids"].unsqueeze(0).to(teacher_device)
        with torch.no_grad():
            logits_out = teacher(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=keep,
            )
        logits = logits_out.logits.squeeze(0).detach().float()
        top_values, top_indices = torch.topk(logits, k=min(args.top_k, logits.shape[-1]), dim=-1)
        row["top_values"] = top_values.cpu().to(torch.float16)
        row["top_indices"] = top_indices.cpu().to(torch.int32)
        for layer_id in layers:
            item = captured[row_offset][layer_id]
            h_in = item["h_in"].unsqueeze(0).to(teacher_device)
            cos = item["cos"].unsqueeze(0).to(teacher_device)
            sin = item["sin"].unsqueeze(0).to(teacher_device)
            position_ids = item["position_ids"]
            if position_ids is not None:
                position_ids = position_ids.unsqueeze(0).to(teacher_device)
            with torch.no_grad():
                target = layer_module(teacher, layer_id)(
                    h_in,
                    position_embeddings=(cos, sin),
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
            target_hidden = target[0] if isinstance(target, tuple) else target
            item["target_last"] = target_hidden[:, -keep:, :].detach().cpu().to(torch.bfloat16).squeeze(0).contiguous()
            print(json.dumps({"event": "teacher_block_labeled", "row": row["index"], "layer": layer_id, "positions": keep}), flush=True)

    payload = {
        "format": "qwen_onpolicy_block_cache_v1",
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "weight_patches": [str(path) for path in args.weight_patch or []],
        "patch_reports": patch_reports,
        "layers": layers,
        "top_k": args.top_k,
        "format_token_ids": token_ids_for_strings(
            tokenizer,
            ["<think>", "</think>", "<predicted_observation>", "</predicted_observation>", "\n", "[", "]", "{", "}", '"'],
        ),
        "collapse_token_ids": token_ids_for_strings(tokenizer, ["<think>", "assistant", "\n\n", "\n"]),
        "rows": rows,
        "blocks": {int(row_index): dict(layer_items) for row_index, layer_items in captured.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps({"format": payload["format"], "rows": rows_to_json(rows), "layers": layers}, indent=2) + "\n"
    )
    print(json.dumps({"out": str(args.out), "rows": len(rows), "layers": layers}, indent=2))


if __name__ == "__main__":
    main()
