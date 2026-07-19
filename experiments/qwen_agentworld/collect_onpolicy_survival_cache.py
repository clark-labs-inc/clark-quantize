"""Collect teacher labels on low-bit Qwen's own AgentWorld prefixes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake_quant_policy import apply_fake_quant_policy
from packed_manifest_loader import apply_packed_manifest
from weight_patch import apply_weight_patch


MODEL_LOADER_CANDIDATES = (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForCausalLM",
    "AutoModelForVision2Seq",
)
BLOCK_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)$")


def load_agentworld_eval(eval_dir: Path) -> Any:
    sys.path.insert(0, str(eval_dir))
    spec = importlib.util.spec_from_file_location("agentworld_eval", eval_dir / "eval.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load AgentWorld eval from {eval_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_dir: Path) -> torch.nn.Module:
    errors = []
    extra_kwargs: dict[str, Any] = {}
    attention_impl = os.environ.get("CLARK_ATTENTION_IMPLEMENTATION")
    if attention_impl:
        extra_kwargs["attn_implementation"] = attention_impl
    for loader_name in MODEL_LOADER_CANDIDATES:
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                **extra_kwargs,
            )
            print(f"loaded_model loader={loader_name} class={model.__class__.__name__}", flush=True)
            model.eval()
            return model
        except Exception as exc:  # noqa: BLE001 - preserve fallback diagnostics.
            errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not load model:\n" + "\n".join(errors))


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def parse_layers(value: str) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def messages_for_job(job: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = job.get("system_str", "")
    current_prompt = job.get("current_prompt", "")
    if not current_prompt:
        prompts = job.get("prompt", [])
        turn_idx = job.get("turn_idx", 1) - 1
        if isinstance(prompts, list) and turn_idx < len(prompts):
            current_prompt = prompts[turn_idx]
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": current_prompt})
    return messages


def remove_thinking_tags(text: str, response_tag: str = "predicted_observation") -> str:
    """Mirror AgentWorld's robust think-tag stripping for local smoke gates."""
    if not text:
        return text

    tags = []
    for match in re.finditer(r"<think>", text, re.IGNORECASE):
        tags.append((match.start(), "open"))
    for match in re.finditer(r"</think>", text, re.IGNORECASE):
        tags.append((match.start(), "close"))
    if not tags:
        return text
    tags.sort(key=lambda item: item[0])

    remove_ranges: list[tuple[int, int]] = []
    used: set[int] = set()
    for index, (pos, tag_type) in enumerate(tags):
        if index in used or tag_type != "open":
            continue
        for close_index in range(index + 1, len(tags)):
            if close_index in used:
                continue
            close_pos, close_type = tags[close_index]
            if close_type == "close":
                remove_ranges.append((pos, close_pos + len("</think>")))
                used.add(index)
                used.add(close_index)
                break

    for index, (pos, tag_type) in enumerate(tags):
        if index in used:
            continue
        if tag_type == "close":
            remove_ranges.append((0, pos + len("</think>")))
            used.add(index)

    for index, (pos, tag_type) in enumerate(tags):
        if index in used or tag_type != "open":
            continue
        end_pos = len(text)
        if response_tag:
            pattern = rf"<{re.escape(response_tag)}>"
            match = re.search(pattern, text[pos:], re.IGNORECASE)
            if match:
                end_pos = pos + match.start()
        remove_ranges.append((pos, end_pos))
        used.add(index)

    if not remove_ranges:
        return text
    remove_ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in remove_ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    pieces = []
    prev_end = 0
    for start, end in merged:
        if start > prev_end:
            pieces.append(text[prev_end:start])
        prev_end = end
    if prev_end < len(text):
        pieces.append(text[prev_end:])
    return "".join(pieces).strip()


def parse_agentworld_model_output(raw_output: str, response_tag: str = "predicted_observation") -> str:
    """Local copy of AgentWorld's parse_model_output behavior."""
    if not raw_output:
        return "No output"
    cleaned = remove_thinking_tags(raw_output, response_tag)
    start_pattern = rf"<{re.escape(response_tag)}>"
    start_matches = list(re.finditer(start_pattern, cleaned, re.IGNORECASE))
    if not start_matches:
        return cleaned.strip() if cleaned.strip() else "No output"

    start_pos = start_matches[-1].end()
    close_pattern = rf"</{re.escape(response_tag)}>"
    close_match = re.search(close_pattern, cleaned[start_pos:], re.IGNORECASE)
    if close_match:
        end_pos = start_pos + close_match.start()
        return cleaned[start_pos:end_pos].strip()
    return cleaned[start_pos:].strip()


def char_fractions(text: str) -> tuple[float, float]:
    length = max(len(text), 1)
    alnum = sum(1 for char in text if char.isalnum())
    newline = text.count("\n")
    return alnum / length, newline / length


def looks_repetitive(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9_<>/]+|[^\s]", text.lower())
    if len(tokens) < 12:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    most_common = max(counts.values())
    if most_common / len(tokens) >= 0.35:
        return True
    for width in (2, 3):
        if len(tokens) < width * 4:
            continue
        ngrams = [" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
        ngram_counts: dict[str, int] = {}
        for item in ngrams:
            ngram_counts[item] = ngram_counts.get(item, 0) + 1
        if max(ngram_counts.values()) / len(ngrams) >= 0.25:
            return True
    return False


def looks_structural_junk(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    special = sum(1 for char in stripped if char in '<>{}[]"\\=/')
    special_fraction = special / max(len(stripped), 1)
    escaped_quote_count = stripped.count('\\"') + stripped.count('""')
    words = re.findall(r"[A-Za-z]+", stripped.lower())
    if not words:
        return special_fraction > 0.25
    junk_words = {
        "action",
        "actionback",
        "address",
        "content",
        "data",
        "description",
        "div",
        "invoke",
        "name",
        "namename",
        "null",
        "parameter",
        "sub",
        "text",
        "times",
        "title",
    }
    junk_ratio = sum(1 for word in words if word in junk_words) / max(len(words), 1)
    low_diversity = len(set(words)) <= 12 and len(words) >= 10
    return (
        (special_fraction > 0.25 and junk_ratio > 0.25)
        or (escaped_quote_count >= 8 and junk_ratio > 0.15)
        or (low_diversity and junk_ratio > 0.45)
    )


def looks_low_content(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    words = re.findall(r"[A-Za-z0-9_]+", stripped.lower())
    if len(stripped) < 32 and len(words) <= 4:
        return True
    generic = {
        "assistant",
        "the",
        "no",
        "none",
        "null",
        "content",
        "data",
        "index",
    }
    if words and len(words) <= 8 and sum(1 for word in words if word in generic) / len(words) >= 0.5:
        return True
    return False


def looks_gibberish(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80:
        return False
    words = re.findall(r"[A-Za-z]+", stripped.lower())
    if len(words) < 12:
        return True
    common_junk = {
        "a",
        "are",
        "cent",
        "data",
        "he",
        "have",
        "is",
        "language",
        "life",
        "name",
        "of",
        "the",
        "time",
        "true",
        "you",
    }
    junk_word_ratio = sum(1 for word in words if word in common_junk) / max(len(words), 1)
    long_word_ratio = sum(1 for word in words if len(word) >= 5 and word not in common_junk) / max(len(words), 1)
    underscore_fraction = stripped.count("_") / max(len(stripped), 1)
    punctuation = sum(1 for char in stripped if char in '.,:;()<>[]{}"`_=/\\')
    punctuation_fraction = punctuation / max(len(stripped), 1)
    non_ascii_fraction = sum(1 for char in stripped if ord(char) > 127) / max(len(stripped), 1)
    return (
        (junk_word_ratio > 0.45 and long_word_ratio < 0.12)
        or (underscore_fraction > 0.08 and long_word_ratio < 0.18)
        or (punctuation_fraction > 0.28 and long_word_ratio < 0.12)
        or (non_ascii_fraction > 0.12 and long_word_ratio < 0.12)
    )


def collapse_summary(text: str) -> dict[str, Any]:
    lowered = text.lower()
    length = len(text)
    think = lowered.count("<think")
    valid_tagged = "<predicted_observation>" in lowered and "</predicted_observation>" in lowered
    extracted = parse_agentworld_model_output(text)
    official_parseable = bool(extracted.strip()) and extracted.strip() != "No output"
    has_response_marker = extracted.lstrip().lower().startswith("**environment observation:**")
    alnum_fraction, newline_fraction = char_fractions(text)
    extracted_alnum_fraction, extracted_newline_fraction = char_fractions(extracted)

    if valid_tagged and official_parseable:
        kind = "valid_tagged"
    elif not official_parseable:
        kind = "think_loop" if think else "empty_output"
    elif newline_fraction > 0.5 and alnum_fraction < 0.2:
        kind = "newline_collapse"
    elif length < 8:
        kind = "premature_stop"
    elif extracted_alnum_fraction < 0.1 and len(extracted) >= 8:
        kind = "punctuation_collapse"
    elif has_response_marker:
        kind = "parseable_output"
    elif looks_low_content(extracted):
        kind = "low_content"
    elif looks_structural_junk(extracted):
        kind = "structural_junk"
    elif looks_gibberish(extracted):
        kind = "gibberish"
    elif looks_repetitive(extracted):
        kind = "repetition_collapse"
    else:
        kind = "parseable_output"

    return {
        "chars": length,
        "alnum_fraction": alnum_fraction,
        "newline_fraction": newline_fraction,
        "think_count": think,
        "valid_observation": valid_tagged,
        "official_parseable": official_parseable,
        "extracted_chars": len(extracted),
        "extracted_alnum_fraction": extracted_alnum_fraction,
        "extracted_newline_fraction": extracted_newline_fraction,
        "extracted_preview": extracted[:240],
        "has_response_marker": has_response_marker,
        "failure_kind": kind,
    }


def token_ids_for_strings(tokenizer: Any, values: list[str]) -> list[int]:
    ids: set[int] = set()
    for value in values:
        encoded = tokenizer.encode(value, add_special_tokens=False)
        ids.update(int(token_id) for token_id in encoded)
    return sorted(ids)


def register_hidden_hooks(model: torch.nn.Module, layers: list[int], keep_ref: dict[str, int]) -> tuple[list[Any], dict[int, torch.Tensor]]:
    captured: dict[int, torch.Tensor] = {}
    wanted = set(layers)
    handles = []

    def make_hook(layer_id: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            keep = int(keep_ref["keep"])
            captured[layer_id] = hidden.detach()[:, -keep:, :].squeeze(0).cpu().to(torch.bfloat16)

        return hook

    for name, module in model.named_modules():
        match = BLOCK_RE.match(name)
        if not match:
            continue
        layer_id = int(match.group(1))
        if layer_id in wanted:
            handles.append(module.register_forward_hook(make_hook(layer_id)))
    if wanted and len(handles) != len(wanted):
        raise RuntimeError(f"registered {len(handles)} hidden hooks, wanted {len(wanted)}")
    return handles, captured


def apply_student_overrides(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    reports: dict[str, Any] = {"weight_patches": []}
    for patch in args.weight_patch or []:
        reports["weight_patches"].append(apply_weight_patch(model, patch))
    if args.fake_quant_policy:
        reports["fake_quant"] = apply_fake_quant_policy(
            model,
            args.fake_quant_policy,
            rates=args.fake_quant_rates,
            buckets=args.fake_quant_buckets,
            out_path=args.fake_quant_report,
            measure_rmse=False,
        )
    if args.packed_manifest:
        reports["packed_manifest"] = apply_packed_manifest(
            model,
            args.packed_manifest,
            out_path=args.packed_report,
        )
    if args.lift_lora:
        reports["lift_lora"] = apply_lift_project_lora(model, args)
    return reports


@torch.no_grad()
def apply_lift_project_lora(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    from safetensors import safe_open

    from train_lift_project_qat import attach_lora, parse_int_csv, parameter_dtype, set_projection_alpha

    args.train_layer_set = parse_int_csv(args.lift_train_layers)
    args.train_regex = args.lift_train_regex
    args.train_head = args.lift_train_head
    args.train_embed = args.lift_train_embed
    args.rank_mlp = args.lift_rank_mlp
    args.rank_attention = args.lift_rank_attention
    args.rank_head = args.lift_rank_head
    args.rank_embed = args.lift_rank_embed
    args.lora_alpha = args.lift_lora_alpha
    args.lora_dtype = args.lift_lora_dtype
    args.project_mode = args.lift_project_mode
    args.project_mlp_mode = args.lift_project_mlp_mode
    args.project_attention_mode = args.lift_project_attention_mode
    args.project_mlp_group_size = args.lift_project_mlp_group_size
    args.project_attention_group_size = args.lift_project_attention_group_size
    args.projection_aware = args.lift_projection_aware
    args.use_dora = args.lift_use_dora
    args.low_memory_projection = args.lift_low_memory_projection
    args.group_size = args.lift_group_size

    modules = attach_lora(model, args)
    set_projection_alpha(modules, args.lift_projection_alpha)
    module_names = set(modules)
    loaded = []
    missing = []
    unexpected = []
    lora_path = Path(args.lift_lora)
    with safe_open(lora_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for name, module in modules.items():
            for suffix in ("lora_a", "lora_b"):
                key = f"{name}.{suffix}"
                if key not in keys:
                    missing.append(key)
                    continue
                param = getattr(module, suffix)
                tensor = handle.get_tensor(key)
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded.append(key)
            key = f"{name}.log_mag"
            if hasattr(module, "log_mag") and key in keys:
                param = getattr(module, "log_mag")
                tensor = handle.get_tensor(key)
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded.append(key)
        for key in keys:
            root = key.rsplit(".", 1)[0]
            if root not in module_names:
                unexpected.append(key)

    dtype = parameter_dtype(args.lift_lora_dtype)
    return {
        "path": str(lora_path),
        "attached_modules": len(modules),
        "loaded_tensors": len(loaded),
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "lora_dtype": str(dtype),
        "projection_aware": bool(args.lift_projection_aware),
        "projection_alpha": float(args.lift_projection_alpha),
        "use_dora": bool(args.lift_use_dora),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--eval-dir", default="/workspace/Qwen-AgentWorld/eval", type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weight-patch", action="append", type=Path)
    parser.add_argument("--fake-quant-policy", type=Path)
    parser.add_argument("--fake-quant-report", type=Path)
    parser.add_argument("--fake-quant-rates")
    parser.add_argument("--fake-quant-buckets")
    parser.add_argument("--packed-manifest", type=Path)
    parser.add_argument("--packed-report", type=Path)
    parser.add_argument("--lift-lora", type=Path)
    parser.add_argument("--lift-rank-mlp", type=int, default=8)
    parser.add_argument("--lift-rank-attention", type=int, default=4)
    parser.add_argument("--lift-rank-head", type=int, default=0)
    parser.add_argument("--lift-rank-embed", type=int, default=0)
    parser.add_argument("--lift-lora-alpha", type=float, default=16.0)
    parser.add_argument("--lift-lora-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--lift-train-layers", default="")
    parser.add_argument("--lift-train-regex")
    parser.add_argument("--lift-train-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lift-train-embed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lift-project-mode", choices=("ternary", "binary"), default="ternary")
    parser.add_argument("--lift-project-mlp-mode", choices=("ternary", "binary"))
    parser.add_argument("--lift-project-attention-mode", choices=("ternary", "binary"))
    parser.add_argument("--lift-project-mlp-group-size", type=int)
    parser.add_argument("--lift-project-attention-group-size", type=int)
    parser.add_argument("--lift-projection-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lift-projection-alpha", type=float, default=1.0)
    parser.add_argument("--lift-use-dora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lift-low-memory-projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lift-group-size", type=int, default=128)
    parser.add_argument("--hidden-layers", default="0,1,10,12,27,28,30,31")
    parser.add_argument("--max-rows", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--generate-tokens", type=int, default=192)
    parser.add_argument("--label-positions", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    hidden_layers = parse_layers(args.hidden_layers)
    agentworld_eval = load_agentworld_eval(args.eval_dir)
    jobs = agentworld_eval.load_data(str(args.data_dir))[: args.max_rows]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.truncation_side = args.truncation_side

    student = load_model(args.model_dir)
    for param in student.parameters():
        param.requires_grad_(False)
    student_reports = apply_student_overrides(student, args)
    device = first_parameter_device(student)
    generated_rows = []
    for index, job in enumerate(jobs, start=1):
        text = tokenizer.apply_chat_template(
            messages_for_job(job),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=args.max_model_len)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        print(
            json.dumps(
                {
                    "event": "student_generate_start",
                    "index": index,
                    "task": job.get("task"),
                    "id": job.get("id"),
                    "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                }
            ),
            flush=True,
        )
        with torch.no_grad():
            output = student.generate(
                **inputs,
                max_new_tokens=args.generate_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        new_tokens = output[0, prompt_tokens:]
        generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        keep = max(1, min(args.label_positions, int(new_tokens.numel()), int(output.shape[-1])))
        row = {
            "index": index,
            "task": job.get("task"),
            "id": job.get("id"),
            "turn_idx": job.get("turn_idx"),
            "total_turns": job.get("total_turns"),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(new_tokens.numel()),
            "tokens": int(output.shape[-1]),
            "keep": keep,
            "input_ids": output.squeeze(0).cpu().to(torch.long),
            "attention_mask": torch.ones(int(output.shape[-1]), dtype=torch.long),
            "generation": generation,
            "collapse": collapse_summary(generation),
        }
        generated_rows.append(row)
        print(json.dumps({k: row[k] for k in ("index", "task", "id", "tokens", "keep", "collapse")}), flush=True)
    del student
    torch.cuda.empty_cache()

    teacher = load_model(args.model_dir)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher_device = first_parameter_device(teacher)
    keep_ref = {"keep": args.label_positions}
    handles, hidden_capture = register_hidden_hooks(teacher, hidden_layers, keep_ref)
    try:
        for row in generated_rows:
            keep_ref["keep"] = int(row["keep"])
            hidden_capture.clear()
            model_inputs = {
                "input_ids": row["input_ids"].unsqueeze(0).to(teacher_device),
                "attention_mask": row["attention_mask"].unsqueeze(0).to(teacher_device),
            }
            with torch.no_grad():
                outputs = teacher(
                    **model_inputs,
                    use_cache=False,
                    logits_to_keep=int(row["keep"]),
                )
            logits = outputs.logits.squeeze(0).detach().float()
            top_values, top_indices = torch.topk(logits, k=min(args.top_k, logits.shape[-1]), dim=-1)
            row["top_values"] = top_values.cpu().to(torch.float16)
            row["top_indices"] = top_indices.cpu().to(torch.int32)
            row["teacher_hidden"] = {layer: hidden_capture[layer].contiguous() for layer in hidden_layers}
            print(
                json.dumps(
                    {
                        "event": "teacher_labeled",
                        "index": row["index"],
                        "positions": int(row["top_values"].shape[0]),
                        "top_k": int(row["top_values"].shape[1]),
                    }
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    format_token_ids = token_ids_for_strings(
        tokenizer,
        [
            "<think>",
            "</think>",
            "<predicted_observation>",
            "</predicted_observation>",
            "\n",
            "[",
            "]",
            "{",
            "}",
            '"',
        ],
    )
    collapse_token_ids = token_ids_for_strings(tokenizer, ["<think>", "assistant", "\n\n", "\n"])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        format_token_ids = sorted({*format_token_ids, int(eos_id)})

    payload = {
        "format": "qwen_onpolicy_survival_cache_v1",
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "student_reports": student_reports,
        "weight_patches": [str(path) for path in args.weight_patch or []],
        "fake_quant_policy": str(args.fake_quant_policy) if args.fake_quant_policy else None,
        "packed_manifest": str(args.packed_manifest) if args.packed_manifest else None,
        "max_model_len": args.max_model_len,
        "generate_tokens": args.generate_tokens,
        "label_positions": args.label_positions,
        "top_k": args.top_k,
        "hidden_layers": hidden_layers,
        "format_token_ids": format_token_ids,
        "collapse_token_ids": collapse_token_ids,
        "rows": generated_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(json.dumps({"out": str(args.out), "rows": len(generated_rows), "hidden_layers": hidden_layers}, indent=2))


if __name__ == "__main__":
    main()
