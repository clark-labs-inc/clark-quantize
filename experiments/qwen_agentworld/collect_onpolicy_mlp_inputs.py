"""Collect MLP inputs from compressed Qwen on its own generated prefixes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake_quant_policy import apply_fake_quant_policy
from packed_manifest_loader import apply_packed_manifest
from collect_onpolicy_survival_cache import collapse_summary
from weight_patch import apply_weight_patch


MODEL_LOADER_CANDIDATES = (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForCausalLM",
    "AutoModelForVision2Seq",
)
LAYER_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp$")


def load_agentworld_eval(eval_dir: Path) -> Any:
    eval_path = eval_dir / "eval.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"AgentWorld eval.py not found: {eval_path}")
    sys.path.insert(0, str(eval_dir))
    spec = importlib.util.spec_from_file_location("agentworld_eval", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load import spec for {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_dir: Path) -> torch.nn.Module:
    errors = []
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
            )
            print(f"loaded_model loader={loader_name} class={model.__class__.__name__}", flush=True)
            model.eval()
            return model
        except Exception as exc:  # noqa: BLE001 - preserve loader fallback diagnostics.
            errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not load model:\n" + "\n".join(errors))


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def parse_layers(value: str) -> set[int] | None:
    if value == "all":
        return None
    out = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--eval-dir", default="/workspace/Qwen-AgentWorld/eval", type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weight-patch", action="append", type=Path)
    parser.add_argument("--packed-manifest", type=Path)
    parser.add_argument("--packed-report", type=Path)
    parser.add_argument("--fake-quant-policy", type=Path)
    parser.add_argument("--fake-quant-report", type=Path)
    parser.add_argument("--fake-quant-rates")
    parser.add_argument("--fake-quant-buckets")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-rows", type=int, default=10)
    parser.add_argument("--generate-tokens", type=int, default=256)
    parser.add_argument("--capture-positions", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.capture_positions <= 0:
        raise ValueError("--capture-positions must be positive")

    torch.manual_seed(args.seed)
    wanted_layers = parse_layers(args.layers)
    agentworld_eval = load_agentworld_eval(args.eval_dir)
    jobs = agentworld_eval.load_data(str(args.data_dir))[: args.max_rows]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.truncation_side = args.truncation_side
    model = load_model(args.model_dir)
    for param in model.parameters():
        param.requires_grad_(False)

    patch_reports = []
    for patch in args.weight_patch or []:
        patch_reports.append(apply_weight_patch(model, patch))
    packed_result = None
    if args.packed_manifest:
        packed_result = apply_packed_manifest(model, args.packed_manifest, out_path=args.packed_report)
    fake_quant_result = None
    if args.fake_quant_policy:
        fake_quant_result = apply_fake_quant_policy(
            model,
            args.fake_quant_policy,
            rates=args.fake_quant_rates,
            buckets=args.fake_quant_buckets,
            out_path=args.fake_quant_report,
            measure_rmse=False,
        )

    captured: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(lambda: {"x": []})
    current_keep = {"value": args.capture_positions}
    handles = []

    def make_hook(layer_id: int):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            keep = int(current_keep["value"])
            x = inputs[0]
            captured[layer_id]["x"].append(
                x.detach()[:, -keep:, :].reshape(-1, x.shape[-1]).cpu().to(torch.bfloat16)
            )

        return hook

    for name, module in model.named_modules():
        match = LAYER_RE.match(name)
        if not match:
            continue
        layer_id = int(match.group(1))
        if wanted_layers is not None and layer_id not in wanted_layers:
            continue
        handles.append(module.register_forward_hook(make_hook(layer_id)))
    if not handles:
        raise RuntimeError("no MLP hooks registered")

    device = first_parameter_device(model)
    rows = []
    try:
        for index, job in enumerate(jobs, start=1):
            text = tokenizer.apply_chat_template(
                messages_for_job(job),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_inputs = tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_model_len,
            )
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
            print(
                json.dumps(
                    {
                        "event": "generate_start",
                        "index": index,
                        "task": job.get("task"),
                        "id": job.get("id"),
                        "prompt_tokens": int(prompt_inputs["input_ids"].shape[-1]),
                    }
                ),
                flush=True,
            )
            with torch.no_grad():
                generated = model.generate(
                    **prompt_inputs,
                    max_new_tokens=args.generate_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, prompt_inputs["input_ids"].shape[-1] :]
            content = tokenizer.decode(new_tokens, skip_special_tokens=True)
            keep = min(args.capture_positions, int(generated.shape[-1]))
            current_keep["value"] = keep
            attention_mask = torch.ones_like(generated, device=device)
            with torch.no_grad():
                model(
                    input_ids=generated.to(device=device),
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )
            row = {
                "index": index,
                "task": job.get("task"),
                "id": job.get("id"),
                "prompt_tokens": int(prompt_inputs["input_ids"].shape[-1]),
                "generated_tokens": int(new_tokens.numel()),
                "capture_positions": keep,
                "generation": content,
                "collapse": collapse_summary(content),
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("index", "task", "id", "prompt_tokens", "generated_tokens", "collapse")}), flush=True)
    finally:
        for handle in handles:
            handle.remove()

    layers: dict[int, dict[str, torch.Tensor]] = {}
    for layer_id, tensors in captured.items():
        layers[layer_id] = {"x": torch.cat(tensors["x"], dim=0).contiguous()}

    payload = {
        "format": "qwen_onpolicy_mlp_input_cache_v1",
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "weight_patches": [str(path) for path in args.weight_patch or []],
        "patch_reports": patch_reports,
        "packed_manifest": str(args.packed_manifest) if args.packed_manifest else None,
        "packed_result": packed_result,
        "fake_quant_policy": str(args.fake_quant_policy) if args.fake_quant_policy else None,
        "fake_quant_result": fake_quant_result,
        "rows": rows,
        "layers": layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(json.dumps({"out": str(args.out), "rows": len(rows), "layers": sorted(layers)}, indent=2))


if __name__ == "__main__":
    main()
