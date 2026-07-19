"""Collect Qwen AgentWorld router utilization on AgentWorldBench prompts.

This is a non-judge, local-model calibration pass. It loads the BF16 teacher,
runs truncated AgentWorld prompts through the model, and records which MoE
experts the router chooses. The output feeds Clark Air expert-wise
rate-distortion allocation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def load_jobs(data_dir: Path, limit_per_domain: int | None) -> list[dict[str, Any]]:
    jobs = []
    for path in sorted(data_dir.glob("*_test.jsonl")):
        selected = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                jobs.append(json.loads(line))
                selected += 1
                if limit_per_domain is not None and selected >= limit_per_domain:
                    break
    return jobs


def messages_for_job(job: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = job.get("system_str", "")
    current_prompt = job.get("current_prompt", "")
    if not current_prompt:
        prompts = job.get("prompt", [])
        turn_idx = int(job.get("turn_idx", 1)) - 1
        if isinstance(prompts, list) and turn_idx < len(prompts):
            current_prompt = prompts[turn_idx]
        elif isinstance(prompts, str):
            current_prompt = prompts

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": current_prompt})
    return messages


def find_router_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    routers = {}
    for name, module in model.named_modules():
        if name.endswith(".mlp.gate") and hasattr(module, "weight"):
            routers[name] = module
    return routers


class RouterStats:
    def __init__(self, router_names: list[str], num_experts: int, top_k: int) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.counts = {name: torch.zeros(num_experts, dtype=torch.float64) for name in router_names}
        self.prob_sums = {name: torch.zeros(num_experts, dtype=torch.float64) for name in router_names}
        self.tokens = {name: 0 for name in router_names}
        self.calls = defaultdict(int)

    def update_selected(self, name: str, selected_experts: torch.Tensor, routing_weights: torch.Tensor | None = None) -> None:
        indices = selected_experts.detach().cpu().long().reshape(-1)
        valid = (indices >= 0) & (indices < self.num_experts)
        indices = indices[valid]
        if routing_weights is None:
            weights = torch.ones(indices.numel(), dtype=torch.float64)
        else:
            weights = routing_weights.detach().cpu().double().reshape(-1)[valid]

        flat_counts = torch.zeros(self.num_experts, dtype=torch.float64)
        flat_probs = torch.zeros(self.num_experts, dtype=torch.float64)
        flat_counts.scatter_add_(0, indices, torch.ones_like(weights, dtype=torch.float64))
        flat_probs.scatter_add_(0, indices, weights)

        self.counts[name].add_(flat_counts)
        self.prob_sums[name].add_(flat_probs)
        self.tokens[name] += int(selected_experts.reshape(-1, selected_experts.shape[-1]).shape[0])
        self.calls[name] += 1

    def update(self, name: str, logits: torch.Tensor | tuple) -> None:
        if isinstance(logits, tuple):
            if len(logits) >= 3 and torch.is_tensor(logits[2]):
                routing_weights = logits[1] if len(logits) >= 2 and torch.is_tensor(logits[1]) else None
                self.update_selected(name, logits[2], routing_weights)
                return
            logits = logits[0]

        values = logits.detach().float()
        if values.ndim == 3:
            values = values.reshape(-1, values.shape[-1])
        elif values.ndim != 2:
            values = values.reshape(-1, values.shape[-1])

        if values.shape[-1] != self.num_experts:
            return

        top = torch.topk(values, k=min(self.top_k, values.shape[-1]), dim=-1)
        probs = torch.softmax(top.values, dim=-1).cpu().double()
        indices = top.indices.cpu()

        flat_counts = torch.zeros(self.num_experts, dtype=torch.float64)
        flat_probs = torch.zeros(self.num_experts, dtype=torch.float64)
        ones = torch.ones(indices.numel(), dtype=torch.float64)
        flat_counts.scatter_add_(0, indices.reshape(-1), ones)
        flat_probs.scatter_add_(0, indices.reshape(-1), probs.reshape(-1))

        self.counts[name].add_(flat_counts)
        self.prob_sums[name].add_(flat_probs)
        self.tokens[name] += int(values.shape[0])
        self.calls[name] += 1

    def as_dict(self) -> dict[str, Any]:
        layers = {}
        for name in sorted(self.counts):
            match = LAYER_RE.search("." + name)
            layer = int(match.group(1)) if match else None
            total_routes = float(self.counts[name].sum().item())
            total_prob = float(self.prob_sums[name].sum().item())
            layers[str(layer) if layer is not None else name] = {
                "router_name": name,
                "tokens": self.tokens[name],
                "calls": int(self.calls[name]),
                "top_k": self.top_k,
                "expert_count": self.num_experts,
                "route_fraction": (self.counts[name] / max(total_routes, 1.0)).tolist(),
                "prob_fraction": (self.prob_sums[name] / max(total_prob, 1.0)).tolist(),
            }
        return {"num_experts": self.num_experts, "top_k": self.top_k, "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--data-dir", default="/workspace/AgentWorldBench_smoke_1x7", type=Path)
    parser.add_argument("--out", default="/workspace/agentworld_runs/router_calibration_smoke.json", type=Path)
    parser.add_argument("--limit-per-domain", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    routers = find_router_modules(model)
    if not routers:
        raise RuntimeError("no router modules found with suffix .mlp.gate")

    num_experts = int(getattr(model.config, "num_experts", 0) or next(iter(routers.values())).weight.shape[0])
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or getattr(model.config, "moe_top_k", 0) or 8)
    stats = RouterStats(sorted(routers), num_experts=num_experts, top_k=top_k)

    handles = []
    for name, module in routers.items():
        handles.append(module.register_forward_hook(lambda _m, _inp, out, router_name=name: stats.update(router_name, out)))

    jobs = load_jobs(args.data_dir, args.limit_per_domain)
    first_device = next(model.parameters()).device
    token_lengths = []
    tasks = []
    try:
        for index, job in enumerate(jobs, start=1):
            text = tokenizer.apply_chat_template(messages_for_job(job), tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=args.max_length)
            token_lengths.append(int(inputs["input_ids"].shape[-1]))
            tasks.append(job.get("task", "unknown"))
            inputs = {key: value.to(first_device) for key, value in inputs.items()}
            with torch.no_grad():
                model(**inputs, use_cache=False)
            print(f"[{index}/{len(jobs)}] task={job.get('task')} tokens={token_lengths[-1]}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    result = {
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "jobs": len(jobs),
        "tasks": tasks,
        "max_length": args.max_length,
        "token_lengths": token_lengths,
        "router": stats.as_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "jobs": len(jobs), "routers": len(routers)}, indent=2))


if __name__ == "__main__":
    main()
