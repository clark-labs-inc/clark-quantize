"""Jointly QAT train Qwen SwiGLU MLP triplets from local activation cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant.master_weight_qat import ste_quantized_weight


def parse_layers(value: str, available: set[int]) -> list[int]:
    if value == "all":
        return sorted(available)
    rows = []
    for part in value.split(","):
        part = part.strip()
        if part:
            rows.append(int(part))
    return rows


def patch_key_candidates(tensor_name: str) -> list[str]:
    candidates = [tensor_name]
    language_prefix = "model.language_model."
    model_prefix = "model."
    if tensor_name.startswith(language_prefix):
        candidates.append(model_prefix + tensor_name[len(language_prefix) :])
    elif tensor_name.startswith(model_prefix):
        candidates.append(language_prefix + tensor_name[len(model_prefix) :])
    return candidates


def load_weight_patch(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    rows: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            rows[key] = handle.get_tensor(key)
    return rows


def load_tensor(model_dir: Path, inventory: dict[str, Any], tensor_name: str, patch: dict[str, torch.Tensor]) -> torch.Tensor:
    for key in patch_key_candidates(tensor_name):
        if key in patch:
            return patch[key].contiguous()
    info = inventory["tensor_summary"][tensor_name]
    with safe_open(model_dir / info["shard"], framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name).contiguous()


def layer_tensor_name(layer_id: int, proj: str) -> str:
    return f"model.language_model.layers.{layer_id}.mlp.{proj}_proj.weight"


class QuantMLPTriplet(torch.nn.Module):
    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, *, mode: str, group_size: int) -> None:
        super().__init__()
        self.gate = torch.nn.Parameter(gate.detach().float())
        self.up = torch.nn.Parameter(up.detach().float())
        self.down = torch.nn.Parameter(down.detach().float())
        self.mode = mode
        self.group_size = group_size

    def quantized(self, master: torch.Tensor) -> torch.Tensor:
        return ste_quantized_weight(master, mode=self.mode, group_size=self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.quantized(self.gate))
        up = F.linear(x, self.quantized(self.up))
        return F.linear(F.silu(gate) * up, self.quantized(self.down))

    def codebook_loss(self) -> torch.Tensor:
        losses = []
        for master in (self.gate, self.up, self.down):
            with torch.no_grad():
                q = ste_quantized_weight(master, mode=self.mode, group_size=self.group_size).detach()
            denom = master.detach().float().square().mean().clamp_min(1e-8)
            losses.append((master.float() - q.float()).square().mean() / denom)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def hard_weights(self) -> dict[str, torch.Tensor]:
        return {
            "gate": ste_quantized_weight(self.gate, mode=self.mode, group_size=self.group_size).detach().cpu().to(torch.bfloat16),
            "up": ste_quantized_weight(self.up, mode=self.mode, group_size=self.group_size).detach().cpu().to(torch.bfloat16),
            "down": ste_quantized_weight(self.down, mode=self.mode, group_size=self.group_size).detach().cpu().to(torch.bfloat16),
        }


def train_layer(
    *,
    layer_id: int,
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = QuantMLPTriplet(gate, up, down, mode=args.mode, group_size=args.group_size).to(device)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + layer_id)
    rows = x.shape[0]
    history = []
    for step in range(args.steps_per_layer):
        index = torch.randint(rows, (min(args.batch_size, rows),), generator=generator)
        xb = x[index.to(device)]
        yb = y[index.to(device)]
        pred = model(xb)
        mse = (pred.float() - yb.float()).square().mean()
        denom = yb.float().square().mean().clamp_min(1e-8)
        local = mse / denom
        code = model.codebook_loss()
        loss = local + args.code_weight * code
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps_per_layer:
            row = {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "local_loss": float(local.detach().cpu()),
                "codebook_loss": float(code.detach().cpu()),
            }
            history.append(row)
            print(json.dumps({"layer": layer_id, **row}), flush=True)
    return model.hard_weights(), {"layer": layer_id, "samples": rows, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--inventory", default="/workspace/agentworld_runs/qwen35_9b_model_inventory.json", type=Path)
    parser.add_argument("--activation-cache", required=True, type=Path)
    parser.add_argument("--base-patch", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--mode", choices=["ternary", "binary"], default="ternary")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--steps-per-layer", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--code-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")

    activation_cache = torch.load(args.activation_cache, map_location="cpu")
    if activation_cache.get("format") != "qwen_mlp_activation_cache_v1":
        raise ValueError(f"unsupported activation cache format: {activation_cache.get('format')!r}")
    layers_payload: dict[int, dict[str, torch.Tensor]] = activation_cache["layers"]
    available = {int(k) for k in layers_payload}
    selected = parse_layers(args.layers, available)
    if args.max_layers > 0:
        selected = selected[: args.max_layers]
    if not selected:
        raise RuntimeError("no layers selected")

    inventory = json.loads(args.inventory.read_text())
    patch = load_weight_patch(args.base_patch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output: dict[str, torch.Tensor] = {}
    reports = []
    for layer_id in selected:
        payload = layers_payload[layer_id]
        weights = {
            proj: load_tensor(args.model_dir, inventory, layer_tensor_name(layer_id, proj), patch)
            for proj in ("gate", "up", "down")
        }
        hard, report = train_layer(
            layer_id=layer_id,
            x=payload["x"],
            y=payload["y"],
            gate=weights["gate"],
            up=weights["up"],
            down=weights["down"],
            args=args,
            device=device,
        )
        output[layer_tensor_name(layer_id, "gate")] = hard["gate"]
        output[layer_tensor_name(layer_id, "up")] = hard["up"]
        output[layer_tensor_name(layer_id, "down")] = hard["down"]
        reports.append(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(args.out))
    report_payload = {
        "out": str(args.out),
        "activation_cache": str(args.activation_cache),
        "base_patch": str(args.base_patch) if args.base_patch else None,
        "layers": selected,
        "mode": args.mode,
        "group_size": args.group_size,
        "reports": reports,
    }
    report_path = args.report or args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "report": str(report_path), "layers": selected}, indent=2))


if __name__ == "__main__":
    main()
