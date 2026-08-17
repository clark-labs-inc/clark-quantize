"""Jointly QAT train Qwen SwiGLU MLP triplets from local activation cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from clark_quantize.qwen_qat import hard_ternary_ste


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def parse_layers(value: str, available: set[int]) -> list[int]:
    if value == "all":
        return sorted(available)
    rows = []
    for part in value.split(","):
        part = part.strip()
        if part:
            rows.append(int(part))
    return rows


def parse_row_indices(value: str, row_count: int, *, allow_empty: bool) -> list[int]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list(range(row_count))
    if normalized in {"", "none"}:
        if allow_empty:
            return []
        raise ValueError("row selection cannot be empty")
    selected: list[int] = []
    for part in value.split(","):
        index = int(part.strip())
        if not 0 <= index < row_count:
            raise ValueError(f"row index {index} is outside [0, {row_count})")
        if index in selected:
            raise ValueError(f"duplicate row index: {index}")
        selected.append(index)
    return selected


def activation_position_indices(
    rows: list[dict[str, Any]], selected_rows: list[int], total_positions: int
) -> torch.Tensor:
    selected = set(selected_rows)
    indices: list[torch.Tensor] = []
    offset = 0
    for row_index, row in enumerate(rows):
        positions = int(row["positions"])
        if positions <= 0:
            raise ValueError(f"activation row {row_index} has no positions")
        if row_index in selected:
            indices.append(torch.arange(offset, offset + positions, dtype=torch.long))
        offset += positions
    if offset != total_positions:
        raise ValueError(
            f"activation row metadata covers {offset} positions, expected {total_positions}"
        )
    if not indices:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(indices)


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
    def __init__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        *,
        mode: str,
        group_size: int,
        artifact_exact_ternary: bool = False,
    ) -> None:
        super().__init__()
        self.gate = torch.nn.Parameter(gate.detach().float())
        self.up = torch.nn.Parameter(up.detach().float())
        self.down = torch.nn.Parameter(down.detach().float())
        self.mode = mode
        self.group_size = group_size
        self.artifact_exact_ternary = artifact_exact_ternary

    def quantized(self, master: torch.Tensor) -> torch.Tensor:
        if self.mode == "ternary" and self.artifact_exact_ternary:
            # The current GGUF bridge materializes a BF16 patch before CQ3
            # packing. Round in the hard forward as well so CPU phase-1 trains
            # against the values the packer actually sees.
            return hard_ternary_ste(
                master, group_size=self.group_size, row_batch=128
            ).to(torch.bfloat16).float()
        return ste_quantized_weight(master, mode=self.mode, group_size=self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.quantized(self.gate))
        up = F.linear(x, self.quantized(self.up))
        return F.linear(F.silu(gate) * up, self.quantized(self.down))

    def codebook_loss(self) -> torch.Tensor:
        losses = []
        for master in (self.gate, self.up, self.down):
            with torch.no_grad():
                q = self.quantized(master).detach()
            denom = master.detach().float().square().mean().clamp_min(1e-8)
            losses.append((master.float() - q.float()).square().mean() / denom)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def hard_weights(self) -> dict[str, torch.Tensor]:
        return {
            "gate": self.quantized(self.gate).detach().cpu().to(torch.bfloat16),
            "up": self.quantized(self.up).detach().cpu().to(torch.bfloat16),
            "down": self.quantized(self.down).detach().cpu().to(torch.bfloat16),
        }


@torch.no_grad()
def normalized_reconstruction_loss(
    model: QuantMLPTriplet,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("validation tensors must be nonempty, equally shaped matrices")
    squared_error = 0.0
    squared_target = 0.0
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        prediction = model(x[start:stop])
        squared_error += float(
            (prediction.float() - y[start:stop].float()).square().sum().cpu()
        )
        squared_target += float(y[start:stop].float().square().sum().cpu())
    return squared_error / max(squared_target, 1e-12)


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
    validation_x: torch.Tensor | None = None,
    validation_y: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = QuantMLPTriplet(
        gate,
        up,
        down,
        mode=args.mode,
        group_size=args.group_size,
        artifact_exact_ternary=args.artifact_exact_ternary,
    ).to(device)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    if (validation_x is None) != (validation_y is None):
        raise ValueError("validation_x and validation_y must be supplied together")
    if validation_x is not None and validation_y is not None:
        validation_x = validation_x.to(device=device, dtype=torch.float32)
        validation_y = validation_y.to(device=device, dtype=torch.float32)
    validation_every = int(getattr(args, "validation_every", 1))
    if validation_every <= 0:
        raise ValueError("validation_every must be positive")
    validation_enabled = validation_x is not None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + layer_id)
    rows = x.shape[0]
    history = []
    validation_history: list[dict[str, Any]] = []
    best_validation_loss: float | None = None
    best_step: int | None = None
    best_hard_weights: dict[str, torch.Tensor] | None = None
    checkpoint_path = args.checkpoint_dir / f"layer-{layer_id}-latest.pt" if args.checkpoint_dir else None
    checkpoint_contract = {
        "layer": layer_id,
        "mode": args.mode,
        "group_size": args.group_size,
        "artifact_exact_ternary": args.artifact_exact_ternary,
        "steps_per_layer": args.steps_per_layer,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "code_weight": args.code_weight,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "activation_cache_sha256": args.activation_cache_sha256,
        "inventory_sha256": args.inventory_sha256,
        "base_patch_sha256": args.base_patch_sha256,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "validation_x_shape": list(validation_x.shape) if validation_x is not None else None,
        "validation_y_shape": list(validation_y.shape) if validation_y is not None else None,
        "validation_every": validation_every if validation_enabled else None,
        "train_row_indices": getattr(args, "train_row_indices_resolved", None),
        "validation_row_indices": getattr(args, "validation_row_indices_resolved", None),
    }
    completed_steps = 0
    resumed = False
    resumed_complete = False
    if checkpoint_path is not None and checkpoint_path.exists():
        if not args.resume:
            raise FileExistsError(f"checkpoint exists; pass --resume or choose a new directory: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("schema") != "qwen_mlp_triplet_qat_checkpoint_v1":
            raise ValueError(f"unsupported checkpoint schema: {checkpoint.get('schema')!r}")
        if checkpoint.get("contract") != checkpoint_contract:
            raise ValueError("checkpoint contract does not match this run")
        model.load_state_dict(checkpoint["model_state"])
        opt.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to_device(opt, device)
        generator.set_state(checkpoint["generator_state"])
        history = list(checkpoint.get("history", []))
        validation_history = list(checkpoint.get("validation_history", []))
        best_validation_loss = checkpoint.get("best_validation_loss")
        best_step = checkpoint.get("best_step")
        best_hard_weights = checkpoint.get("best_hard_weights")
        if validation_enabled and (
            best_validation_loss is None or best_step is None or best_hard_weights is None
        ):
            raise ValueError("validation checkpoint is missing best-artifact state")
        completed_steps = int(checkpoint["completed_steps"])
        resumed_complete = bool(checkpoint.get("complete", False))
        resumed = True
        if completed_steps > args.steps_per_layer:
            raise ValueError("checkpoint completed_steps exceeds steps_per_layer")
        print(
            json.dumps(
                {
                    "event": "resume",
                    "layer": layer_id,
                    "completed_steps": completed_steps,
                    "checkpoint": str(checkpoint_path),
                }
            ),
            flush=True,
        )

    def save_checkpoint(*, complete: bool) -> None:
        if checkpoint_path is None:
            return
        atomic_torch_save(
            {
                "schema": "qwen_mlp_triplet_qat_checkpoint_v1",
                "contract": checkpoint_contract,
                "completed_steps": completed_steps,
                "complete": complete,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer_state": opt.state_dict(),
                "generator_state": generator.get_state(),
                "history": history,
                "validation_history": validation_history,
                "best_validation_loss": best_validation_loss,
                "best_step": best_step,
                "best_hard_weights": best_hard_weights,
            },
            checkpoint_path,
        )

    if validation_enabled and not resumed:
        assert validation_x is not None and validation_y is not None
        best_validation_loss = normalized_reconstruction_loss(
            model, validation_x, validation_y, batch_size=args.batch_size
        )
        best_step = 0
        best_hard_weights = model.hard_weights()
        validation_history.append({"step": 0, "normalized_loss": best_validation_loss})
        print(
            json.dumps(
                {
                    "layer": layer_id,
                    "step": 0,
                    "validation_normalized_loss": best_validation_loss,
                    "selected_best": True,
                }
            ),
            flush=True,
        )

    for step in range(completed_steps, args.steps_per_layer):
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
        completed_steps = step + 1
        if validation_enabled and (
            completed_steps % validation_every == 0
            or completed_steps == args.steps_per_layer
        ):
            assert validation_x is not None and validation_y is not None
            validation_loss = normalized_reconstruction_loss(
                model, validation_x, validation_y, batch_size=args.batch_size
            )
            selected_best = (
                best_validation_loss is None or validation_loss < best_validation_loss
            )
            if selected_best:
                best_validation_loss = validation_loss
                best_step = completed_steps
                best_hard_weights = model.hard_weights()
            validation_history.append(
                {"step": completed_steps, "normalized_loss": validation_loss}
            )
            print(
                json.dumps(
                    {
                        "layer": layer_id,
                        "step": completed_steps,
                        "validation_normalized_loss": validation_loss,
                        "selected_best": selected_best,
                    }
                ),
                flush=True,
            )
        if args.checkpoint_every and completed_steps % args.checkpoint_every == 0:
            save_checkpoint(complete=False)
    if not (resumed_complete and completed_steps == args.steps_per_layer):
        save_checkpoint(complete=True)
    checkpoint_sha256 = sha256_file(checkpoint_path) if checkpoint_path is not None else None
    selected_hard = best_hard_weights if validation_enabled else model.hard_weights()
    assert selected_hard is not None
    return selected_hard, {
        "layer": layer_id,
        "samples": rows,
        "validation_samples": int(validation_x.shape[0]) if validation_x is not None else 0,
        "history": history,
        "validation_history": validation_history,
        "selected_best_step": best_step,
        "selected_best_validation_loss": best_validation_loss,
        "resumed": resumed,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
    }


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
    parser.add_argument("--artifact-exact-ternary", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--steps-per-layer", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--code-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--train-row-indices", default="all")
    parser.add_argument("--validation-row-indices", default="none")
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if args.torch_threads < 0:
        raise ValueError("torch-threads must be non-negative")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint-every must be non-negative")
    if args.validation_every <= 0:
        raise ValueError("validation-every must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    args.activation_cache_sha256 = sha256_file(args.activation_cache)
    args.inventory_sha256 = sha256_file(args.inventory)
    args.base_patch_sha256 = sha256_file(args.base_patch) if args.base_patch else None

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

    cache_rows: list[dict[str, Any]] = activation_cache["rows"]
    train_row_indices = parse_row_indices(
        args.train_row_indices, len(cache_rows), allow_empty=False
    )
    validation_row_indices = parse_row_indices(
        args.validation_row_indices, len(cache_rows), allow_empty=True
    )
    overlap = sorted(set(train_row_indices) & set(validation_row_indices))
    if overlap:
        raise ValueError(f"training and validation rows overlap: {overlap}")
    args.train_row_indices_resolved = train_row_indices
    args.validation_row_indices_resolved = validation_row_indices

    inventory = json.loads(args.inventory.read_text())
    patch = load_weight_patch(args.base_patch)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    output: dict[str, torch.Tensor] = {}
    reports = []
    for layer_id in selected:
        payload = layers_payload[layer_id]
        train_positions = activation_position_indices(
            cache_rows, train_row_indices, int(payload["x"].shape[0])
        )
        validation_positions = activation_position_indices(
            cache_rows, validation_row_indices, int(payload["x"].shape[0])
        )
        weights = {
            proj: load_tensor(args.model_dir, inventory, layer_tensor_name(layer_id, proj), patch)
            for proj in ("gate", "up", "down")
        }
        hard, report = train_layer(
            layer_id=layer_id,
            x=payload["x"].index_select(0, train_positions),
            y=payload["y"].index_select(0, train_positions),
            gate=weights["gate"],
            up=weights["up"],
            down=weights["down"],
            args=args,
            device=device,
            validation_x=(
                payload["x"].index_select(0, validation_positions)
                if validation_positions.numel()
                else None
            ),
            validation_y=(
                payload["y"].index_select(0, validation_positions)
                if validation_positions.numel()
                else None
            ),
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
        "artifact_exact_ternary": args.artifact_exact_ternary,
        "runtime_device": str(device),
        "torch_threads": torch.get_num_threads(),
        "activation_cache_sha256": args.activation_cache_sha256,
        "inventory_sha256": args.inventory_sha256,
        "base_patch_sha256": args.base_patch_sha256,
        "train_row_indices": train_row_indices,
        "validation_row_indices": validation_row_indices,
        "train_rows": [cache_rows[index] for index in train_row_indices],
        "validation_rows": [cache_rows[index] for index in validation_row_indices],
        "validation_every": args.validation_every,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "checkpoint_every": args.checkpoint_every,
        "reports": reports,
        "out_sha256": sha256_file(args.out),
    }
    report_path = args.report or args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "report": str(report_path), "layers": selected}, indent=2))


if __name__ == "__main__":
    main()
