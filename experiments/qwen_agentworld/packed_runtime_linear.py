"""Packed-aware module replacement for Qwen AgentWorld smoke tests.

This is a correctness/runtime bridge, not an optimized low-bit GEMM kernel. It
can replace whole-tensor ``nn.Linear`` modules with modules that source weights
from the parent-pack manifest and keep no trainable BF16 weight parameter.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake_quant_policy import _parse_csv, resolve_parameter_name
from packed_manifest_loader import iter_manifest_groups
from verify_parent_pack_manifest import reconstruct_group


class PackedLinearRuntime(torch.nn.Module):
    """Linear layer backed by one packed-manifest whole-tensor group."""

    def __init__(
        self,
        *,
        artifact_arrays: dict[str, np.ndarray],
        group: dict[str, Any],
        dtype: torch.dtype,
        device: torch.device,
        cache: str,
    ) -> None:
        super().__init__()
        if cache not in {"none", "cpu", "gpu", "active", "active_lazy"}:
            raise ValueError(f"unsupported cache mode: {cache}")
        self.artifact_arrays = artifact_arrays
        self.group = group
        self.out_features = int(group["shape"][0])
        self.in_features = int(group["shape"][1])
        self.dtype = dtype
        self.cache = cache
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        self._cpu_weight: torch.Tensor | None = None
        self._gpu_weight: torch.Tensor | None = None
        if cache in {"cpu", "active"}:
            self._cpu_weight = self._reconstruct().to(dtype=dtype, device="cpu")
        elif cache == "gpu":
            self._gpu_weight = self._reconstruct().to(dtype=dtype, device=device)

    def _reconstruct(self) -> torch.Tensor:
        return reconstruct_group(self.artifact_arrays, self.group)

    def _weight_for(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache == "gpu":
            assert self._gpu_weight is not None
            return self._gpu_weight.to(device=x.device, dtype=x.dtype)
        if self.cache == "cpu":
            assert self._cpu_weight is not None
            return self._cpu_weight.to(device=x.device, dtype=x.dtype)
        if self.cache == "active":
            assert self._cpu_weight is not None
            return self._cpu_weight.to(device=x.device, dtype=x.dtype)
        return self._reconstruct().to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._weight_for(x), None)


class PackedMoeExpertsRuntime(torch.nn.Module):
    """Qwen3.5-MoE expert module backed by packed parent tensors.

    The official Transformers fallback implementation stores all expert
    weights in two fused parameters:

    - ``gate_up_proj``: [num_experts, 2 * intermediate_dim, hidden_dim]
    - ``down_proj``:    [num_experts, hidden_dim, intermediate_dim]

    This runtime keeps the same slow fallback routing loop but removes those
    two BF16 parameters from the module. In ``cache=cpu`` mode it stores
    dequantized expert slices on CPU and copies only hit experts to the active
    device during forward.
    """

    def __init__(
        self,
        *,
        source_module: torch.nn.Module,
        artifact_arrays: dict[str, dict[str, np.ndarray]],
        groups: dict[str, dict[int, dict[str, Any]]],
        dtype: torch.dtype,
        device: torch.device,
        cache: str,
        active_cache_limit: int,
        active_cpu_cache_limit: int = 0,
    ) -> None:
        super().__init__()
        if cache not in {"none", "cpu", "gpu", "active", "active_lazy"}:
            raise ValueError(f"unsupported cache mode: {cache}")
        self.num_experts = int(source_module.num_experts)
        self.hidden_dim = int(source_module.hidden_dim)
        self.intermediate_dim = int(source_module.intermediate_dim)
        self.act_fn = source_module.act_fn
        self.artifact_arrays = artifact_arrays
        self.groups = groups
        self.dtype = dtype
        self.cache = cache
        self.active_cache_limit = int(active_cache_limit)
        self.active_cpu_cache_limit = int(active_cpu_cache_limit)
        self.register_parameter("gate_up_proj", None)
        self.register_parameter("down_proj", None)
        self._cpu_weights: dict[str, torch.Tensor] = {}
        self._gpu_weights: dict[str, torch.Tensor] = {}
        self._active_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._lazy_cpu_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._lazy_cpu_hits = 0
        self._lazy_cpu_misses = 0
        self._lazy_cpu_evictions = 0
        self._peak_active_experts = 0
        self._peak_active_bytes = 0
        self._peak_lazy_cpu_experts = 0
        self._peak_lazy_cpu_bytes = 0
        if cache in {"cpu", "active"}:
            self._cpu_weights = {
                key: self._stack_weight(key).to(dtype=dtype, device="cpu")
                for key in ("gate_up_proj", "down_proj")
            }
        elif cache == "gpu":
            self._gpu_weights = {
                key: self._stack_weight(key).to(dtype=dtype, device=device)
                for key in ("gate_up_proj", "down_proj")
            }

    def _reconstruct_expert(self, key: str, expert_idx: int) -> torch.Tensor:
        group = self.groups[key][expert_idx]
        arrays = self.artifact_arrays[key]
        return reconstruct_group(arrays, group)

    def _stack_weight(self, key: str) -> torch.Tensor:
        return torch.stack(
            [self._reconstruct_expert(key, expert_idx) for expert_idx in range(self.num_experts)],
            dim=0,
        )

    def _expert_weight_for(self, key: str, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        if self.cache == "gpu":
            return self._gpu_weights[key][expert_idx].to(device=x.device, dtype=x.dtype)
        if self.cache == "cpu":
            return self._cpu_weights[key][expert_idx].to(device=x.device, dtype=x.dtype)
        if self.cache in {"active", "active_lazy"}:
            gate_up, down = self._active_weights_for(expert_idx, x)
            return gate_up if key == "gate_up_proj" else down
        return self._reconstruct_expert(key, expert_idx).to(device=x.device, dtype=x.dtype)

    def _active_weights_for(self, expert_idx: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._active_cache.get(expert_idx)
        if cached is not None:
            self._cache_hits += 1
            self._active_cache.move_to_end(expert_idx)
            return cached

        self._cache_misses += 1
        if self._cpu_weights:
            gate_up = self._cpu_weights["gate_up_proj"][expert_idx].to(device=x.device, dtype=x.dtype)
            down = self._cpu_weights["down_proj"][expert_idx].to(device=x.device, dtype=x.dtype)
        else:
            cpu_pair = self._lazy_cpu_weights_for(expert_idx)
            gate_up = cpu_pair[0].to(device=x.device, dtype=x.dtype)
            down = cpu_pair[1].to(device=x.device, dtype=x.dtype)
        self._active_cache[expert_idx] = (gate_up, down)
        self._active_cache.move_to_end(expert_idx)
        while self.active_cache_limit > 0 and len(self._active_cache) > self.active_cache_limit:
            self._active_cache.popitem(last=False)
            self._cache_evictions += 1
        self._peak_active_experts = max(self._peak_active_experts, len(self._active_cache))
        self._peak_active_bytes = max(self._peak_active_bytes, self._active_cache_bytes())
        return self._active_cache[expert_idx]

    def _lazy_cpu_weights_for(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._lazy_cpu_cache.get(expert_idx)
        if cached is not None:
            self._lazy_cpu_hits += 1
            self._lazy_cpu_cache.move_to_end(expert_idx)
            return cached

        self._lazy_cpu_misses += 1
        gate_up = self._reconstruct_expert("gate_up_proj", expert_idx).to(dtype=self.dtype, device="cpu")
        down = self._reconstruct_expert("down_proj", expert_idx).to(dtype=self.dtype, device="cpu")
        if self.active_cpu_cache_limit <= 0:
            return gate_up, down

        self._lazy_cpu_cache[expert_idx] = (gate_up, down)
        self._lazy_cpu_cache.move_to_end(expert_idx)
        while len(self._lazy_cpu_cache) > self.active_cpu_cache_limit:
            self._lazy_cpu_cache.popitem(last=False)
            self._lazy_cpu_evictions += 1
        self._peak_lazy_cpu_experts = max(self._peak_lazy_cpu_experts, len(self._lazy_cpu_cache))
        self._peak_lazy_cpu_bytes = max(self._peak_lazy_cpu_bytes, self._lazy_cpu_cache_bytes())
        return self._lazy_cpu_cache[expert_idx]

    def _active_cache_bytes(self) -> int:
        total = 0
        for gate_up, down in self._active_cache.values():
            total += int(gate_up.numel() * gate_up.element_size())
            total += int(down.numel() * down.element_size())
        return total

    def _lazy_cpu_cache_bytes(self) -> int:
        total = 0
        for gate_up, down in self._lazy_cpu_cache.values():
            total += int(gate_up.numel() * gate_up.element_size())
            total += int(down.numel() * down.element_size())
        return total

    def cache_stats(self) -> dict[str, Any]:
        return {
            "cache": self.cache,
            "active_cache_limit": self.active_cache_limit,
            "active_cpu_cache_limit": self.active_cpu_cache_limit,
            "active_experts": len(self._active_cache),
            "active_cache_bytes": self._active_cache_bytes(),
            "peak_active_experts": self._peak_active_experts,
            "peak_active_cache_bytes": self._peak_active_bytes,
            "lazy_cpu_experts": len(self._lazy_cpu_cache),
            "lazy_cpu_cache_bytes": self._lazy_cpu_cache_bytes(),
            "peak_lazy_cpu_experts": self._peak_lazy_cpu_experts,
            "peak_lazy_cpu_cache_bytes": self._peak_lazy_cpu_bytes,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "evictions": self._cache_evictions,
            "lazy_cpu_hits": self._lazy_cpu_hits,
            "lazy_cpu_misses": self._lazy_cpu_misses,
            "lazy_cpu_evictions": self._lazy_cpu_evictions,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0].item())
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up_weight = self._expert_weight_for("gate_up_proj", expert_idx, current_state)
            gate, up = F.linear(current_state, gate_up_weight).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            down_weight = self._expert_weight_for("down_proj", expert_idx, current_hidden_states)
            current_hidden_states = F.linear(current_hidden_states, down_weight)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def _module_parent(root: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str] | None:
    parts = module_name.split(".")
    if not parts:
        return None
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            return None
        parent = getattr(parent, part)
    return parent, parts[-1]


def _parameter_bytes(param: torch.nn.Parameter) -> int:
    return int(param.numel() * param.element_size())


def _load_npz_arrays(path: str, artifact_cache: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    arrays = artifact_cache.get(path)
    if arrays is not None:
        return arrays
    loaded = np.load(path)
    arrays = {name: loaded[name] for name in loaded.files}
    loaded.close()
    artifact_cache[path] = arrays
    return arrays


def collect_packed_runtime_stats(model: torch.nn.Module) -> dict[str, Any]:
    modules = []
    totals = {
        "active_experts": 0,
        "active_cache_bytes": 0,
        "peak_active_experts": 0,
        "peak_active_cache_bytes": 0,
        "lazy_cpu_experts": 0,
        "lazy_cpu_cache_bytes": 0,
        "peak_lazy_cpu_experts": 0,
        "peak_lazy_cpu_cache_bytes": 0,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "lazy_cpu_hits": 0,
        "lazy_cpu_misses": 0,
        "lazy_cpu_evictions": 0,
    }
    for name, module in model.named_modules():
        if not hasattr(module, "cache_stats"):
            continue
        stats = module.cache_stats()
        modules.append({"module": name, **stats})
        for key in totals:
            if key.startswith("peak_"):
                totals[key] = max(totals[key], int(stats.get(key, 0)))
            else:
                totals[key] += int(stats.get(key, 0))
    return {"modules": modules, "totals": totals}


@torch.no_grad()
def apply_packed_runtime_linears(
    model: torch.nn.Module,
    manifest_path: str | Path,
    *,
    max_groups: int | None = None,
    include_experts: bool = False,
    rates: str | None = None,
    buckets: str | None = None,
    cache: str = "cpu",
    active_cache_limit: int = 32,
    active_cpu_cache_limit: int = 0,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replace selected modules with packed-runtime modules."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    params = dict(model.named_parameters())
    modules = dict(model.named_modules())
    allowed_rates = _parse_csv(rates)
    allowed_buckets = _parse_csv(buckets)

    replaced = []
    replaced_experts = []
    skipped = []
    artifact_cache: dict[str, dict[str, np.ndarray]] = {}
    for parent_row, group in iter_manifest_groups(
        manifest,
        include_experts=False,
        allowed_rates=allowed_rates,
        allowed_buckets=allowed_buckets,
        max_groups=max_groups,
    ):
        if "expert" in group:
            continue
        tensor_name = group.get("tensor") or parent_row.get("tensor")
        parameter_name = resolve_parameter_name(str(tensor_name), params) if tensor_name else None
        if not parameter_name or not parameter_name.endswith(".weight"):
            skipped.append({"group": group.get("group"), "tensor": tensor_name, "reason": "weight parameter not found"})
            continue
        module_name = parameter_name[: -len(".weight")]
        module = modules.get(module_name)
        if not isinstance(module, torch.nn.Linear):
            skipped.append({
                "group": group.get("group"),
                "tensor": tensor_name,
                "parameter": parameter_name,
                "reason": f"module is not torch.nn.Linear: {type(module).__name__ if module else None}",
            })
            continue
        if module.bias is not None:
            skipped.append({"group": group.get("group"), "tensor": tensor_name, "parameter": parameter_name, "reason": "bias is not supported"})
            continue

        artifact_path = str(parent_row["artifact"])
        arrays = _load_npz_arrays(artifact_path, artifact_cache)

        parent = _module_parent(model, module_name)
        if parent is None:
            skipped.append({"group": group.get("group"), "tensor": tensor_name, "parameter": parameter_name, "reason": "module parent not found"})
            continue

        original_bytes = _parameter_bytes(module.weight)
        replacement = PackedLinearRuntime(
            artifact_arrays=arrays,
            group=group,
            dtype=module.weight.dtype,
            device=module.weight.device,
            cache=cache,
        )
        setattr(parent[0], parent[1], replacement)
        replaced.append(
            {
                "group": group.get("group"),
                "tensor": tensor_name,
                "parameter": parameter_name,
                "module": module_name,
                "rate": group.get("rate"),
                "bucket": group.get("bucket"),
                "params": int(group.get("params", module.weight.numel())),
                "removed_parameter_bytes": original_bytes,
                "artifact": artifact_path,
                "cache": cache,
            }
        )
        modules[module_name] = replacement
        params.pop(parameter_name, None)

    if include_experts:
        expert_candidates: dict[str, dict[str, Any]] = {}
        groups_seen = 0
        for parent_row in manifest["packed"]:
            tensor_name = str(parent_row.get("tensor"))
            if not tensor_name.endswith((".gate_up_proj", ".down_proj")):
                continue
            parent_groups = parent_row.get("groups") or []
            if not parent_groups or parent_groups[0].get("bucket") != "moe_expert":
                continue
            if allowed_buckets is not None and "moe_expert" not in allowed_buckets:
                continue
            if allowed_rates is not None and any(group.get("rate") not in allowed_rates for group in parent_groups):
                continue
            if max_groups is not None and groups_seen + len(parent_groups) > max_groups:
                break

            parameter_name = resolve_parameter_name(tensor_name, params)
            if parameter_name is None:
                skipped.append({"tensor": tensor_name, "reason": "expert parameter not found"})
                continue
            if parameter_name.endswith(".gate_up_proj"):
                expert_module_name = parameter_name[: -len(".gate_up_proj")]
                key = "gate_up_proj"
            elif parameter_name.endswith(".down_proj"):
                expert_module_name = parameter_name[: -len(".down_proj")]
                key = "down_proj"
            else:
                continue

            by_expert = {int(group["expert"]): group for group in parent_groups if "expert" in group}
            module = modules.get(expert_module_name)
            expected = int(getattr(module, "num_experts", 0) or 0)
            if expected <= 0 or len(by_expert) != expected:
                skipped.append({
                    "tensor": tensor_name,
                    "parameter": parameter_name,
                    "module": expert_module_name,
                    "reason": f"incomplete expert groups: {len(by_expert)} of {expected}",
                })
                continue

            candidate = expert_candidates.setdefault(
                expert_module_name,
                {
                    "module": module,
                    "parameter_names": {},
                    "removed_parameter_bytes": 0,
                    "artifact_arrays": {},
                    "groups": {},
                    "tensors": {},
                },
            )
            param = params[parameter_name]
            artifact_path = str(parent_row["artifact"])
            candidate["parameter_names"][key] = parameter_name
            candidate["removed_parameter_bytes"] += _parameter_bytes(param)
            candidate["artifact_arrays"][key] = _load_npz_arrays(artifact_path, artifact_cache)
            candidate["groups"][key] = by_expert
            candidate["tensors"][key] = tensor_name
            groups_seen += len(parent_groups)

        for module_name, candidate in sorted(expert_candidates.items()):
            if set(candidate["groups"]) != {"gate_up_proj", "down_proj"}:
                skipped.append({
                    "module": module_name,
                    "reason": f"incomplete expert module tensors: {sorted(candidate['groups'])}",
                })
                continue
            module = candidate["module"]
            if module is None:
                skipped.append({"module": module_name, "reason": "expert module not found"})
                continue
            parent = _module_parent(model, module_name)
            if parent is None:
                skipped.append({"module": module_name, "reason": "expert module parent not found"})
                continue

            dtype = module.gate_up_proj.dtype
            device = module.gate_up_proj.device
            replacement = PackedMoeExpertsRuntime(
                source_module=module,
                artifact_arrays=candidate["artifact_arrays"],
                groups=candidate["groups"],
                dtype=dtype,
                device=device,
                cache=cache,
                active_cache_limit=active_cache_limit,
                active_cpu_cache_limit=active_cpu_cache_limit,
            )
            setattr(parent[0], parent[1], replacement)
            for parameter_name in candidate["parameter_names"].values():
                params.pop(parameter_name, None)
            modules[module_name] = replacement
            replaced_experts.append(
                {
                    "module": module_name,
                    "tensors": candidate["tensors"],
                    "groups": {
                        key: len(value)
                        for key, value in candidate["groups"].items()
                    },
                    "rates": sorted({
                        group["rate"]
                        for rows in candidate["groups"].values()
                        for group in rows.values()
                    }),
                    "removed_parameter_bytes": int(candidate["removed_parameter_bytes"]),
                    "cache": cache,
                    "active_cache_limit": active_cache_limit,
                    "active_cpu_cache_limit": active_cpu_cache_limit,
                }
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "manifest": str(manifest_path),
        "requested_max_groups": max_groups,
        "rates": sorted(allowed_rates) if allowed_rates else None,
        "buckets": sorted(allowed_buckets) if allowed_buckets else None,
        "cache": cache,
        "active_cache_limit": active_cache_limit,
        "active_cpu_cache_limit": active_cpu_cache_limit,
        "include_experts": include_experts,
        "replaced_count": len(replaced),
        "replaced_expert_count": len(replaced_experts),
        "skipped_count": len(skipped),
        "removed_parameter_bytes": (
            sum(row["removed_parameter_bytes"] for row in replaced)
            + sum(row["removed_parameter_bytes"] for row in replaced_experts)
        ),
        "replaced": replaced,
        "replaced_experts": replaced_experts,
        "skipped": skipped,
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    return result
