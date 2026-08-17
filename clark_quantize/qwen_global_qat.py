"""Exact, disk-bounded checkpoint state for global Qwen hard-ternary QAT."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any, Mapping

import numpy as np
import torch

from .qwen_qat import load_qat_checkpoint, save_qat_checkpoint


GLOBAL_QAT_STATE_SCHEMA = "clark_qwen35_global_qat_state_v1"


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def split_model_state(
    model_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split one complete state dict without duplicating master tensors."""
    masters: dict[str, torch.Tensor] = {}
    retained: dict[str, torch.Tensor] = {}
    for name, tensor in model_state.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"model state entry is not a tensor: {name}")
        target = masters if name.endswith("weight_master") else retained
        target[name] = tensor
    if not masters:
        raise ValueError("global QAT checkpoint has no weight_master tensors")
    if masters.keys() & retained.keys():
        raise AssertionError("model checkpoint split overlaps")
    return masters, retained


def join_model_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    masters = state.get("masters")
    retained = state.get("retained_model_state")
    if not isinstance(masters, dict) or not isinstance(retained, dict):
        raise ValueError("global QAT checkpoint model split is missing")
    if masters.keys() & retained.keys():
        raise ValueError("global QAT checkpoint model split overlaps")
    joined = {**retained, **masters}
    if not joined or not all(torch.is_tensor(value) for value in joined.values()):
        raise ValueError("global QAT checkpoint model state is invalid")
    return joined


def tensor_tree_nbytes(value: Any) -> int:
    """Count unique tensor storages in an arbitrary checkpoint tree."""
    seen: set[tuple[str, int, int]] = set()

    def visit(item: Any) -> int:
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            key = (str(item.device), int(storage.data_ptr()), int(storage.nbytes()))
            if key in seen:
                return 0
            seen.add(key)
            return int(storage.nbytes())
        if isinstance(item, Mapping):
            return sum(visit(child) for child in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(child) for child in item)
        return 0

    return visit(value)


def capture_replay_state(
    *,
    arm_rng_states: Mapping[str, dict[str, Any]],
    bandit_kl: Mapping[str, float],
    bandit_rng_state: object,
    servo_state: Mapping[str, Any],
    kl_history: list[tuple[float, float]],
) -> dict[str, Any]:
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": cuda_states,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "arm_rng_states": copy.deepcopy(dict(arm_rng_states)),
        "bandit_kl": {name: float(value) for name, value in bandit_kl.items()},
        "bandit_rng_state": copy.deepcopy(bandit_rng_state),
        "servo_state": copy.deepcopy(dict(servo_state)),
        "kl_history": [(float(u), float(loss)) for u, loss in kl_history],
    }


def validate_replay_state(value: Any, *, arms: list[str], cuda_devices: int) -> None:
    if not isinstance(value, dict):
        raise ValueError("global QAT replay state is missing")
    cpu = value.get("cpu_rng_state")
    cuda = value.get("cuda_rng_state")
    if (
        not torch.is_tensor(cpu)
        or cpu.dtype != torch.uint8
        or cpu.ndim != 1
        or not isinstance(cuda, list)
        or len(cuda) != cuda_devices
        or not all(
            torch.is_tensor(item) and item.dtype == torch.uint8 and item.ndim == 1
            for item in cuda
        )
    ):
        raise ValueError("global QAT torch RNG state is incomplete")
    arm_states = value.get("arm_rng_states")
    bandit = value.get("bandit_kl")
    if (
        not isinstance(arm_states, dict)
        or sorted(arm_states) != sorted(arms)
        or not isinstance(bandit, dict)
        or sorted(bandit) != sorted(arms)
    ):
        raise ValueError("global QAT arm replay state differs from the run arms")
    if not isinstance(value.get("servo_state"), dict):
        raise ValueError("global QAT servo replay state is missing")
    if not isinstance(value.get("kl_history"), list):
        raise ValueError("global QAT KL history is missing")


def restore_process_rng(replay: Mapping[str, Any], *, cuda_devices: int) -> None:
    torch.random.set_rng_state(replay["cpu_rng_state"])
    if cuda_devices:
        torch.cuda.set_rng_state_all(replay["cuda_rng_state"])
    random.setstate(replay["python_rng_state"])
    np.random.set_state(replay["numpy_rng_state"])


def save_global_qat_checkpoint(
    root: Path,
    *,
    step: int,
    model_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
    replay_state: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    reserve_bytes: int = 0,
) -> Path:
    if reserve_bytes < 0:
        raise ValueError("checkpoint reserve must be nonnegative")
    masters, retained = split_model_state(model_state)
    identity_sha256 = canonical_json_sha256(run_identity)
    state = {
        "schema": GLOBAL_QAT_STATE_SCHEMA,
        "step": int(step),
        "masters": masters,
        "retained_model_state": retained,
        "optimizer": dict(optimizer_state),
        "replay": dict(replay_state),
        "run_identity_sha256": identity_sha256,
    }
    estimated_bytes = tensor_tree_nbytes(state)
    root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root).free
    required_bytes = int(estimated_bytes * 1.10) + int(reserve_bytes)
    if free_bytes < required_bytes:
        raise OSError(
            f"checkpoint admission failed: free={free_bytes} required={required_bytes} "
            f"estimated_state={estimated_bytes} reserve={reserve_bytes}"
        )
    tensor_metadata = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in masters.items()
    }
    return save_qat_checkpoint(
        root,
        step=step,
        state=state,
        tensor_metadata=tensor_metadata,
        metadata={
            "global_state_schema": GLOBAL_QAT_STATE_SCHEMA,
            "run_identity": dict(run_identity),
            "run_identity_sha256": identity_sha256,
            "estimated_tensor_bytes": estimated_bytes,
            "free_bytes_before_save": free_bytes,
            "required_bytes_before_save": required_bytes,
        },
    )


def load_global_qat_checkpoint(
    path: Path,
    *,
    run_identity: Mapping[str, Any],
    arms: list[str],
    cuda_devices: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state, manifest = load_qat_checkpoint(path, map_location="cpu")
    expected_identity = canonical_json_sha256(run_identity)
    if (
        state.get("schema") != GLOBAL_QAT_STATE_SCHEMA
        or state.get("run_identity_sha256") != expected_identity
        or manifest.get("metadata", {}).get("run_identity_sha256") != expected_identity
    ):
        raise ValueError("global QAT checkpoint run identity mismatch")
    step = state.get("step")
    if not isinstance(step, int) or step <= 0 or step != manifest.get("step"):
        raise ValueError("global QAT checkpoint step mismatch")
    join_model_state(state)
    if not isinstance(state.get("optimizer"), dict):
        raise ValueError("global QAT checkpoint optimizer state is missing")
    validate_replay_state(state.get("replay"), arms=arms, cuda_devices=cuda_devices)
    return state, manifest
