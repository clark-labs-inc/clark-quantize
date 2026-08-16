"""Bounded hard-ternary primitives for the Qwen3.8 one-layer QAT rung."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
import torch.nn.functional as functional

from .ternary import ternary_codes_scales


CHECKPOINT_SCHEMA = "clark_qwen38_qat_checkpoint_v1"
CHECKPOINT_NAME = re.compile(r"checkpoint-(\d+)$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_qat_checkpoint(
    root: Path,
    *,
    step: int,
    state: dict[str, Any],
    tensor_metadata: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Write and atomically promote one complete trainer checkpoint."""
    if step <= 0 or not root.is_absolute():
        raise ValueError("checkpoint step must be positive and root must be absolute")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    incomplete = sorted(path.name for path in root.iterdir() if path.name.startswith(".checkpoint-") and path.is_dir())
    if incomplete:
        raise RuntimeError(f"incomplete checkpoint requires inspection before retry: {incomplete[0]}")
    final = root / f"checkpoint-{step}"
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"checkpoint already exists: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step}.", dir=root))
    try:
        state_path = temporary / "state.pt"
        with state_path.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(state_path, 0o640)
        state_sha256 = _file_sha256(state_path)
        manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "complete": True,
            "step": step,
            "state_file": state_path.name,
            "state_sha256": state_sha256,
            "tensor_metadata": tensor_metadata,
            "metadata": metadata,
        }
        manifest_path = temporary / "checkpoint.json"
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(manifest_path, 0o640)
        _fsync_directory(temporary)
        os.replace(temporary, final)
        _fsync_directory(root)
        return final
    except BaseException:
        # Keep the dot-prefixed incomplete directory for forensic diagnosis.
        # The retention tool intentionally ignores it.
        raise


def load_qat_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> tuple[dict, dict]:
    """Hash, schema, name, and tensor-metadata validate a complete checkpoint."""
    if path.is_symlink():
        raise ValueError(f"checkpoint directory must not be a symlink: {path}")
    path = path.resolve()
    match = CHECKPOINT_NAME.fullmatch(path.name)
    if match is None or not path.is_dir():
        raise ValueError(f"invalid checkpoint directory: {path}")
    manifest = json.loads((path / "checkpoint.json").read_text())
    step = int(match.group(1))
    if (
        manifest.get("schema") != CHECKPOINT_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("step") != step
        or manifest.get("state_file") != "state.pt"
    ):
        raise ValueError("checkpoint manifest identity mismatch")
    state_path = path / "state.pt"
    if not state_path.is_file() or _file_sha256(state_path) != manifest.get("state_sha256"):
        raise ValueError("checkpoint state SHA-256 mismatch")
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    masters = state.get("masters")
    if not isinstance(masters, dict):
        raise ValueError("checkpoint state has no master tensor map")
    expected = manifest.get("tensor_metadata")
    actual = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in masters.items()
        if torch.is_tensor(tensor)
    }
    if actual != expected or len(actual) != len(masters):
        raise ValueError("checkpoint master tensor metadata mismatch")
    return state, manifest


def state_tree_equal(left: Any, right: Any) -> bool:
    """Compare nested checkpoint/optimizer state exactly across device moves."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(state_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(state_tree_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def checkpoint_rng_state_valid(state: dict[str, Any], *, cuda_devices: int) -> bool:
    cpu = state.get("cpu_rng_state")
    cuda = state.get("cuda_rng_state")
    return (
        torch.is_tensor(cpu)
        and cpu.dtype == torch.uint8
        and cpu.ndim == 1
        and cpu.numel() > 0
        and isinstance(cuda, list)
        and len(cuda) == cuda_devices
        and all(
            torch.is_tensor(item)
            and item.dtype == torch.uint8
            and item.ndim == 1
            and item.numel() > 0
            for item in cuda
        )
    )


@torch.no_grad()
def hard_ternary_weight(
    master: torch.Tensor,
    *,
    group_size: int = 128,
    row_batch: int = 128,
) -> torch.Tensor:
    """Materialize artifact-equivalent ternary weights with bounded workspace."""
    if master.ndim != 2 or group_size <= 0 or row_batch <= 0:
        raise ValueError("hard ternary weight requires a matrix and positive group/row sizes")
    rows, columns = master.shape
    output = torch.empty_like(master, dtype=torch.float32)
    for start in range(0, rows, row_batch):
        stop = min(start + row_batch, rows)
        codes, scales = ternary_codes_scales(
            master[start:stop].float(), group=group_size, method="optimal"
        )
        padding = (-columns) % group_size
        if padding:
            codes = functional.pad(codes, (0, padding))
        # Match the artifact reader: code and scale multiplication occurs in
        # FP16 before the runtime parameter receives BF16 values.
        reconstructed = (
            codes.reshape(stop - start, -1, group_size).to(torch.float16)
            * scales.to(torch.float16).unsqueeze(-1)
        ).reshape(stop - start, -1)[:, :columns]
        output[start:stop].copy_(reconstructed.float())
    return output


class _HardTernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, master: torch.Tensor, group_size: int, row_batch: int) -> torch.Tensor:
        del ctx
        return hard_ternary_weight(master, group_size=int(group_size), row_batch=int(row_batch))

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        del ctx
        return gradient, None, None


def hard_ternary_ste(master: torch.Tensor, *, group_size: int = 128, row_batch: int = 128) -> torch.Tensor:
    return _HardTernarySTE.apply(master, group_size, row_batch)


class HardTernaryLinear(torch.nn.Module):
    """Linear layer with an FP32 master and artifact-equivalent hard forward."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        group_size: int = 128,
        row_batch: int = 128,
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("HardTernaryLinear weight must be two-dimensional")
        self.master = torch.nn.Parameter(weight.detach().float().clone())
        self.group_size = int(group_size)
        self.row_batch = int(row_batch)
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = hard_ternary_ste(
            self.master,
            group_size=self.group_size,
            row_batch=self.row_batch,
        ).to(dtype=inputs.dtype)
        bias = self.bias.to(dtype=inputs.dtype) if self.bias is not None else None
        return functional.linear(inputs, weight, bias)

    @torch.no_grad()
    def hard_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return hard_ternary_weight(
            self.master,
            group_size=self.group_size,
            row_batch=self.row_batch,
        ).to(dtype=dtype)


def topk_tail_teacher_kl(
    student_logits: torch.Tensor,
    teacher_top_values: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    teacher_logsumexp: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Forward KL over cached teacher top-k classes plus one exact tail bucket."""
    if student_logits.ndim != 3 or student_logits.shape[0] != 1:
        raise ValueError("student logits must have shape [1, tokens, vocab]")
    values = teacher_top_values.to(device=student_logits.device, dtype=torch.float32)
    indices = teacher_top_indices.to(device=student_logits.device, dtype=torch.long)
    teacher_lse = teacher_logsumexp.to(device=student_logits.device, dtype=torch.float32)
    tokens = student_logits.shape[1]
    if values.ndim != 2 or values.shape[0] != tokens or indices.shape != values.shape:
        raise ValueError("cached teacher top-k shapes do not match student tokens")
    if teacher_lse.shape != (tokens,):
        raise ValueError("cached teacher logsumexp shape does not match student tokens")

    student_logp = torch.log_softmax(student_logits[:, :-1].float(), dim=-1).squeeze(0)
    teacher_logp = values[:-1] - teacher_lse[:-1, None]
    teacher_prob = teacher_logp.exp()
    teacher_top_mass = teacher_prob.sum(dim=-1).clamp(max=1.0 - 1e-8)
    teacher_tail = (1.0 - teacher_top_mass).clamp_min(1e-8)
    student_top_logp = student_logp.gather(dim=-1, index=indices[:-1])
    student_top_mass = student_top_logp.exp().sum(dim=-1).clamp(max=1.0 - 1e-8)
    student_tail = (1.0 - student_top_mass).clamp_min(1e-8)
    top_term = (teacher_prob * (teacher_logp - student_top_logp)).sum(dim=-1)
    tail_term = teacher_tail * (teacher_tail.log() - student_tail.log())
    per_token = top_term + tail_term
    loss = per_token.mean()
    stats = {
        "mean_teacher_top_mass": float(teacher_top_mass.mean().detach().cpu()),
        "mean_student_top_mass": float(student_top_mass.mean().detach().cpu()),
        "mean_teacher_kl": float(loss.detach().cpu()),
        "positions": int(per_token.numel()),
    }
    return loss, stats
