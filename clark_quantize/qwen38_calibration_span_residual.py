"""Low-rank calibration-span residual correctors for quantized Qwen blocks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import torch


def normalized_mse(candidate: torch.Tensor, target: torch.Tensor) -> float:
    if candidate.shape != target.shape:
        raise ValueError("candidate and target shapes differ")
    denominator = target.float().square().sum()
    if float(denominator) <= 0.0:
        raise ValueError("target energy is zero")
    return float((candidate.float() - target.float()).square().sum() / denominator)


@dataclass(frozen=True)
class FloatResidualAdapter:
    """A correction ``x @ left @ right + bias`` fitted in activation space."""

    left: torch.Tensor
    right: torch.Tensor
    bias: torch.Tensor
    input_mean: torch.Tensor
    error_mean: torch.Tensor
    ridge_fraction: float

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    def correction(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.float().matmul(self.left).matmul(self.right) + self.bias


@dataclass(frozen=True)
class Int8ResidualAdapter:
    """Per-rank symmetric INT8 storage for a calibration-span corrector."""

    left_codes: torch.Tensor
    left_scales: torch.Tensor
    right_codes: torch.Tensor
    right_scales: torch.Tensor
    bias: torch.Tensor
    ridge_fraction: float

    def __post_init__(self) -> None:
        if (
            self.left_codes.ndim != 2
            or self.right_codes.ndim != 2
            or self.left_codes.dtype != torch.int8
            or self.right_codes.dtype != torch.int8
            or self.left_codes.shape[1] != self.right_codes.shape[0]
            or self.left_scales.shape != (self.left_codes.shape[1],)
            or self.right_scales.shape != (self.right_codes.shape[0],)
            or self.bias.shape != (self.right_codes.shape[1],)
        ):
            raise ValueError("invalid INT8 residual adapter")

    @property
    def rank(self) -> int:
        return int(self.left_codes.shape[1])

    @property
    def persistent_bytes(self) -> int:
        # Production storage uses INT8 codes and FP16 scales/bias.
        return (
            self.left_codes.numel()
            + self.right_codes.numel()
            + 2 * self.left_scales.numel()
            + 2 * self.right_scales.numel()
            + 2 * self.bias.numel()
        )

    def float_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        left = self.left_codes.float() * self.left_scales.float().unsqueeze(0)
        right = self.right_codes.float() * self.right_scales.float().unsqueeze(1)
        return left, right

    def correction(self, inputs: torch.Tensor) -> torch.Tensor:
        left, right = self.float_factors()
        return inputs.float().matmul(left).matmul(right) + self.bias.float()


def expected_int8_adapter_bytes(
    input_features: int, output_features: int, rank: int
) -> int:
    if input_features <= 0 or output_features <= 0 or rank <= 0:
        raise ValueError("adapter dimensions must be positive")
    return (
        input_features * rank
        + rank * output_features
        + 4 * rank
        + 2 * output_features
    )


def int8_adapter_bytes(adapter: Int8ResidualAdapter) -> bytes:
    payload = b"".join(
        tensor.detach().cpu().contiguous().numpy().tobytes()
        for tensor in (
            adapter.left_codes,
            adapter.left_scales,
            adapter.right_codes,
            adapter.right_scales,
            adapter.bias,
        )
    )
    if len(payload) != adapter.persistent_bytes:
        raise AssertionError("INT8 adapter serialization byte count changed")
    return payload


def write_int8_adapter(path: Path, adapter: Int8ResidualAdapter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(int8_adapter_bytes(adapter))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_int8_adapter(
    path: Path,
    *,
    input_features: int,
    output_features: int,
    rank: int,
    ridge_fraction: float,
) -> Int8ResidualAdapter:
    expected = expected_int8_adapter_bytes(input_features, output_features, rank)
    payload = bytearray(path.read_bytes())
    if len(payload) != expected:
        raise ValueError("INT8 adapter payload byte count changed")
    offset = 0

    def take(dtype: torch.dtype, count: int) -> torch.Tensor:
        nonlocal offset
        element_size = torch.empty((), dtype=dtype).element_size()
        stop = offset + count * element_size
        tensor = torch.frombuffer(payload, dtype=dtype, count=count, offset=offset).clone()
        offset = stop
        return tensor

    left_codes = take(torch.int8, input_features * rank).reshape(input_features, rank)
    left_scales = take(torch.float16, rank)
    right_codes = take(torch.int8, rank * output_features).reshape(rank, output_features)
    right_scales = take(torch.float16, rank)
    bias = take(torch.float16, output_features)
    if offset != len(payload):
        raise AssertionError("INT8 adapter parser did not consume the payload")
    return Int8ResidualAdapter(
        left_codes=left_codes,
        left_scales=left_scales,
        right_codes=right_codes,
        right_scales=right_scales,
        bias=bias,
        ridge_fraction=ridge_fraction,
    )


def _validate_fit_tensors(
    inputs: torch.Tensor,
    candidate_outputs: torch.Tensor,
    target_outputs: torch.Tensor,
    rank: int,
    ridge_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        inputs.ndim != 2
        or candidate_outputs.ndim != 2
        or target_outputs.ndim != 2
        or inputs.shape[0] != candidate_outputs.shape[0]
        or candidate_outputs.shape != target_outputs.shape
    ):
        raise ValueError("residual fit expects aligned two-dimensional rows")
    maximum_rank = min(inputs.shape[0] - 1, inputs.shape[1], target_outputs.shape[1])
    if rank <= 0 or rank > maximum_rank:
        raise ValueError(f"rank must be in [1, {maximum_rank}]")
    if ridge_fraction <= 0.0:
        raise ValueError("ridge_fraction must be positive")
    tensors = tuple(value.detach().float().contiguous() for value in (
        inputs,
        candidate_outputs,
        target_outputs,
    ))
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("residual fit contains non-finite values")
    return tensors


def fit_float_residual_adapter(
    inputs: torch.Tensor,
    candidate_outputs: torch.Tensor,
    target_outputs: torch.Tensor,
    *,
    rank: int,
    ridge_fraction: float,
) -> FloatResidualAdapter:
    """Fit a ridge map and compute its exact thin truncated SVD.

    With ``n`` calibration rows, ridge regression has the dual factorization
    ``DeltaW = X.T @ solve(X @ X.T + lambda I, E)``.  QR factorizations of the
    two thin factors reduce the SVD to an at-most ``n x n`` matrix, avoiding a
    dense hidden-size-squared correction matrix.
    """

    inputs, candidate_outputs, target_outputs = _validate_fit_tensors(
        inputs, candidate_outputs, target_outputs, rank, ridge_fraction
    )
    input_mean = inputs.mean(dim=0)
    error = target_outputs - candidate_outputs
    error_mean = error.mean(dim=0)
    centered_inputs = inputs - input_mean
    centered_error = error - error_mean
    gram = centered_inputs.matmul(centered_inputs.T)
    gram_scale = float(torch.trace(gram) / gram.shape[0])
    if not gram_scale > 0.0:
        raise ValueError("calibration inputs have zero centered energy")
    ridge = ridge_fraction * gram_scale
    dual = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype),
        centered_error,
    )

    left_thin = centered_inputs.T
    right_thin = dual
    q_left, r_left = torch.linalg.qr(left_thin, mode="reduced")
    q_right, r_right = torch.linalg.qr(right_thin.T, mode="reduced")
    middle = r_left.matmul(r_right.T)
    u_small, singular_values, vh_small = torch.linalg.svd(
        middle, full_matrices=False
    )
    root = singular_values[:rank].clamp_min(0.0).sqrt()
    left = q_left.matmul(u_small[:, :rank]) * root.unsqueeze(0)
    right = root.unsqueeze(1) * vh_small[:rank].matmul(q_right.T)
    bias = error_mean - input_mean.matmul(left).matmul(right)
    if not all(bool(torch.isfinite(value).all()) for value in (left, right, bias)):
        raise RuntimeError("residual factorization produced non-finite values")
    return FloatResidualAdapter(
        left=left.contiguous(),
        right=right.contiguous(),
        bias=bias.contiguous(),
        input_mean=input_mean.contiguous(),
        error_mean=error_mean.contiguous(),
        ridge_fraction=ridge_fraction,
    )


def _symmetric_int8_per_rank(
    tensor: torch.Tensor, *, rank_dimension: int
) -> tuple[torch.Tensor, torch.Tensor]:
    reduction = 0 if rank_dimension == 1 else 1
    maxima = tensor.abs().amax(dim=reduction)
    scales = (maxima / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
    view = scales.unsqueeze(0 if rank_dimension == 1 else 1)
    codes = torch.round(tensor / view).clamp(-127, 127).to(torch.int8)
    return codes.contiguous(), scales.to(torch.float16).contiguous()


def quantize_residual_adapter(
    adapter: FloatResidualAdapter,
) -> Int8ResidualAdapter:
    left_codes, left_scales = _symmetric_int8_per_rank(
        adapter.left, rank_dimension=1
    )
    right_codes, right_scales = _symmetric_int8_per_rank(
        adapter.right, rank_dimension=0
    )
    left = left_codes.float() * left_scales.float().unsqueeze(0)
    right = right_codes.float() * right_scales.float().unsqueeze(1)
    # Recenter after factor rounding so the INT8 map preserves the fitted mean
    # correction without storing a separate input-centering vector.
    quantized_bias = adapter.error_mean - adapter.input_mean.matmul(left).matmul(right)
    return Int8ResidualAdapter(
        left_codes=left_codes,
        left_scales=left_scales,
        right_codes=right_codes,
        right_scales=right_scales,
        bias=quantized_bias.to(torch.float16).contiguous(),
        ridge_fraction=adapter.ridge_fraction,
    )
