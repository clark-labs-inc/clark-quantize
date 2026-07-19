"""Master-weight hard-forward QAT modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary import quantize_binary
from .optimal_ternary import quantize_ternary_mse


def _quantized_from_master(master: torch.Tensor, mode: str, group_size: int) -> torch.Tensor:
    if mode == "ternary":
        return quantize_ternary_mse(master, group_size=group_size).dequantize().to(master.device, master.dtype)
    if mode == "binary":
        return quantize_binary(master, group_size=group_size).dequantize().to(master.device, master.dtype)
    raise ValueError(f"unsupported quantization mode: {mode}")


def ste_quantized_weight(master: torch.Tensor, mode: str, group_size: int = 112) -> torch.Tensor:
    """Hard low-bit forward value with straight-through master-weight gradient."""

    q = _quantized_from_master(master, mode=mode, group_size=group_size)
    return q.detach() + (master - master.detach())


class MasterWeightQuantLinear(nn.Module):
    """Linear layer using hard binary/ternary weights in forward pass.

    This is the universal Clark Air abstraction: codebook support is derived
    from continuous master weights. There are no explicit masks to edit.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        mode: str = "ternary",
        group_size: int = 112,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.group_size = group_size
        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight_master = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        self.reset_parameters()

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        mode: str = "ternary",
        group_size: int = 112,
        train_bias: bool = True,
    ) -> "MasterWeightQuantLinear":
        quant = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            mode=mode,
            group_size=group_size,
            dtype=layer.weight.dtype,
            device=layer.weight.device,
        )
        quant.weight_master.data.copy_(layer.weight.data)
        if layer.bias is not None and quant.bias is not None:
            quant.bias.data.copy_(layer.bias.data)
            quant.bias.requires_grad_(train_bias)
        return quant

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_master, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_master)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def quantized_weight(self) -> torch.Tensor:
        return ste_quantized_weight(self.weight_master, mode=self.mode, group_size=self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantized_weight(), self.bias)


class MasterWeightQuantEmbedding(nn.Module):
    """Embedding layer using hard binary/ternary weights in forward pass."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        mode: str = "ternary",
        group_size: int = 112,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.mode = mode
        self.group_size = group_size
        self.weight_master = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        )
        self.reset_parameters()

    @classmethod
    def from_embedding(
        cls,
        layer: nn.Embedding,
        *,
        mode: str = "ternary",
        group_size: int = 112,
    ) -> "MasterWeightQuantEmbedding":
        quant = cls(
            layer.num_embeddings,
            layer.embedding_dim,
            padding_idx=layer.padding_idx,
            max_norm=layer.max_norm,
            norm_type=layer.norm_type,
            scale_grad_by_freq=layer.scale_grad_by_freq,
            sparse=layer.sparse,
            mode=mode,
            group_size=group_size,
            dtype=layer.weight.dtype,
            device=layer.weight.device,
        )
        quant.weight_master.data.copy_(layer.weight.data)
        return quant

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight_master)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight_master[self.padding_idx].fill_(0)

    def quantized_weight(self) -> torch.Tensor:
        return ste_quantized_weight(self.weight_master, mode=self.mode, group_size=self.group_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input_ids,
            self.quantized_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
