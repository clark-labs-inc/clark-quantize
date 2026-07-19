"""Bit ledgers for binary/ternary/FP8 mixed-rate policies."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class RateSpec:
    """Storage format for one homogeneous parameter region."""

    name: str
    code_bits_per_weight: float
    scale_bits: float = 0.0
    group_size: int = 1
    fraction: float = 1.0

    @property
    def bits_per_weight(self) -> float:
        return self.code_bits_per_weight + self.scale_bits / max(self.group_size, 1)

    @classmethod
    def binary(cls, *, fraction: float = 1.0, scale_bits: float = 8, group_size: int = 112) -> "RateSpec":
        return cls("binary", 1.0, scale_bits, group_size, fraction)

    @classmethod
    def ternary_entropy(cls, *, fraction: float = 1.0, scale_bits: float = 8, group_size: int = 112) -> "RateSpec":
        return cls("ternary_entropy", log2(3), scale_bits, group_size, fraction)

    @classmethod
    def ternary_40trit(cls, *, fraction: float = 1.0, scale_bits: float = 8, group_size: int = 112) -> "RateSpec":
        return cls("ternary_40trit", 64.0 / 40.0, scale_bits, group_size, fraction)

    @classmethod
    def fp8(cls, *, fraction: float = 1.0) -> "RateSpec":
        return cls("fp8", 8.0, 0.0, 1, fraction)

    @classmethod
    def fp16(cls, *, fraction: float = 1.0) -> "RateSpec":
        return cls("fp16", 16.0, 0.0, 1, fraction)


@dataclass(frozen=True)
class RateLedger:
    """Global storage accounting for a mixed-rate model."""

    specs: tuple[RateSpec, ...]
    reference_bits_per_weight: float = 16.0

    @property
    def bits_per_weight(self) -> float:
        return sum(spec.fraction * spec.bits_per_weight for spec in self.specs)

    @property
    def compression(self) -> float:
        bpw = self.bits_per_weight
        return float("inf") if bpw == 0 else self.reference_bits_per_weight / bpw

    def as_dict(self) -> dict[str, float]:
        return {
            "bits_per_weight": self.bits_per_weight,
            "compression": self.compression,
            "reference_bits_per_weight": self.reference_bits_per_weight,
        }


def global_bits_per_weight(specs: list[RateSpec] | tuple[RateSpec, ...]) -> float:
    """Return weighted global bits per parameter."""

    return RateLedger(tuple(specs)).bits_per_weight

