"""Finite-block rate bounds for nonuniform Qwen3.8 ternary codes."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


ALPHABET = (-1, 0, 1)
DEFAULT_NUMERICAL_GUARD_BITS = 64


@dataclass(frozen=True)
class TernaryTypeClassLedger:
    counts: tuple[int, int, int]
    trits: int
    entropy_bits_per_trit: float
    conservative_payload_bits: int
    payload_bytes: int
    header_bytes: int

    @property
    def section_bytes_before_tensor_alignment(self) -> int:
        return self.header_bytes + self.payload_bytes

    @property
    def effective_bits_per_trit(self) -> float:
        return self.section_bytes_before_tensor_alignment * 8 / self.trits


def empirical_entropy_bits_per_trit(counts: Iterable[int]) -> float:
    values = tuple(int(value) for value in counts)
    if len(values) != 3 or any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("ternary counts must contain three nonnegative nonempty counts")
    total = sum(values)
    return -sum(
        value / total * math.log2(value / total) for value in values if value
    )


def conservative_type_class_payload_bits(
    counts: Iterable[int],
    *,
    numerical_guard_bits: int = DEFAULT_NUMERICAL_GUARD_BITS,
) -> int:
    """Bound an enumerative ternary payload without constructing huge integers.

    A sequence with composition ``counts`` is one of
    ``n! / product(count!)`` orderings.  The type-class inequality bounds that
    cardinality by ``2 ** (n * H(counts/n))``.  We round upward and add a
    deliberately generous numerical guard; this remains materially tighter
    than uniform base-3 for the observed Qwen distribution while avoiding a
    floating-point undercount claim.
    """
    values = tuple(int(value) for value in counts)
    if numerical_guard_bits < 0:
        raise ValueError("numerical guard must be nonnegative")
    entropy = empirical_entropy_bits_per_trit(values)
    return math.ceil(sum(values) * entropy) + numerical_guard_bits


def ternary_type_class_ledger(
    counts: Iterable[int],
    *,
    header_bytes: int = 16,
    numerical_guard_bits: int = DEFAULT_NUMERICAL_GUARD_BITS,
) -> TernaryTypeClassLedger:
    values = tuple(int(value) for value in counts)
    if header_bytes < 12:
        raise ValueError("ternary type-class header must be at least 12 bytes")
    entropy = empirical_entropy_bits_per_trit(values)
    payload_bits = conservative_type_class_payload_bits(
        values, numerical_guard_bits=numerical_guard_bits
    )
    return TernaryTypeClassLedger(
        counts=values,
        trits=sum(values),
        entropy_bits_per_trit=entropy,
        conservative_payload_bits=payload_bits,
        payload_bytes=(payload_bits + 7) // 8,
        header_bytes=header_bytes,
    )


def exact_type_class_payload_bits_small(counts: Iterable[int]) -> int:
    """Return the exact fixed-length enumerative rate for tests/small blocks."""
    values = tuple(int(value) for value in counts)
    empirical_entropy_bits_per_trit(values)
    remaining = sum(values)
    cardinality = 1
    for value in values[:-1]:
        cardinality *= math.comb(remaining, value)
        remaining -= value
    return max(cardinality - 1, 0).bit_length()
