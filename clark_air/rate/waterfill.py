"""Water-filling allocator for binary/ternary/FP8 rate candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RateCandidate:
    """One representation choice for one weight group."""

    group: str
    rate: str
    bits: float
    damage: float
    meta: dict | None = None

    def objective(self, bit_price: float) -> float:
        return self.damage + bit_price * self.bits


@dataclass(frozen=True)
class RatePolicy:
    """Chosen rate policy and accounting."""

    choices: tuple[RateCandidate, ...]
    target_bits: float
    total_bits: float
    total_damage: float
    bit_price: float
    feasible: bool

    def by_group(self) -> dict[str, RateCandidate]:
        return {candidate.group: candidate for candidate in self.choices}

    def as_rows(self) -> list[dict]:
        rows = []
        for candidate in self.choices:
            row = {
                "group": candidate.group,
                "rate": candidate.rate,
                "bits": candidate.bits,
                "damage": candidate.damage,
            }
            if candidate.meta:
                row["meta"] = candidate.meta
            rows.append(row)
        return rows


def _group_candidates(candidates: Iterable[RateCandidate]) -> dict[str, list[RateCandidate]]:
    groups: dict[str, list[RateCandidate]] = {}
    for candidate in candidates:
        if candidate.bits < 0:
            raise ValueError(f"negative bits for {candidate.group}:{candidate.rate}")
        if candidate.damage < 0:
            raise ValueError(f"negative damage for {candidate.group}:{candidate.rate}")
        groups.setdefault(candidate.group, []).append(candidate)
    if not groups:
        raise ValueError("at least one candidate is required")
    return groups


def _choose_at_price(groups: dict[str, list[RateCandidate]], bit_price: float) -> tuple[RateCandidate, ...]:
    choices = []
    for group_name in sorted(groups):
        options = groups[group_name]
        choices.append(
            min(
                options,
                key=lambda c: (c.objective(bit_price), c.damage, c.bits, c.rate),
            )
        )
    return tuple(choices)


def _policy(groups: dict[str, list[RateCandidate]], target_bits: float, bit_price: float) -> RatePolicy:
    choices = _choose_at_price(groups, bit_price)
    total_bits = sum(c.bits for c in choices)
    total_damage = sum(c.damage for c in choices)
    return RatePolicy(
        choices=choices,
        target_bits=target_bits,
        total_bits=total_bits,
        total_damage=total_damage,
        bit_price=bit_price,
        feasible=total_bits <= target_bits,
    )


def _all_integer_bits(groups: dict[str, list[RateCandidate]]) -> bool:
    for options in groups.values():
        for candidate in options:
            if abs(candidate.bits - round(candidate.bits)) > 1e-6:
                return False
    return True


def _exact_small_policy(
    groups: dict[str, list[RateCandidate]],
    target_bits: float,
    *,
    max_saving_states: int,
) -> RatePolicy | None:
    """Exact multi-choice knapsack for small integer-bit problems.

    We start from the minimum-damage choice for every group and choose lower-bit
    alternatives until enough bits are saved. The state is capped at the required
    saving, so exact search is practical for unit tests and small policies. Large
    transformer policies use the Lagrangian fallback below.
    """

    if not _all_integer_bits(groups) or abs(target_bits - round(target_bits)) > 1e-6:
        return None

    base_choices = []
    base_bits = 0
    base_damage = 0.0
    for group_name in sorted(groups):
        base = min(groups[group_name], key=lambda c: (c.damage, c.bits, c.rate))
        base_choices.append(base)
        base_bits += int(round(base.bits))
        base_damage += base.damage

    required_saving = base_bits - int(round(target_bits))
    if required_saving <= 0:
        return RatePolicy(
            choices=tuple(base_choices),
            target_bits=target_bits,
            total_bits=float(base_bits),
            total_damage=base_damage,
            bit_price=0.0,
            feasible=True,
        )
    if required_saving > max_saving_states:
        return None

    # saved_bits -> (extra_damage, choices_so_far)
    states: dict[int, tuple[float, tuple[RateCandidate, ...]]] = {0: (0.0, tuple())}
    for base, group_name in zip(base_choices, sorted(groups)):
        next_states: dict[int, tuple[float, tuple[RateCandidate, ...]]] = {}
        for saved, (extra_damage, chosen) in states.items():
            for candidate in groups[group_name]:
                saving = int(round(base.bits - candidate.bits))
                if saving < 0:
                    continue
                new_saved = min(required_saving, saved + saving)
                new_damage = extra_damage + (candidate.damage - base.damage)
                old = next_states.get(new_saved)
                if old is None or new_damage < old[0]:
                    next_states[new_saved] = (new_damage, chosen + (candidate,))
        states = next_states

    best = states.get(required_saving)
    if best is None:
        min_bit_choices = tuple(min(options, key=lambda c: (c.bits, c.damage, c.rate)) for options in groups.values())
        return RatePolicy(
            choices=min_bit_choices,
            target_bits=target_bits,
            total_bits=sum(c.bits for c in min_bit_choices),
            total_damage=sum(c.damage for c in min_bit_choices),
            bit_price=float("inf"),
            feasible=False,
        )

    extra_damage, choices = best
    total_bits = sum(c.bits for c in choices)
    return RatePolicy(
        choices=choices,
        target_bits=target_bits,
        total_bits=total_bits,
        total_damage=base_damage + extra_damage,
        bit_price=0.0,
        feasible=total_bits <= target_bits,
    )


def choose_rates(
    candidates: Iterable[RateCandidate],
    target_bits: float,
    *,
    iterations: int = 80,
    max_exact_saving_states: int = 200_000,
) -> RatePolicy:
    """Choose one candidate per group under a total bit target.

    This solves the Lagrangian relaxation:

        argmin_r damage(group, r) + lambda * bits(group, r)

    and increases lambda until the selected policy fits the requested bit
    budget. It returns the lowest-damage feasible policy found by the search.
    If even the minimum-bit choice does not fit, `feasible` is false.
    """

    if target_bits < 0:
        raise ValueError("target_bits must be non-negative")

    groups = _group_candidates(candidates)

    exact = _exact_small_policy(groups, target_bits, max_saving_states=max_exact_saving_states)
    if exact is not None:
        return exact

    zero_policy = _policy(groups, target_bits, 0.0)
    if zero_policy.feasible:
        return zero_policy

    lo = 0.0
    hi = 1.0
    hi_policy = _policy(groups, target_bits, hi)
    while not hi_policy.feasible and hi < 1e30:
        lo = hi
        hi *= 2.0
        hi_policy = _policy(groups, target_bits, hi)

    if not hi_policy.feasible:
        min_bit_choices = tuple(min(options, key=lambda c: (c.bits, c.damage, c.rate)) for options in groups.values())
        return RatePolicy(
            choices=min_bit_choices,
            target_bits=target_bits,
            total_bits=sum(c.bits for c in min_bit_choices),
            total_damage=sum(c.damage for c in min_bit_choices),
            bit_price=hi,
            feasible=False,
        )

    best = hi_policy
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        policy = _policy(groups, target_bits, mid)
        if policy.feasible:
            hi = mid
            if policy.total_damage < best.total_damage:
                best = policy
        else:
            lo = mid

    return best
