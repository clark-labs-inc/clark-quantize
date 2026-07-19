"""Rate-distortion utilities for Clark Air mixed-rate allocation."""

from .fisher_diag import DiagonalFisher, fisher_quadratic_damage
from .ledger import RateLedger, RateSpec
from .mixed_rate_policy import GroupPolicyInput, candidates_for_group, candidates_for_groups
from .waterfill import RateCandidate, RatePolicy, choose_rates

__all__ = [
    "DiagonalFisher",
    "GroupPolicyInput",
    "RateCandidate",
    "RateLedger",
    "RatePolicy",
    "RateSpec",
    "candidates_for_group",
    "candidates_for_groups",
    "choose_rates",
    "fisher_quadratic_damage",
]
