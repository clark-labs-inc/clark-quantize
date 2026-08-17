from __future__ import annotations

import unittest

import torch

from clark_quantize.qwen38_geometric_transform import (
    block_rht,
    inverse_block_rht,
    normalized_hadamard,
)


class Qwen38GeometricTransformTests(unittest.TestCase):
    def test_hadamard_is_self_inverse_and_norm_preserving(self) -> None:
        generator = torch.Generator().manual_seed(38)
        source = torch.randn(5, 4, 128, generator=generator)
        transformed = normalized_hadamard(source)
        recovered = normalized_hadamard(transformed)
        torch.testing.assert_close(recovered, source, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(
            transformed.square().sum(), source.square().sum(), rtol=2e-6, atol=2e-6
        )

    def test_block_rht_preserves_dense_gemv(self) -> None:
        generator = torch.Generator().manual_seed(158)
        weight = torch.randn(7, 256, generator=generator)
        activation = torch.randn(3, 256, generator=generator)
        rotated_weight, signs = block_rht(weight, group=128, seed=27)
        rotated_activation, observed_signs = block_rht(
            activation, group=128, signs=signs
        )
        torch.testing.assert_close(observed_signs, signs)
        torch.testing.assert_close(
            rotated_activation @ rotated_weight.T,
            activation @ weight.T,
            rtol=3e-5,
            atol=3e-5,
        )

    def test_inverse_recovers_original_coordinates(self) -> None:
        generator = torch.Generator().manual_seed(27)
        source = torch.randn(4, 256, generator=generator)
        transformed, signs = block_rht(source, group=128, seed=158)
        recovered = inverse_block_rht(transformed, signs, group=128)
        torch.testing.assert_close(recovered, source, rtol=2e-6, atol=2e-6)

    def test_seed_is_reproducible_and_changes_geometry(self) -> None:
        source = torch.arange(256, dtype=torch.float32).reshape(1, -1)
        first, first_signs = block_rht(source, seed=1)
        repeated, repeated_signs = block_rht(source, seed=1)
        other, other_signs = block_rht(source, seed=2)
        torch.testing.assert_close(first, repeated)
        torch.testing.assert_close(first_signs, repeated_signs)
        self.assertFalse(torch.equal(first_signs, other_signs))
        self.assertFalse(torch.equal(first, other))

    def test_rejects_non_power_of_two_or_partial_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            block_rht(torch.ones(2, 120), group=120)
        with self.assertRaisesRegex(ValueError, "dividing"):
            block_rht(torch.ones(2, 192), group=128)


if __name__ == "__main__":
    unittest.main()
