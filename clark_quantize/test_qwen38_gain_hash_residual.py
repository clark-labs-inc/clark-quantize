from __future__ import annotations

import unittest

import torch

from clark_quantize.qwen38_gain_hash_residual import (
    GainHashProjection,
    expand_gain_hash,
    fit_fixed_gain_hash_ternary_residual,
    fit_gain_hash_coefficients,
    gain_hash_sketch_rows,
    gain_hash_storage_projection,
    learn_gain_hash_routing,
)
from clark_quantize.qwen38_learned_hash_residual import (
    expand_learned_hash,
    learn_signed_hash_routing,
)


class Qwen38GainHashResidualTests(unittest.TestCase):
    def make_projection(self) -> GainHashProjection:
        return GainHashProjection(
            input_dim=8,
            sketch_dim=3,
            gain_bits=2,
            buckets=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
            signs=torch.tensor([1, -1, 1, -1, 1, -1, 1, 1], dtype=torch.int8),
            gain_codes=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.uint8),
            gain_codebook=torch.tensor([0.5, 0.75, 1.25, 2.0], dtype=torch.float16),
        )

    def test_implicit_runtime_matches_expanded_matrix(self) -> None:
        projection = self.make_projection()
        generator = torch.Generator().manual_seed(158)
        inputs = torch.randn(7, 8, generator=generator)
        coefficients = torch.randn(5, 3, generator=generator)
        torch.testing.assert_close(
            gain_hash_sketch_rows(inputs, projection) @ coefficients.T,
            inputs @ expand_gain_hash(coefficients, projection).T,
        )

    def test_exact_coefficients_are_weighted_cluster_means(self) -> None:
        projection = GainHashProjection(
            input_dim=4,
            sketch_dim=2,
            gain_bits=1,
            buckets=torch.tensor([0, 0, 1, 1]),
            signs=torch.tensor([1, -1, 1, -1], dtype=torch.int8),
            gain_codes=torch.tensor([0, 1, 0, 1], dtype=torch.uint8),
            gain_codebook=torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
        target = torch.tensor([[3.0, -2.0, 8.0, -8.0]])
        coefficients = fit_gain_hash_coefficients(target, projection)
        torch.testing.assert_close(coefficients, torch.tensor([[1.4, 4.8]]))
        optimum = (target - expand_gain_hash(coefficients, projection)).square().sum()
        perturbed = coefficients + torch.tensor([[0.2, -0.1]])
        alternate = (target - expand_gain_hash(perturbed, projection)).square().sum()
        self.assertLess(optimum.item(), alternate.item())

    def test_quantized_gains_recover_variable_magnitude_directions(self) -> None:
        generator = torch.Generator().manual_seed(27)
        prototypes = torch.randn(6, 3, generator=generator)
        buckets = torch.tensor([0, 1, 2, 0, 2, 1, 0, 1, 2, 2, 0, 1])
        signs = torch.tensor([1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1])
        gains = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0, 0.25] * 2)
        target = (prototypes[:, buckets] * signs * gains).contiguous()
        gain = learn_gain_hash_routing(
            target, sketch_dim=3, gain_bits=3, iterations=7
        )
        sign_only = learn_signed_hash_routing(
            target, sketch_dim=3, iterations=7
        )
        sign_error = float(
            (target - expand_learned_hash(
                sign_only.coefficients, sign_only.projection
            )).square().sum().item()
        )
        self.assertLess(gain.exported_squared_error, sign_error * 0.05)
        for earlier, later in zip(
            gain.objective_history, gain.objective_history[1:]
        ):
            self.assertLessEqual(later, earlier + max(1e-8, earlier * 2e-6))

    def test_fixed_ternary_gain_hash_fit_is_monotone_and_exported(self) -> None:
        generator = torch.Generator().manual_seed(38)
        routing_source = torch.randn(12, 64, generator=generator)
        routing = learn_gain_hash_routing(
            routing_source, sketch_dim=6, gain_bits=3, iterations=5
        )
        heldout = torch.randn(7, 64, generator=generator)
        fit = fit_fixed_gain_hash_ternary_residual(
            heldout,
            routing.projection,
            group=16,
            nonzero_per_group=8,
            iterations=3,
        )
        for earlier, later in zip(
            fit.fp32_objective_history, fit.fp32_objective_history[1:]
        ):
            self.assertLessEqual(later, earlier + max(1e-8, earlier * 2e-5))
        self.assertTrue(torch.isfinite(torch.tensor(fit.exported_squared_error)))
        self.assertEqual(fit.ternary_codes.dtype, torch.int8)

    def test_gain_side_channel_remains_below_component_budget(self) -> None:
        gate = gain_hash_storage_projection(
            input_dim=5120,
            output_rows=17408,
            sketch_dim=40,
            gain_bits=3,
            nonzero_per_group=64,
        )
        down = gain_hash_storage_projection(
            input_dim=17408,
            output_rows=5120,
            sketch_dim=136,
            gain_bits=3,
            nonzero_per_group=64,
        )
        self.assertEqual(gate["gain_codebook_bytes"], 16)
        self.assertGreater(gate["routing_and_gain_index_bytes"], 0)
        self.assertLess(gate["bits_per_weight"], 1.57)
        self.assertLess(down["bits_per_weight"], 1.57)


if __name__ == "__main__":
    unittest.main()
