from __future__ import annotations

import unittest

import torch

from clark_quantize.qwen38_learned_hash_residual import (
    LearnedHashProjection,
    expand_learned_hash,
    fit_fixed_learned_hash_ternary_residual,
    fit_learned_hash_coefficients,
    learn_hash_ternary_residual,
    learn_signed_hash_routing,
    learned_hash_storage_projection,
    learned_sketch_rows,
)


class Qwen38LearnedHashResidualTests(unittest.TestCase):
    def test_implicit_runtime_matches_expanded_matrix(self) -> None:
        projection = LearnedHashProjection(
            input_dim=8,
            sketch_dim=3,
            buckets=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
            signs=torch.tensor([1, -1, 1, -1, 1, -1, 1, 1], dtype=torch.int8),
        )
        generator = torch.Generator().manual_seed(158)
        inputs = torch.randn(7, 8, generator=generator)
        coefficients = torch.randn(5, 3, generator=generator)
        torch.testing.assert_close(
            learned_sketch_rows(inputs, projection) @ coefficients.T,
            inputs @ expand_learned_hash(coefficients, projection).T,
        )

    def test_exact_coefficients_are_clusterwise_signed_means(self) -> None:
        projection = LearnedHashProjection(
            input_dim=4,
            sketch_dim=2,
            buckets=torch.tensor([0, 0, 1, 1]),
            signs=torch.tensor([1, -1, 1, -1], dtype=torch.int8),
        )
        target = torch.tensor([[3.0, -1.0, 8.0, -4.0]])
        coefficients = fit_learned_hash_coefficients(target, projection)
        torch.testing.assert_close(coefficients, torch.tensor([[2.0, 6.0]]))
        candidate = (target - expand_learned_hash(coefficients, projection)).square().sum()
        perturbed = coefficients + torch.tensor([[0.1, -0.2]])
        alternate = (target - expand_learned_hash(perturbed, projection)).square().sum()
        self.assertLess(candidate.item(), alternate.item())

    def test_signed_lloyd_recovers_low_cardinality_columns(self) -> None:
        generator = torch.Generator().manual_seed(27)
        prototypes = torch.randn(6, 3, generator=generator)
        buckets = torch.tensor([0, 1, 2, 0, 2, 1, 0, 1, 2, 2])
        signs = torch.tensor([1, -1, 1, -1, -1, 1, 1, 1, -1, 1])
        target = (prototypes[:, buckets] * signs).contiguous()
        learned = learn_signed_hash_routing(target, sketch_dim=3, iterations=6)
        self.assertLess(learned.objective_history[-1], 1e-10)
        for earlier, later in zip(
            learned.objective_history, learned.objective_history[1:]
        ):
            self.assertLessEqual(later, earlier + 1e-8)
        torch.testing.assert_close(
            expand_learned_hash(learned.coefficients, learned.projection),
            target,
            rtol=0,
            atol=1e-6,
        )

    def test_hash_ternary_coordinate_fits_are_monotone_and_exported(self) -> None:
        generator = torch.Generator().manual_seed(38)
        source = torch.randn(12, 64, generator=generator)
        trained = learn_hash_ternary_residual(
            source,
            sketch_dim=6,
            group=16,
            nonzero_per_group=7,
            coordinate_iterations=3,
            routing_iterations=4,
        )
        for earlier, later in zip(
            trained.fp32_objective_history,
            trained.fp32_objective_history[1:],
        ):
            self.assertLessEqual(later, earlier + max(1e-8, earlier * 2e-5))
        heldout = torch.randn(5, 64, generator=generator)
        fitted = fit_fixed_learned_hash_ternary_residual(
            heldout,
            trained.projection,
            group=16,
            nonzero_per_group=7,
            iterations=3,
        )
        self.assertTrue(torch.isfinite(torch.tensor(fitted.exported_squared_error)))
        self.assertEqual(fitted.ternary_codes.dtype, torch.int8)

    def test_complete_component_projection_counts_learned_routing(self) -> None:
        gate = learned_hash_storage_projection(
            input_dim=5120,
            output_rows=17408,
            sketch_dim=40,
            nonzero_per_group=64,
        )
        down = learned_hash_storage_projection(
            input_dim=17408,
            output_rows=5120,
            sketch_dim=136,
            nonzero_per_group=64,
        )
        self.assertEqual(gate["routing_bucket_bits"], 6)
        self.assertEqual(down["routing_bucket_bits"], 8)
        self.assertGreater(gate["routing_bytes"], 0)
        self.assertLess(gate["bits_per_weight"], 1.56)
        self.assertLess(down["bits_per_weight"], 1.56)


if __name__ == "__main__":
    unittest.main()
