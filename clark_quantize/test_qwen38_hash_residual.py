from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import torch

from clark_quantize.qwen38_hash_residual import (
    expand_sketch_rows,
    fit_hash_ternary_residual,
    fixed_support_ternary,
    least_squares_hash_coefficients,
    projected_hash_residual_bits_per_weight,
    projection_gram,
    sketch_rows,
    sparse_jl_projection,
)


class Qwen38HashResidualTests(unittest.TestCase):
    def test_implicit_sparse_projection_matches_dense_algebra(self) -> None:
        projection = sparse_jl_projection(32, 9, hashes_per_input=4, seed=12345)
        identity = torch.eye(projection.sketch_dim)
        dense = expand_sketch_rows(identity, projection)
        generator = torch.Generator().manual_seed(38)
        values = torch.randn(5, 32, generator=generator)
        torch.testing.assert_close(sketch_rows(values, projection), values @ dense.T)
        coefficients = torch.randn(5, 9, generator=generator)
        torch.testing.assert_close(
            expand_sketch_rows(coefficients, projection), coefficients @ dense
        )
        torch.testing.assert_close(projection_gram(projection), dense @ dense.T)

    def test_hash_least_squares_beats_unscaled_clark_estimator(self) -> None:
        generator = torch.Generator().manual_seed(158)
        target = torch.randn(7, 64, generator=generator)
        projection = sparse_jl_projection(64, 16, seed=27)
        clark = sketch_rows(target, projection)
        least_squares = least_squares_hash_coefficients(target, projection)
        clark_error = (target - expand_sketch_rows(clark, projection)).square().sum()
        fitted_error = (target - expand_sketch_rows(least_squares, projection)).square().sum()
        self.assertLess(fitted_error.item(), clark_error.item())

    def test_fixed_support_quantizer_has_exact_nonzero_budget(self) -> None:
        source = torch.arange(-16, 16, dtype=torch.float32).reshape(2, 16)
        codes, scales, reconstructed = fixed_support_ternary(
            source, group=8, nonzero_per_group=3
        )
        self.assertEqual(int((codes != 0).sum()), 2 * 2 * 3)
        self.assertEqual(scales.shape, (2, 2))
        self.assertEqual(reconstructed.shape, source.shape)

    def test_alternating_hash_residual_objective_is_monotone(self) -> None:
        generator = torch.Generator().manual_seed(27)
        source = torch.randn(6, 64, generator=generator)
        fit = fit_hash_ternary_residual(
            source,
            sketch_dim=8,
            hashes_per_input=2,
            group=16,
            nonzero_per_group=7,
            iterations=4,
        )
        for earlier, later in zip(
            fit.fp32_objective_history, fit.fp32_objective_history[1:]
        ):
            self.assertLessEqual(later, earlier + max(1e-8, earlier * 2e-5))
        self.assertTrue(torch.isfinite(torch.tensor(fit.exported_squared_error)))

    def test_rate_projection_can_target_sub_158(self) -> None:
        rate = projected_hash_residual_bits_per_weight(
            input_dim=5120,
            sketch_dim=80,
            group=128,
            nonzero_per_group=61,
        )
        self.assertLess(rate, 1.58)
        self.assertGreater(rate, 1.5)

    def test_hash_screen_preregistration_freezes_qwen38_and_clark_hash(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        path = repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_hash_residual_screen_preregistration_v1.json"
        )
        preregistration = json.loads(path.read_text())
        self.assertEqual(
            preregistration["scope"]["active_model_only"], "Qwen/Qwen3.8-27B"
        )
        sampled = sum(
            (tensor["row_span"][1] - tensor["row_span"][0]) * tensor["shape"][1]
            for tensor in preregistration["scope"]["tensors"]
        )
        self.assertEqual(sampled, preregistration["scope"]["sampled_parameters"])
        self.assertEqual(
            preregistration["clark_hash_source"]["paper_blob_sha"],
            "4a7c7ffa06ac9c650c6b822cd0bc111d501c3818",
        )
        for descriptor in preregistration["frozen_inputs"].values():
            source = repo / descriptor["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), descriptor["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
