from __future__ import annotations

import unittest
import hashlib
import json
from pathlib import Path

import torch

from clark_quantize.qwen38_affine_ternary import (
    affine_ternary_codes_scales,
    reconstruct_affine_ternary,
)
from clark_quantize.ternary import ternary_codes_scales


class Qwen38AffineTernaryTests(unittest.TestCase):
    def test_affine_fit_recovers_shifted_ternary_rows(self) -> None:
        generator = torch.Generator().manual_seed(38)
        codes = torch.randint(-1, 2, (3, 16), generator=generator).to(torch.float32)
        centers = torch.tensor([0.75, -0.40, 1.25])[:, None]
        scales = torch.tensor(
            [[0.5, 1.2], [0.8, 0.3], [1.1, 0.6]], dtype=torch.float32
        ).repeat_interleave(8, dim=1)
        source = centers + scales * codes
        fit = affine_ternary_codes_scales(source, group=8, max_iterations=20)
        reconstructed = reconstruct_affine_ternary(fit, group=8)
        self.assertLess(torch.linalg.vector_norm(reconstructed - source).item(), 1e-5)
        self.assertTrue(set(fit.codes.unique().tolist()) <= {-1, 0, 1})

    def test_objective_is_monotone_and_no_worse_than_symmetric_start(self) -> None:
        generator = torch.Generator().manual_seed(158)
        source = torch.randn(7, 29, generator=generator) * 0.4 + torch.linspace(
            -0.25, 0.35, 7
        )[:, None]
        diagonal = torch.linspace(0.1, 2.0, 29)
        fit = affine_ternary_codes_scales(
            source,
            group=8,
            activation_weights=diagonal,
            max_iterations=20,
        )
        for earlier, later in zip(fit.objective_history, fit.objective_history[1:]):
            self.assertLessEqual(later, earlier + max(1e-9, abs(earlier) * 5e-7))
        self.assertLessEqual(fit.objective_history[-1], fit.objective_history[0])

    def test_weighting_prefers_high_variance_columns(self) -> None:
        source = torch.tensor([[0.0, 0.0, 1.0, 2.0]], dtype=torch.float32)
        uniform = affine_ternary_codes_scales(source, group=4)
        weighted = affine_ternary_codes_scales(
            source,
            group=4,
            activation_weights=torch.tensor([1.0, 1.0, 100.0, 100.0]),
        )
        uniform_error = reconstruct_affine_ternary(uniform, group=4) - source
        weighted_error = reconstruct_affine_ternary(weighted, group=4) - source
        importance = torch.tensor([1.0, 1.0, 100.0, 100.0])
        self.assertLessEqual(
            (weighted_error.square() * importance).sum().item(),
            (uniform_error.square() * importance).sum().item(),
        )

    def test_padding_is_excluded_and_zero_weights_are_valid(self) -> None:
        source = torch.zeros(2, 11)
        fit = affine_ternary_codes_scales(source, group=8)
        self.assertEqual(fit.codes.shape, source.shape)
        self.assertEqual(fit.scales.shape, (2, 2))
        torch.testing.assert_close(reconstruct_affine_ternary(fit, group=8), source)

    def test_symmetric_codes_are_the_frozen_initial_objective(self) -> None:
        generator = torch.Generator().manual_seed(27)
        source = torch.randn(4, 16, generator=generator)
        codes, scales = ternary_codes_scales(source, group=8, method="optimal")
        symmetric = scales.float().repeat_interleave(8, dim=1) * codes.float()
        expected = (source - symmetric).square().sum().item()
        fit = affine_ternary_codes_scales(source, group=8)
        self.assertAlmostEqual(fit.objective_history[0], expected, places=4)

    def test_rejects_invalid_activation_weights(self) -> None:
        source = torch.ones(2, 4)
        with self.assertRaisesRegex(ValueError, "one value"):
            affine_ternary_codes_scales(source, activation_weights=torch.ones(3))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            affine_ternary_codes_scales(
                source, activation_weights=torch.tensor([1.0, -1.0, 1.0, 1.0])
            )

    def test_geometric_preregistration_freezes_qwen38_sources_and_denominator(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        path = repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_geometric_ternary_ablation_preregistration_v1.json"
        )
        preregistration = json.loads(path.read_text())
        self.assertEqual(
            preregistration["scope"]["active_model_only"], "Qwen/Qwen3.8-27B"
        )
        sampled = sum(
            (stop - start) * tensor["shape"][1]
            for tensor in preregistration["scope"]["tensors"]
            for start, stop in tensor["row_spans"]
        )
        self.assertEqual(sampled, preregistration["scope"]["sampled_parameters"])
        self.assertEqual(
            sum(len(tensor["row_spans"]) for tensor in preregistration["scope"]["tensors"]),
            preregistration["gate"]["required_spans"],
        )
        for descriptor in preregistration["frozen_inputs"].values():
            source = repo / descriptor["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), descriptor["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
