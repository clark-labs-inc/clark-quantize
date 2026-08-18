from __future__ import annotations

import unittest

import torch

from clark_quantize.qwen38_activation_binary_mlp import (
    complete_mlp_storage_projection,
    error_metrics,
    exported_int4_left,
    exported_row_normalized_ternary,
    fit_activation_left_int4,
)


class Qwen38ActivationBinaryMLPTests(unittest.TestCase):
    def test_activation_fit_recovers_exportable_low_rank_outputs(self) -> None:
        generator = torch.Generator().manual_seed(20260818)
        features = torch.randn(96, 8, generator=generator)
        codes = torch.randint(-7, 8, (48, 8), generator=generator, dtype=torch.int8)
        scales = torch.linspace(0.01, 0.03, 48).to(torch.float16)
        target = features @ exported_int4_left(codes, scales).T
        fit = fit_activation_left_int4(features, target, relative_ridge=1e-6)
        baseline = torch.zeros_like(target)
        recovered = features @ fit.coefficients.T
        before = error_metrics(target, baseline)["squared_error"]
        after = error_metrics(target, recovered)["squared_error"]
        self.assertLess(after, before * 0.03)
        torch.testing.assert_close(
            fit.coefficients, exported_int4_left(fit.codes, fit.scales)
        )

    def test_row_normalized_ternary_uses_frozen_scale_storage(self) -> None:
        generator = torch.Generator().manual_seed(20260819)
        source = torch.randn(7, 256, generator=generator)
        exported = exported_row_normalized_ternary(source)
        self.assertEqual(exported.weights.shape, source.shape)
        self.assertEqual(exported.scale_record["groups"], 14)
        self.assertEqual(exported.scale_record["row_anchor_bytes"], 14)
        self.assertEqual(exported.scale_record["ratio_codebook_bytes"], 32)
        self.assertEqual(exported.scale_record["scale_index_bytes"], 7)
        self.assertTrue(bool(torch.isfinite(exported.weights).all()))

    def test_complete_mlp_projection_is_honestly_below_gate(self) -> None:
        ledger = complete_mlp_storage_projection(
            hidden=5120, intermediate=17408, rank=40
        )
        self.assertEqual(ledger["parameters"], 267_386_880)
        self.assertLessEqual(ledger["bits_per_weight"], 1.55)
        self.assertGreater(ledger["bits_per_weight"], 1.5)

    def test_invalid_exported_left_rejects_zero_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            exported_int4_left(
                torch.ones(2, 3, dtype=torch.int8),
                torch.tensor([0.0, 1.0], dtype=torch.float16),
            )


if __name__ == "__main__":
    unittest.main()
