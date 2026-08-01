import unittest

import torch
from transformers import Swin2SRForImageSuperResolution

from clark_air.quant.optimal_ternary import quantize_ternary_mse
from quantizer_controls import quantize_ternary_absmean
from recovery_controls import (
    ScalarDecomposedLowBit,
    add_recovery_parametrizations,
    build_recovery_optimizer,
    localized_feature_loss,
    normalized_feature_loss,
    recovery_parameter_groups,
    recovery_loss_weights,
    scalar_storage_bytes,
)
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


class ScalarDecompositionTest(unittest.TestCase):
    def quantizer(self, weight):
        quantizer = ScalarDecomposedLowBit(
            "ternary",
            "optimal_mse",
            3,
            0.0,
            1.0,
            weight,
        )
        quantizer.set_progress(1.0)
        return quantizer

    def test_zero_initialized_scalars_match_optimal_mse(self):
        weight = torch.tensor([
            [-1.2, -0.4, -0.1],
            [0.2, 0.8, 1.4],
        ])
        quantizer = self.quantizer(weight)
        expected = quantize_ternary_mse(weight, 3).dequantize()

        torch.testing.assert_close(quantizer.project(weight), expected)
        torch.testing.assert_close(quantizer(weight), expected)

    def test_zero_initialized_scalars_match_absmean(self):
        weight = torch.tensor([
            [-1.2, -0.4, -0.1],
            [0.2, 0.8, 1.4],
        ])
        quantizer = ScalarDecomposedLowBit(
            "ternary",
            "absmean",
            3,
            0.0,
            1.0,
            weight,
        )
        quantizer.set_progress(1.0)
        expected = quantize_ternary_absmean(weight, 3).dequantize()

        torch.testing.assert_close(quantizer.project(weight), expected)
        torch.testing.assert_close(quantizer(weight), expected)

    def test_master_and_two_scalars_receive_gradients(self):
        weight = torch.tensor(
            [[-1.2, -0.4, -0.1], [0.2, 0.8, 1.4]],
            requires_grad=True,
        )
        quantizer = self.quantizer(weight)

        quantizer(weight).square().sum().backward()

        self.assertIsNotNone(weight.grad)
        self.assertGreater(weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(quantizer.log_scale_multiplier.grad)
        self.assertGreater(quantizer.log_scale_multiplier.grad.abs().item(), 0)
        self.assertIsNotNone(quantizer.offset_multiplier.grad)
        self.assertGreater(quantizer.offset_multiplier.grad.abs().item(), 0)

    def test_fp16_storage_is_four_bytes_per_matrix(self):
        self.assertEqual(scalar_storage_bytes(144), 576)

    def test_loss_recipes_are_normalized_and_teacher_heavy_is_stricter(self):
        balanced = recovery_loss_weights("balanced")
        teacher_heavy = recovery_loss_weights("teacher_heavy")

        self.assertAlmostEqual(sum(balanced.values()), 1.0)
        self.assertAlmostEqual(sum(teacher_heavy.values()), 1.0)
        self.assertGreater(teacher_heavy["teacher"], balanced["teacher"])


class LocalizedFeatureLossTest(unittest.TestCase):
    def test_selects_one_stage_or_joint_mean(self):
        teacher = (
            torch.zeros(1, 1, 2),
            torch.tensor([[[0.0, 1.0]]]),
            torch.tensor([[[1.0, 0.0]]]),
        )
        student = (
            torch.zeros(1, 1, 2),
            torch.tensor([[[0.2, 0.8]]]),
            torch.tensor([[[0.4, 0.6]]]),
        )
        first = localized_feature_loss(student, teacher, 0)
        second = localized_feature_loss(student, teacher, 1)
        joint = localized_feature_loss(student, teacher, None)

        torch.testing.assert_close(joint, (first + second) / 2)

    def test_rejects_invalid_stage(self):
        states = (torch.zeros(1, 1, 2), torch.ones(1, 1, 2))
        with self.assertRaisesRegex(ValueError, "out of range"):
            localized_feature_loss(states, states, 2)

    def test_l2_feature_calibration_is_finite(self):
        states = (torch.zeros(1, 1, 2), torch.ones(1, 1, 2))
        loss = localized_feature_loss(states, states, 0, "l2")

        self.assertTrue(torch.isfinite(loss))

    def test_l2_feature_loss_is_cosine_distance_not_resolution_diluted(self):
        student = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        teacher = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        small = normalized_feature_loss(student, teacher, "l2")
        tiled = normalized_feature_loss(
            student.repeat(1, 128, 1),
            teacher.repeat(1, 128, 1),
            "l2",
        )

        self.assertAlmostEqual(small.item(), 1.0, delta=1e-6)
        self.assertAlmostEqual(tiled.item(), 1.0, delta=1e-6)


class RecoveryModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        cls.quantizers = add_recovery_parametrizations(
            cls.model,
            group_size=60,
            method="optimal_mse",
            rotation="none",
            rotation_block=4,
            scalar_decomposition=True,
        )

    def test_selects_the_exact_core_and_scalar_budget(self):
        self.assertEqual(len(self.quantizers), 144)
        self.assertEqual(
            sum(module.weight.numel() for _name, _quantizer, module in self.quantizers),
            691_200,
        )
        self.assertEqual(scalar_storage_bytes(len(self.quantizers)), 576)

    def test_absmean_recovery_keeps_the_base_quantizer(self):
        model = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        quantizers = add_recovery_parametrizations(
            model,
            group_size=60,
            method="absmean",
            rotation="none",
            rotation_block=4,
            scalar_decomposition=True,
        )

        self.assertTrue(
            all(quantizer.method == "absmean" for _name, quantizer, _module in quantizers)
        )

    def test_island_groups_have_preregistered_learning_rates(self):
        groups = recovery_parameter_groups(
            self.model,
            self.quantizers,
            "norms",
        )
        optimizer = build_recovery_optimizer(
            "adamw",
            groups,
            learning_rate=2e-5,
            weight_decay=1e-4,
            muon_every=16,
            muon_steps=2,
            scalar_lr_multiplier=100.0,
        )
        auxiliary_rates = [
            group["lr"]
            for group in optimizer.param_groups
            if group["role"] != "matrix_qat8"
        ]

        self.assertIn("masters", groups)
        self.assertEqual(
            sum(parameter.numel() for parameter in groups["normalization"]),
            6_000,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in groups["scalars"]),
            288,
        )
        self.assertTrue(
            any(abs(rate - 2e-3) < 1e-12 for rate in auxiliary_rates)
        )
        self.assertTrue(
            any(abs(rate - 2e-6) < 1e-12 for rate in auxiliary_rates)
        )


if __name__ == "__main__":
    unittest.main()
