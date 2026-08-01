import unittest
from collections import Counter

import torch
from transformers import Swin2SRForImageSuperResolution

from full_surface_optimizer import build_full_surface_optimizer
from rotation_control import block_hadamard
from surface_artifact_state import (
    continuous_surface_state_dict,
    hard_surface_state_dict,
    runtime_surface_state_dict,
)
from surface_quantization import (
    add_surface_parametrizations,
    apply_surface_group_sizes,
    set_surface_progress,
    surface_group_size,
    surface_parameter_groups,
)
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


class SurfaceQuantizationTests(unittest.TestCase):
    def model(self):
        return Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        ).eval()

    def test_full_surface_selects_exact_97_percent(self):
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
        )

        self.assertEqual(len(surfaces), 204)
        self.assertEqual(
            Counter(surface.category for surface in surfaces),
            {
                "core_linear": 144,
                "position_bias_linear": 48,
                "residual_body_conv": 10,
                "image_io_conv": 2,
            },
        )
        self.assertEqual(
            sum(surface.module.weight.numel() for surface in surfaces),
            977_604,
        )

    def test_full_surface_propagates_absmean_to_every_surface(self):
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
            method="absmean",
        )

        self.assertTrue(
            all(surface.quantizer.method == "absmean" for surface in surfaces)
        )

    def test_convolutions_can_use_a_separate_quantizer_method(self):
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
            method="absmean",
            conv_method="optimal_mse",
        )
        self.assertTrue(all(
            surface.quantizer.method == (
                "optimal_mse" if surface.module.weight.ndim == 4 else "absmean"
            )
            for surface in surfaces
        ))

    def test_convolution_grouping_respects_filter_geometry(self):
        model = self.model()
        flat = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
            conv_grouping="flat",
        )
        self.assertTrue(
            all(surface.quantizer.group_size == 60 for surface in flat)
        )

        model = self.model()
        by_output = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
            conv_grouping="output_channel",
        )
        image_io = {
            surface.name: surface.quantizer.group_size
            for surface in by_output
            if surface.category == "image_io_conv"
        }
        self.assertEqual(
            image_io,
            {
                "swin2sr.first_convolution": 27,
                "upsample.conv": 540,
            },
        )

        model = self.model()
        by_kernel = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
            conv_grouping="kernel",
        )
        self.assertTrue(all(
            surface.quantizer.group_size
            == (9 if surface.module.kernel_size == (3, 3) else 60)
            for surface in by_kernel
            if surface.module.weight.ndim == 4
        ))
        self.assertTrue(all(
            surface.quantizer.group_size == 60
            for surface in by_kernel
            if surface.module.weight.ndim == 2
        ))

    def test_convolution_grouping_rejects_unknown_layout(self):
        convolution = torch.nn.Conv2d(3, 4, 3)
        with self.assertRaisesRegex(ValueError, "unsupported convolution grouping"):
            surface_group_size(
                convolution,
                "image_io_conv",
                60,
                "diagonal",
            )

    def test_per_surface_group_size_override_is_exact_and_fail_closed(self):
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
            conv_grouping="kernel",
        )
        original_group_sizes = {
            surface.name: surface.quantizer.group_size
            for surface in surfaces
        }
        applied = apply_surface_group_sizes(
            surfaces,
            {"swin2sr.first_convolution": 2},
        )
        group_sizes = {
            surface.name: surface.quantizer.group_size
            for surface in surfaces
        }

        self.assertEqual(applied, {"swin2sr.first_convolution": 2})
        self.assertEqual(group_sizes["swin2sr.first_convolution"], 2)
        self.assertEqual(group_sizes["upsample.conv"], 9)
        self.assertTrue(all(
            surface.quantizer.group_size == original_group_sizes[surface.name]
            for surface in surfaces
            if surface.name != "swin2sr.first_convolution"
        ))
        with self.assertRaisesRegex(ValueError, "unknown"):
            apply_surface_group_sizes(surfaces, {"missing.surface": 2})
        with self.assertRaisesRegex(ValueError, "positive"):
            apply_surface_group_sizes(
                surfaces,
                {"swin2sr.first_convolution": 0},
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            apply_surface_group_sizes(
                surfaces,
                {"swin2sr.first_convolution": 2.0},
            )

    def test_stage_starts_with_hard_core_and_continuous_new_surfaces(self):
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
        )

        set_surface_progress(surfaces, 0.0)

        for surface in surfaces:
            expected = 1.0 if surface.category == "core_linear" else 0.0
            self.assertEqual(surface.quantizer.amount.item(), expected)
        set_surface_progress(surfaces, 1.0)
        self.assertTrue(all(surface.quantizer.amount.item() == 1 for surface in surfaces))

    def test_fresh_fp32_lane_progressively_quantizes_the_core_too(self):
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
        )

        set_surface_progress(surfaces, 0.0, hard_core=False)

        self.assertTrue(all(surface.quantizer.amount.item() == 0 for surface in surfaces))
        set_surface_progress(surfaces, 0.1, hard_core=False)
        self.assertTrue(any(surface.quantizer.amount.item() > 0 for surface in surfaces))
        self.assertTrue(any(surface.quantizer.amount.item() == 0 for surface in surfaces))

    def test_dense_state_matches_hard_parametrized_output(self):
        torch.manual_seed(17)
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
        )
        set_surface_progress(surfaces, 1.0)
        values = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            expected = model(values).reconstruction
        dense = self.model()
        dense.load_state_dict(
            hard_surface_state_dict(model, surfaces),
            strict=True,
        )
        with torch.inference_mode():
            actual = dense(values).reconstruction

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_runtime_state_preserves_only_the_rotated_core_basis(self):
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
            core_rotation="hadamard",
        )
        set_surface_progress(surfaces, 1.0)
        runtime = runtime_surface_state_dict(model, surfaces)
        dense = hard_surface_state_dict(model, surfaces)
        core = next(
            surface
            for surface in surfaces
            if surface.category == "core_linear"
        )
        convolution = next(
            surface
            for surface in surfaces
            if surface.category == "image_io_conv"
        )
        core_name = f"{core.name}.weight"
        convolution_name = f"{convolution.name}.weight"

        torch.testing.assert_close(runtime[core_name], core.module.weight)
        torch.testing.assert_close(
            dense[core_name],
            block_hadamard(core.module.weight, 4),
        )
        torch.testing.assert_close(
            runtime[convolution_name],
            dense[convolution_name],
        )

    def test_continuous_state_matches_zero_amount_parametrized_output(self):
        torch.manual_seed(19)
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
            core_rotation="hadamard",
        )
        set_surface_progress(surfaces, 0.0, hard_core=False)
        values = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            expected = model(values).reconstruction
        dense = self.model()
        dense.load_state_dict(
            continuous_surface_state_dict(model, surfaces),
            strict=True,
        )
        with torch.inference_mode():
            actual = dense(values).reconstruction

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_optimizer_separates_convolutions_from_matrix_qat8(self):
        model = self.model()
        surfaces = add_surface_parametrizations(
            model,
            scope="all_matrix_and_conv",
            group_size=60,
        )
        groups = surface_parameter_groups(model, surfaces)

        self.assertEqual(
            sum(parameter.numel() for parameter in groups["matrix_masters"]),
            789_504,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in groups["convolution_masters"]),
            188_100,
        )
        self.assertTrue(all(parameter.ndim == 2 for parameter in groups["matrix_masters"]))
        self.assertTrue(all(parameter.ndim == 4 for parameter in groups["convolution_masters"]))

        optimizer = build_full_surface_optimizer(
            groups,
            learning_rate=1e-6,
        )
        roles = [group["role"] for group in optimizer.param_groups]
        rates = [group["lr"] for group in optimizer.param_groups]
        self.assertEqual(roles, ["matrix_qat8", "adamw", "adamw"])
        self.assertEqual(rates, [1e-6, 2.5e-7, 1e-7])

        optimizer = build_full_surface_optimizer(
            groups,
            learning_rate=1e-6,
            convolution_learning_rate_multiplier=4.0,
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1e-6, 4e-6, 1e-7],
        )


if __name__ == "__main__":
    unittest.main()
