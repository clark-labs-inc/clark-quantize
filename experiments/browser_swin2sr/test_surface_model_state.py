import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import surface_model_state


class SurfaceModelStateTests(unittest.TestCase):
    def arguments(self, rotation: str) -> Namespace:
        return Namespace(
            group_size=60,
            quantizer_method="absmean",
            core_rotation=rotation,
            device="cpu",
            initialization="fp32",
            schedule_steps=3000,
            conv_quantizer_method="optimal_mse",
            conv_grouping="kernel",
            schedule_profile="sensitive_io_v2",
        )

    def test_fresh_fp32_lane_uses_requested_core_rotation(self):
        args = self.arguments("hadamard")
        payload = {
            "quantization": {
                "mode": "ternary",
                "method": "absmean",
                "group_size": 60,
                "rotation": "hadamard",
                "rotation_block": 4,
            },
            "step": 0,
        }
        student = Mock()
        student.train.return_value = "training-student"
        surfaces = [Mock()]
        with (
            patch.object(surface_model_state, "pretrained", return_value=student),
            patch.object(
                surface_model_state,
                "add_surface_parametrizations",
                return_value=surfaces,
            ) as add_surfaces,
            patch.object(surface_model_state, "set_surface_progress"),
        ):
            loaded = surface_model_state.load_student(args, payload)

        add_surfaces.assert_called_once_with(
            student,
            scope="all_matrix_and_conv",
            group_size=60,
            method="absmean",
            core_rotation="hadamard",
            rotation_block=4,
            conv_method="optimal_mse",
            conv_grouping="kernel",
            schedule_profile="sensitive_io_v2",
        )
        self.assertEqual(loaded[0], "training-student")
        self.assertFalse(loaded[-1])

    def test_rejects_rotation_mismatch_before_loading_model(self):
        args = self.arguments("none")
        payload = {
            "quantization": {
                "method": "absmean",
                "group_size": 60,
                "rotation": "hadamard",
            },
        }

        with self.assertRaisesRegex(ValueError, "core rotation"):
            surface_model_state.load_student(args, payload)

    def test_checkpoint_surface_group_size_override_is_applied(self):
        args = self.arguments("hadamard")
        payload = {
            "quantization": {
                "mode": "ternary",
                "method": "absmean",
                "group_size": 60,
                "rotation": "hadamard",
                "rotation_block": 4,
                "surface_group_sizes": {
                    "swin2sr.first_convolution": 2,
                },
            },
            "step": 0,
        }
        student = Mock()
        student.train.return_value = "training-student"
        surfaces = [Mock()]
        with (
            patch.object(surface_model_state, "pretrained", return_value=student),
            patch.object(
                surface_model_state,
                "add_surface_parametrizations",
                return_value=surfaces,
            ),
            patch.object(
                surface_model_state,
                "apply_surface_group_sizes",
            ) as apply_overrides,
            patch.object(surface_model_state, "set_surface_progress"),
        ):
            surface_model_state.load_student(args, payload)

        apply_overrides.assert_called_once_with(
            surfaces,
            {"swin2sr.first_convolution": 2},
        )
        self.assertEqual(
            args.surface_group_sizes,
            {"swin2sr.first_convolution": 2},
        )


if __name__ == "__main__":
    unittest.main()
