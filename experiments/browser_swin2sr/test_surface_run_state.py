import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from surface_run_state import checkpoint_payload, validate_resume_config


def arguments(**changes):
    values = {
        "group_size": 60,
        "schedule_steps": 3000,
        "loss_recipe": "balanced",
        "pixel_objective": "charbonnier",
        "ground_truth_mse_floor": 3.2e-4,
        "teacher_mse_floor": 2.0e-5,
        "seed": 20260729,
        "batch_size": 8,
        "crop_size": 128,
        "learning_rate": 1e-6,
        "optimizer": "bonsai_qat8_muon",
        "conv_quantizer_method": "optimal_mse",
        "conv_grouping": "kernel",
        "schedule_profile": "sensitive_io_v2",
        "correction_learning_rate_multiplier": 0.10,
        "hard_shadow_weight_start": 0.05,
        "hard_shadow_weight_end": 0.25,
        "hard_shadow_every": 2,
    }
    values.update(changes)
    return Namespace(**values)


def payload():
    return {
        "surface_training": {
            "group_size": 60,
            "schedule_steps": 3000,
            "loss_recipe": "balanced",
            "pixel_objective": "charbonnier",
            "ground_truth_mse_floor": 3.2e-4,
            "teacher_mse_floor": 2.0e-5,
            "seed": 20260729,
            "batch_size": 8,
            "crop_size": 128,
            "conv_quantizer_method": "optimal_mse",
            "conv_grouping": "kernel",
            "schedule_profile": "sensitive_io_v2",
            "hard_shadow_weight_start": 0.05,
            "hard_shadow_weight_end": 0.25,
            "hard_shadow_every": 2,
            "optimizer": {
                "name": "bonsai_qat8_muon",
                "learning_rate": 1e-6,
            },
        },
    }


class SurfaceRunStateTests(unittest.TestCase):
    def test_matching_resume_configuration_is_accepted(self):
        validate_resume_config(arguments(), payload(), 60)

    def test_legacy_checkpoint_uses_original_charbonnier_defaults(self):
        previous = payload()
        del previous["surface_training"]["pixel_objective"]
        del previous["surface_training"]["ground_truth_mse_floor"]
        del previous["surface_training"]["teacher_mse_floor"]

        validate_resume_config(arguments(), previous, 60)

    def test_changed_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed schedule_steps"):
            validate_resume_config(arguments(schedule_steps=3001), payload(), 60)

    def test_changed_learning_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed learning_rate"):
            validate_resume_config(arguments(learning_rate=2e-6), payload(), 60)

    def test_changed_surface_group_size_override_is_rejected(self):
        previous = payload()
        previous["surface_training"]["surface_group_sizes"] = {
            "swin2sr.first_convolution": 2,
        }
        requested = arguments()
        requested.surface_group_sizes = {}
        with self.assertRaisesRegex(ValueError, "surface_group_sizes"):
            validate_resume_config(requested, previous, 60)

    def test_changed_convolution_learning_rate_is_rejected(self):
        previous = payload()
        previous["surface_training"][
            "convolution_learning_rate_multiplier"
        ] = 1.0
        with self.assertRaisesRegex(
            ValueError,
            "convolution_learning_rate_multiplier",
        ):
            validate_resume_config(arguments(), previous, 60)

    def test_changed_correction_learning_rate_is_rejected(self):
        previous = payload()
        previous["surface_training"][
            "correction_learning_rate_multiplier"
        ] = 0.25
        with self.assertRaisesRegex(
            ValueError,
            "correction_learning_rate_multiplier",
        ):
            validate_resume_config(arguments(), previous, 60)

    def test_changed_optimizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed optimizer"):
            validate_resume_config(
                arguments(optimizer="adamw"),
                payload(),
                60,
            )

    def test_checkpoint_separates_matrix_and_convolution_quantizers(self):
        matrix = SimpleNamespace(
            name="matrix",
            module=SimpleNamespace(weight=torch.zeros(2, 2)),
            quantizer=SimpleNamespace(
                method="absmean",
                group_size=60,
                rotation_block=4,
            ),
        )
        convolution = SimpleNamespace(
            name="convolution",
            module=SimpleNamespace(weight=torch.zeros(2, 2, 3, 3)),
            quantizer=SimpleNamespace(
                method="optimal_mse",
                group_size=9,
                rotation_block=4,
            ),
        )
        student = Mock()
        student.state_dict.return_value = {}
        optimizer = Mock()
        optimizer.state_dict.return_value = {}
        config = {
            "optimizer": {"learning_rate": 1e-6},
            "conv_grouping": "kernel",
            "schedule_profile": "sensitive_io_v2",
            "core_rotation": "hadamard",
        }

        result = checkpoint_payload(
            student,
            optimizer,
            [convolution, matrix],
            surface_step=2,
            base_global_step=0,
            metrics={},
            run_state={},
            config=config,
        )

        self.assertEqual(result["quantization"]["method"], "absmean")
        self.assertEqual(result["quantization"]["conv_method"], "optimal_mse")
        self.assertEqual(result["quantization"]["surface_group_sizes"], {})



if __name__ == "__main__":
    unittest.main()
