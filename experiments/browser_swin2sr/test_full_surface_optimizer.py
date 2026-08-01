import copy
import unittest

import torch

from full_surface_cli import (
    normalize_surface_group_sizes,
    surface_group_size,
)
from full_surface_optimizer import (
    build_full_surface_optimizer,
    optimizer_config,
)


def parameter_groups():
    return {
        "matrix_masters": [torch.nn.Parameter(torch.tensor([1.0, -2.0]))],
        "convolution_masters": [
            torch.nn.Parameter(torch.tensor([[[[0.5]]]]))
        ],
        "corrections": [torch.nn.Parameter(torch.tensor([0.25]))],
    }


class FullSurfaceOptimizerTests(unittest.TestCase):
    def test_adamw_preserves_sensitive_learning_rate_groups(self):
        groups = parameter_groups()
        optimizer = build_full_surface_optimizer(
            groups,
            optimizer_name="adamw",
            learning_rate=2e-4,
            convolution_learning_rate_multiplier=0.25,
            correction_learning_rate_multiplier=0.10,
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(
            [group["role"] for group in optimizer.param_groups],
            ["matrix_adamw", "convolution_adamw", "correction_adamw"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [2e-4, 5e-5, 2e-5],
        )

    def test_qat8_muon_remains_the_legacy_default(self):
        optimizer = build_full_surface_optimizer(
            parameter_groups(),
            learning_rate=1e-6,
        )

        self.assertEqual(
            [group["role"] for group in optimizer.param_groups],
            ["matrix_qat8", "adamw", "adamw"],
        )
        self.assertTrue(optimizer.param_groups[0]["use_muon"])

    def test_adamw_resume_has_exact_next_step(self):
        groups = parameter_groups()
        optimizer = build_full_surface_optimizer(
            groups,
            optimizer_name="adamw",
            learning_rate=1e-3,
        )
        for parameters in groups.values():
            for parameter in parameters:
                parameter.grad = torch.full_like(parameter, 0.125)
        optimizer.step()

        resumed_groups = {
            name: [
                torch.nn.Parameter(parameter.detach().clone())
                for parameter in parameters
            ]
            for name, parameters in groups.items()
        }
        resumed = build_full_surface_optimizer(
            resumed_groups,
            optimizer_name="adamw",
            learning_rate=1e-3,
        )
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))

        for parameters in groups.values():
            for parameter in parameters:
                parameter.grad = torch.full_like(parameter, -0.25)
        for parameters in resumed_groups.values():
            for parameter in parameters:
                parameter.grad = torch.full_like(parameter, -0.25)
        optimizer.step()
        resumed.step()

        for name in groups:
            for expected, actual in zip(groups[name], resumed_groups[name]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_overlapping_groups_fail_closed(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        groups = {
            "matrix_masters": [parameter],
            "convolution_masters": [parameter],
            "corrections": [],
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_full_surface_optimizer(
                groups,
                optimizer_name="adamw",
                learning_rate=1e-3,
            )

    def test_optimizer_config_records_recipe_identity(self):
        config = optimizer_config(
            optimizer_name="adamw",
            learning_rate=8e-5,
            convolution_learning_rate_multiplier=0.25,
            correction_learning_rate_multiplier=0.10,
            reset_from_base=True,
        )
        self.assertEqual(config["name"], "adamw")
        self.assertEqual(config["betas"], [0.9, 0.999])
        self.assertEqual(
            config["convolution_learning_rate_multiplier"],
            0.25,
        )
        self.assertEqual(config["correction_learning_rate_multiplier"], 0.10)

    def test_surface_override_parser_rejects_duplicates(self):
        self.assertEqual(
            surface_group_size("swin2sr.first_convolution=2"),
            ("swin2sr.first_convolution", 2),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_surface_group_sizes([
                ("swin2sr.first_convolution", 2),
                ("swin2sr.first_convolution", 3),
            ])


if __name__ == "__main__":
    unittest.main()
