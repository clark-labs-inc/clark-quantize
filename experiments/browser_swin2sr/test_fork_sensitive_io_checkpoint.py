import unittest
from pathlib import Path

import torch

from fork_sensitive_io_checkpoint import (
    fork_payload,
    training_state_hashes,
)


def payload():
    return {
        "model": {"weight": torch.arange(4, dtype=torch.float32)},
        "optimizer": {
            "state": {0: {"moment": torch.ones(4)}},
            "param_groups": [
                {"params": [0], "lr": 1e-6, "state_dtype": torch.float32},
            ],
        },
        "run_state": {
            "epoch": 2,
            "torch_rng": torch.arange(8, dtype=torch.uint8),
        },
        "surface_step": 14000,
        "quantized_modules": [
            "swin2sr.first_convolution",
            "upsample.conv",
        ],
        "quantization": {
            "scope": "all_matrix_and_conv",
            "conv_grouping": "kernel",
        },
        "surface_training": {},
    }


class ForkSensitiveIoCheckpointTests(unittest.TestCase):
    def test_fork_changes_only_group_size_metadata(self):
        source = payload()
        before = training_state_hashes(source)
        forked, details = fork_payload(
            source,
            parent_checkpoint=Path("checkpoint-surface-14000.pt"),
            parent_sha256="parent-hash",
            surface="swin2sr.first_convolution",
            group_size=2,
            expect_surface_step=14000,
            expect_current_group_size=9,
        )

        self.assertEqual(training_state_hashes(forked), before)
        self.assertEqual(
            forked["quantization"]["surface_group_sizes"],
            {"swin2sr.first_convolution": 2},
        )
        self.assertEqual(
            forked["surface_training"]["surface_group_sizes"],
            {"swin2sr.first_convolution": 2},
        )
        self.assertEqual(details["training_state_sha256"], before)
        self.assertEqual(
            forked["fork_provenance"]["parent_checkpoint_sha256"],
            "parent-hash",
        )

    def test_fork_rejects_mismatched_or_existing_overrides(self):
        source = payload()
        source["surface_training"]["surface_group_sizes"] = {
            "swin2sr.first_convolution": 2,
        }
        with self.assertRaisesRegex(ValueError, "disagrees"):
            fork_payload(
                source,
                parent_checkpoint=Path("checkpoint.pt"),
                parent_sha256="hash",
                surface="swin2sr.first_convolution",
                group_size=2,
            )

    def test_fork_rejects_wrong_step_and_unknown_surface(self):
        with self.assertRaisesRegex(ValueError, "unexpected surface step"):
            fork_payload(
                payload(),
                parent_checkpoint=Path("checkpoint.pt"),
                parent_sha256="hash",
                surface="swin2sr.first_convolution",
                group_size=2,
                expect_surface_step=13000,
            )
        with self.assertRaisesRegex(ValueError, "not quantized"):
            fork_payload(
                payload(),
                parent_checkpoint=Path("checkpoint.pt"),
                parent_sha256="hash",
                surface="missing",
                group_size=2,
            )

    def test_output_fork_preserves_selected_input_override(self):
        source = payload()
        source["surface_step"] = 19000
        source["quantization"]["surface_group_sizes"] = {
            "swin2sr.first_convolution": 2,
        }
        source["surface_training"]["surface_group_sizes"] = {
            "swin2sr.first_convolution": 2,
        }
        before = training_state_hashes(source)

        forked, details = fork_payload(
            source,
            parent_checkpoint=Path("checkpoint-surface-19000.pt"),
            parent_sha256="selected-input",
            surface="upsample.conv",
            group_size=3,
            expect_surface_step=19000,
            expect_current_group_size=9,
        )

        self.assertEqual(training_state_hashes(forked), before)
        self.assertEqual(details["surface_group_sizes"], {
            "swin2sr.first_convolution": 2,
            "upsample.conv": 3,
        })
        self.assertEqual(details["change"], {
            "surface": "upsample.conv",
            "previous_group_size": 9,
            "group_size": 3,
            "convolution_learning_rate": None,
        })
        self.assertEqual(
            forked["fork_provenance"]["parent_checkpoint_sha256"],
            "selected-input",
        )

    def test_fork_can_change_only_convolution_optimizer_hyperparameter(self):
        source = payload()
        source["optimizer"]["param_groups"] = [
            {
                "params": list(range(192)),
                "role": "matrix_qat8",
                "lr": 1e-6,
            },
            {
                "params": list(range(192, 204)),
                "role": "adamw",
                "lr": 2.5e-7,
            },
            {
                "params": list(range(204, 460)),
                "role": "adamw",
                "lr": 1e-7,
            },
        ]
        source["surface_training"]["optimizer"] = {
            "learning_rate": 1e-6,
        }
        source["optimizer_config"] = {"learning_rate": 1e-6}
        before = training_state_hashes(source)

        forked, details = fork_payload(
            source,
            parent_checkpoint=Path("checkpoint.pt"),
            parent_sha256="hash",
            surface="swin2sr.first_convolution",
            group_size=2,
            expect_current_group_size=9,
            convolution_lr_multiplier=4.0,
        )

        self.assertEqual(training_state_hashes(forked), before)
        self.assertEqual(forked["optimizer"]["param_groups"][1]["lr"], 4e-6)
        self.assertEqual(
            forked["surface_training"][
                "convolution_learning_rate_multiplier"
            ],
            4.0,
        )
        self.assertEqual(
            details["convolution_learning_rate"]["previous_multiplier"],
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
