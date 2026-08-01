import unittest
from pathlib import Path

import torch

from fork_pixel_objective_checkpoint import (
    fork_payload,
    immutable_state_hashes,
)


def checkpoint_payload() -> dict:
    return {
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {
            "state": {0: {"moment": torch.tensor([0.2])}},
            "param_groups": [{"params": [0]}],
        },
        "run_state": {"epoch": 3, "cursor": 17},
        "quantization": {"scope": "all_matrix_and_conv"},
        "quantized_modules": [f"surface-{index}" for index in range(204)],
        "surface_step": 24000,
        "surface_training": {"loss_recipe": "balanced"},
    }


class PixelObjectiveForkTests(unittest.TestCase):
    def test_changes_only_objective_metadata(self):
        payload = checkpoint_payload()
        before = immutable_state_hashes(payload)

        forked, receipt = fork_payload(
            payload,
            parent_checkpoint=Path("parent.pt"),
            parent_sha256="abc",
            pixel_objective="psnr_aligned",
            ground_truth_mse_floor=3.2e-4,
            teacher_mse_floor=2e-5,
            expect_surface_step=24000,
        )

        self.assertEqual(immutable_state_hashes(forked), before)
        self.assertEqual(receipt["immutable_state_sha256"], before)
        self.assertEqual(
            forked["surface_training"]["pixel_objective"],
            "psnr_aligned",
        )

    def test_rejects_nonpositive_floor(self):
        with self.assertRaisesRegex(ValueError, "floors must be positive"):
            fork_payload(
                checkpoint_payload(),
                parent_checkpoint=Path("parent.pt"),
                parent_sha256="abc",
                pixel_objective="psnr_aligned",
                ground_truth_mse_floor=0,
                teacher_mse_floor=2e-5,
                expect_surface_step=24000,
            )

    def test_rejects_wrong_surface_step(self):
        with self.assertRaisesRegex(ValueError, "expected surface step"):
            fork_payload(
                checkpoint_payload(),
                parent_checkpoint=Path("parent.pt"),
                parent_sha256="abc",
                pixel_objective="charbonnier",
                ground_truth_mse_floor=3.2e-4,
                teacher_mse_floor=2e-5,
                expect_surface_step=25000,
            )


if __name__ == "__main__":
    unittest.main()
