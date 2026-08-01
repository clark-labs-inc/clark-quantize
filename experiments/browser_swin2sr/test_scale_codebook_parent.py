import unittest

from retained_precision_state import CHECKPOINT_FORMAT
from scale_codebook_parent import (
    scale_parent_sampler_step,
    scale_parent_total_step,
    scale_parent_training,
)


class ScaleCodebookParentTests(unittest.TestCase):
    def test_surface_parent_contract_is_unchanged(self):
        payload = {
            "format": "surface-parent",
            "surface_step": 28000,
            "surface_training": {"seed": 1, "batch_size": 8},
        }
        self.assertEqual(scale_parent_training(payload), payload["surface_training"])
        self.assertEqual(scale_parent_sampler_step(payload), 28000)
        self.assertEqual(scale_parent_total_step(payload), 28000)

    def test_retained_parent_uses_its_exact_sampler_contract(self):
        payload = {
            "format": CHECKPOINT_FORMAT,
            "step": 32000,
            "island_step": 4000,
            "surface_training": {
                "seed": 1,
                "batch_size": 8,
                "crop_size": 128,
                "loss_recipe": "old",
                "pixel_objective": "old",
                "convolution_learning_rate_multiplier": 1.0,
            },
            "retained_fp16_training": {
                "seed": 20260733,
                "batch_size": 8,
                "crop_size": 128,
                "loss_recipe": "balanced",
                "pixel_objective": "psnr_aligned",
            },
        }
        training = scale_parent_training(payload)
        self.assertEqual(training["seed"], 20260733)
        self.assertEqual(training["loss_recipe"], "balanced")
        self.assertEqual(training["convolution_learning_rate_multiplier"], 1.0)
        self.assertEqual(scale_parent_sampler_step(payload), 4000)
        self.assertEqual(scale_parent_total_step(payload), 32000)


if __name__ == "__main__":
    unittest.main()
