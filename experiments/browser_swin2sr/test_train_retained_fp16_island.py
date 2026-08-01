import unittest
from types import SimpleNamespace

from train_retained_fp16_island import (
    FORMAT,
    experiment_config,
    merged_retained_precision_history,
    receipt_without_parameters,
    validate_args,
)


class FakeSampler:
    metadata = {"format": "sealed-sampler", "images": 206_500}


class RetainedFp16TrainingTests(unittest.TestCase):
    def test_config_pins_matched_training_contract(self):
        args = SimpleNamespace(
            island="image_io_fp16",
            precision="fp16",
            seed=20260731,
            steps=4000,
            learning_rate=4e-6,
            batch_size=None,
        )
        training = {
            "batch_size": 8,
            "crop_size": 128,
            "loss_recipe": "balanced",
            "pixel_objective": "psnr_aligned",
        }
        config = experiment_config(args, training, FakeSampler())
        self.assertEqual(config["format"], FORMAT)
        self.assertEqual(config["precision"], "fp16")
        self.assertEqual(config["initialization"], "hard_ternary_then_fp16_roundtrip_ste")
        self.assertEqual(config["trainable_scope"], "declared_island_weights_only")
        self.assertEqual(config["sampler"], FakeSampler.metadata)
        self.assertEqual(config["batch_size"], 8)

    def test_batch_override_is_recorded_in_training_contract(self):
        args = SimpleNamespace(
            island="image_io_fp16",
            precision="fp16",
            seed=20260734,
            steps=1000,
            learning_rate=8e-6,
            batch_size=32,
        )
        training = {
            "batch_size": 8,
            "crop_size": 128,
            "loss_recipe": "balanced",
            "pixel_objective": "psnr_aligned",
        }
        config = experiment_config(args, training, FakeSampler())
        self.assertEqual(config["batch_size"], 32)

    def test_int7_config_pins_deployment_boundary(self):
        args = SimpleNamespace(
            island="rstb_fusion_fp16",
            precision="int7",
            seed=20260803,
            steps=4000,
            learning_rate=1e-6,
            batch_size=8,
            boundary_boost=3.0,
            boundary_temperature=0.1,
            int7_scale_method="mse_iterative",
            int7_scale_blend=0.5,
        )
        training = {
            "batch_size": 8,
            "crop_size": 128,
            "loss_recipe": "balanced",
            "pixel_objective": "psnr_aligned",
        }

        config = experiment_config(args, training, FakeSampler())

        self.assertEqual(config["precision"], "int7")
        self.assertEqual(
            config["initialization"],
            "parent_then_symmetric_int7_per_output_ste",
        )
        self.assertEqual(config["boundary_boost"], 3.0)
        self.assertEqual(config["int7_scale_method"], "mse_iterative")
        self.assertEqual(config["int7_scale_blend"], 0.5)

    def test_receipt_does_not_serialize_parameter_handles(self):
        receipt = receipt_without_parameters({
            "parameters": [object()],
            "trainable_values": 8100,
        })
        self.assertEqual(receipt, {"trainable_values": 8100})

    def test_joint_training_preserves_parent_precision_island(self):
        image = {"island": "image_io_fp16", "precision": "fp16", "seed": 1}
        rstb = {"island": "rstb_fusion_fp16", "precision": "fp16", "seed": 2}
        parent = {"retained_fp16_training": image}
        self.assertEqual(
            merged_retained_precision_history(parent, rstb),
            [image, rstb],
        )

    def test_invalid_step_or_learning_rate_is_rejected(self):
        args = SimpleNamespace(
            steps=0,
            validate_every=500,
            checkpoint_every=500,
            learning_rate=4e-6,
            batch_size=None,
        )
        with self.assertRaises(ValueError):
            validate_args(args)
        args.steps = 4000
        args.learning_rate = 0
        with self.assertRaises(ValueError):
            validate_args(args)
        args.learning_rate = 4e-6
        args.batch_size = 0
        with self.assertRaises(ValueError):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
