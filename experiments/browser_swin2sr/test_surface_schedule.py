import unittest
from collections import Counter

from transformers import Swin2SRForImageSuperResolution

from surface_schedule import category_schedule, surface_targets
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


class SurfaceScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )

    def test_core_scope_exactly_matches_current_training_surface(self):
        targets = surface_targets(self.model, "core")

        self.assertEqual(len(targets), 144)
        self.assertEqual(sum(target.values for target in targets), 691_200)

    def test_full_scope_covers_every_matrix_and_convolution_weight(self):
        targets = surface_targets(self.model, "all_matrix_and_conv")
        counts = Counter(target.category for target in targets)

        self.assertEqual(counts, {
            "core_linear": 144,
            "position_bias_linear": 48,
            "residual_body_conv": 10,
            "image_io_conv": 2,
        })
        self.assertEqual(sum(target.values for target in targets), 977_604)

    def test_image_facing_convolutions_enter_only_after_other_surfaces(self):
        targets = surface_targets(self.model, "all_matrix_and_conv")
        image_io = [target for target in targets if target.category == "image_io_conv"]
        residual = [target for target in targets if target.category == "residual_body_conv"]

        self.assertGreater(
            min(target.start_fraction for target in image_io),
            max(target.start_fraction for target in residual),
        )
        self.assertTrue(all(target.end_fraction < 1 for target in targets))

    def test_sensitive_profile_separates_input_and_output_transitions(self):
        targets = surface_targets(
            self.model,
            "all_matrix_and_conv",
            "sensitive_io_v2",
        )
        input_target = next(
            target
            for target in targets
            if target.name == "swin2sr.first_convolution"
        )
        output_target = next(
            target for target in targets if target.name == "upsample.conv"
        )
        self.assertEqual(
            (input_target.start_fraction, input_target.end_fraction),
            (0.62, 0.79),
        )
        self.assertEqual(
            (output_target.start_fraction, output_target.end_fraction),
            (0.80, 0.98),
        )
        self.assertLess(input_target.end_fraction, output_target.start_fraction)

    def test_sensitive_profile_starts_with_dense_warmup(self):
        attention = category_schedule(
            "swin2sr.encoder.stages.0.layers.0.attention.self.query",
            "core_linear",
            "sensitive_io_v2",
        )
        self.assertEqual(attention, (0.03, 0.15))


if __name__ == "__main__":
    unittest.main()
