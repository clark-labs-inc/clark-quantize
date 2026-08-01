import unittest
from collections import Counter

from transformers import Swin2SRForImageSuperResolution

from island_policy import island_parameters
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


class IslandPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )

    def test_frozen_control_selects_no_compensation_parameters(self):
        self.assertEqual(island_parameters(self.model, "frozen"), [])

    def test_norm_policy_selects_only_layer_norm_parameters(self):
        parameters = island_parameters(self.model, "norms")

        self.assertEqual(Counter(parameter.role for parameter in parameters), {
            "normalization": 100,
        })
        self.assertEqual(sum(parameter.values for parameter in parameters), 6_000)
        self.assertTrue(all(parameter.learning_rate_multiplier == 0.10 for parameter in parameters))

    def test_residual_policy_never_trains_image_io_convolutions(self):
        parameters = island_parameters(self.model, "norms_and_residual")
        names = {parameter.name for parameter in parameters}

        self.assertEqual(
            Counter(parameter.role for parameter in parameters),
            {"normalization": 100, "residual_body": 20},
        )
        self.assertFalse(any(name.startswith("swin2sr.first_convolution") for name in names))
        self.assertFalse(any(name.startswith("upsample.conv") for name in names))
        self.assertEqual(
            sum(parameter.values for parameter in parameters if parameter.role == "residual_body"),
            180_600,
        )


if __name__ == "__main__":
    unittest.main()
