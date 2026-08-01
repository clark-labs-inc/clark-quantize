import unittest

from transformers import Swin2SRForImageSuperResolution

from model_surface_ledger import build_ledger
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


class ModelSurfaceLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        cls.ledger = build_ledger(cls.model)

    def test_accounts_for_every_parameter(self):
        expected = sum(parameter.numel() for parameter in self.model.parameters())

        self.assertEqual(self.ledger["parameters"], expected)
        self.assertEqual(expected, 1_005_348)
        self.assertEqual(
            sum(category["values"] for category in self.ledger["categories"].values()),
            expected,
        )

    def test_current_core_matches_existing_quantized_surface(self):
        self.assertEqual(
            self.ledger["categories"]["core_linear"]["values"],
            691_200,
        )

    def test_full_weight_policy_quantizes_over_97_percent(self):
        policies = self.ledger["policies"]["all_matrix_and_conv"]
        group_240_int8_other = next(
            policy
            for policy in policies
            if policy["group_size"] == 240
            and policy["other_parameter_precision"] == 8
        )

        self.assertGreater(group_240_int8_other["ternary_fraction"], 0.97)
        self.assertLess(
            group_240_int8_other["whole_model_bits_per_parameter"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
