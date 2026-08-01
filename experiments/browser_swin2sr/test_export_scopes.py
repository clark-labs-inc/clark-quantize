import unittest

import torch
from transformers import Swin2SRForImageSuperResolution

from export_browser_artifact import (
    quantize_tensor,
    selected_categories,
    selected_weights,
)
from quantizer_controls import quantize_ternary_absmean
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


class ExportScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )

    def test_core_export_matches_current_144_tensors(self):
        names = selected_weights(self.model, "core")

        self.assertEqual(len(names), 144)
        self.assertEqual(
            sum(self.model.state_dict()[name].numel() for name in names),
            691_200,
        )

    def test_full_export_selects_204_weight_tensors(self):
        names = selected_weights(self.model, "all_matrix_and_conv")

        self.assertEqual(len(names), 204)
        self.assertEqual(
            sum(self.model.state_dict()[name].numel() for name in names),
            977_604,
        )

    def test_full_export_identifies_only_core_linears_for_hadamard(self):
        categories = selected_categories(self.model, "all_matrix_and_conv")

        self.assertEqual(
            sum(category == "core_linear" for category in categories.values()),
            144,
        )
        self.assertEqual(
            sum(category != "core_linear" for category in categories.values()),
            60,
        )

    def test_absmean_export_does_not_requantize_with_optimal_mse(self):
        values = torch.tensor([[-1.2, -0.4, -0.1], [0.2, 0.8, 1.4]])
        actual = quantize_tensor(
            values,
            mode="ternary",
            method="absmean",
            group_size=3,
        )
        expected = quantize_ternary_absmean(values, 3)

        torch.testing.assert_close(actual.codes, expected.codes)
        torch.testing.assert_close(actual.scales, expected.scales)


if __name__ == "__main__":
    unittest.main()
