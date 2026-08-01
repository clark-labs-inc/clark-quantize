import unittest

import torch

from calibrate_scale_fragility import (
    output_fragility,
    scale_candidate,
    weighted_quotas,
)


class ScaleFragilityCalibrationTests(unittest.TestCase):
    def test_weighted_quotas_are_exact(self):
        result = weighted_quotas(10, {"a": 0.7, "b": 0.3})
        self.assertEqual(result, {"a": 7, "b": 3})

    def test_dense_fp16_scale_candidate_preserves_ternary_codes(self):
        weight = torch.tensor([[-0.30001, 0.0, 0.12503, -0.12503]])
        candidate, receipt = scale_candidate(weight, 2, None)
        self.assertEqual(receipt["label"], "fp16")
        self.assertEqual(receipt["groups"], 2)
        self.assertEqual(receipt["storage_bytes"], 4)
        self.assertTrue(torch.equal(torch.sign(candidate), torch.sign(weight)))
        self.assertTrue(torch.equal(candidate, candidate.half().float()))

    def test_identical_output_has_zero_fragility(self):
        output = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
        result = output_fragility(output, output.clone())
        self.assertEqual(result["normalized_output_mse"], 0.0)
        self.assertEqual(result["normalized_edge_mse"], 0.0)
        self.assertEqual(result["fragility"], 0.0)


if __name__ == "__main__":
    unittest.main()
