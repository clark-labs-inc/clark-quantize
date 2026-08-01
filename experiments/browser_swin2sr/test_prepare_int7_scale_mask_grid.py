import unittest

import torch

from prepare_int7_scale_mask_grid import STAGE_WEIGHTS, compose_mask, validate_endpoints


class PrepareInt7ScaleMaskGridTests(unittest.TestCase):
    def setUp(self):
        self.maximum = {"shared": torch.tensor([5.0])}
        self.lloyd = {"shared": torch.tensor([5.0])}
        for stage, name in enumerate(STAGE_WEIGHTS):
            self.maximum[name] = torch.tensor([float(stage)])
            self.lloyd[name] = torch.tensor([float(stage + 10)])

    def test_compose_mask_uses_bits_in_stage_order(self):
        state = compose_mask(self.maximum, self.lloyd, 0b0101)
        self.assertEqual(state[STAGE_WEIGHTS[0]].item(), 10.0)
        self.assertEqual(state[STAGE_WEIGHTS[1]].item(), 1.0)
        self.assertEqual(state[STAGE_WEIGHTS[2]].item(), 12.0)
        self.assertEqual(state[STAGE_WEIGHTS[3]].item(), 3.0)
        self.assertEqual(state["shared"].item(), 5.0)

    def test_validate_endpoints_rejects_non_factor_drift(self):
        changed = dict(self.lloyd)
        changed["shared"] = torch.tensor([6.0])
        with self.assertRaisesRegex(ValueError, "non-factor tensor differs"):
            validate_endpoints(self.maximum, changed)

    def test_validate_endpoints_accepts_only_stage_differences(self):
        validate_endpoints(self.maximum, self.lloyd)


if __name__ == "__main__":
    unittest.main()
