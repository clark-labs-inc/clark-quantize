import unittest

import torch

from analyze_surface_signatures import signature_counts, summarize


class SurfaceSignatureTests(unittest.TestCase):
    def test_counts_support_changes_and_sign_flips(self):
        base = torch.tensor([1.0, -1.0, 0.2, -0.2])
        master = torch.tensor([0.8, -0.8, -0.3, 0.0])
        initial = torch.tensor([1, -1, 0, 0], dtype=torch.int8)
        current = torch.tensor([1, 0, -1, 0], dtype=torch.int8)

        result = summarize(signature_counts(base, master, initial, current))

        self.assertEqual(result["changed_codes"], 2)
        self.assertEqual(result["zero_codes"], 2)
        self.assertEqual(result["direct_sign_flips"], 0)
        self.assertEqual(result["sign_inheritance"], 0.5)


if __name__ == "__main__":
    unittest.main()
