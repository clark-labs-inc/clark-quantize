import unittest

import torch

from mixed_precision_artifact import (
    recover_symmetric_per_output,
    validate_precision_contract,
)


class MixedPrecisionArtifactTests(unittest.TestCase):
    def test_recovers_exact_int7_codes_and_fp16_scales(self):
        scales = torch.tensor([0.25, 0.125])
        codes = torch.arange(-18, 18).reshape(2, 2, 3, 3)
        codes[0, 0, 0, 0] = -63
        codes[1, 0, 0, 0] = 63
        weight = codes * scales[:, None, None, None]

        recovered_codes, recovered_scales, error = recover_symmetric_per_output(
            weight, 7
        )

        self.assertEqual(error, 0.0)
        self.assertEqual(recovered_codes.shape, (2, 2, 3, 3))
        self.assertTrue((recovered_scales == scales.numpy()).all())

    def test_validates_disjoint_fp16_and_int7_islands(self):
        state = {
            "io.weight": torch.ones(1, dtype=torch.float16).float(),
            "rstb.weight": torch.zeros(1, 1, 3, 3),
        }
        materialization = {
            "retained_precision_islands": [
                {
                    "name": "io",
                    "precision": "fp16",
                    "storage_dtype": "float16",
                    "weight_values": 1,
                    "weights": ["io.weight"],
                },
                {
                    "name": "rstb",
                    "precision": "int7",
                    "storage_dtype": "packed_int7",
                    "bit_width": 7,
                    "granularity": "per_output",
                    "weight_values": 9,
                    "weights": ["rstb.weight"],
                },
            ]
        }

        contract = validate_precision_contract(
            materialization, state, set(state)
        )

        self.assertEqual(contract.fp16_weights, {"io.weight"})
        self.assertEqual(contract.int7_weights, {"rstb.weight"})


if __name__ == "__main__":
    unittest.main()
