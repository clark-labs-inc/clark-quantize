import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from probe_sensitive_io_group_sizes import (
    IDEAL_TERNARY_BITS,
    parse_group_sizes,
    storage_receipt,
    study_sealed_source_root,
    target_signature,
)


class SensitiveIoGroupProbeTests(unittest.TestCase):
    def test_group_sizes_must_be_positive_and_unique(self):
        self.assertEqual(parse_group_sizes("1,3,9"), (1, 3, 9))
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_group_sizes("1,0,9")
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_group_sizes("1,3,3")

    def test_finer_scales_report_whole_model_overhead(self):
        receipt = storage_receipt(
            values=1620,
            group_size=3,
            total_quantized_values=977604,
            reference_group_size=9,
        )
        self.assertEqual(receipt["scale_count"], 540)
        self.assertAlmostEqual(
            receipt["layer_bits_per_weight"],
            IDEAL_TERNARY_BITS + 16 / 3,
        )
        self.assertAlmostEqual(
            receipt["whole_model_bit_delta_vs_reference"],
            5760 / 977604,
        )
        self.assertTrue(math.isfinite(receipt["layer_bits_per_weight"]))

    def test_study_specific_sealed_source_root_is_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            manifest = repository / "study.json"
            resolved = study_sealed_source_root(
                {"sealed_source_root": "sealed/arm"},
                manifest,
                repository,
            )
            self.assertEqual(resolved, repository / "sealed/arm")

    def test_target_signature_tracks_each_projection_grid_point(self):
        base = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        master = torch.tensor([-2.0, -0.1, 0.1, 2.0])
        quantizer = SimpleNamespace(method="absmean", group_size=4)

        signature = target_signature(base, master, quantizer)

        self.assertEqual(signature["values"], 4)
        self.assertGreater(signature["changed_codes"], 0)
        self.assertGreater(signature["zero_fraction"], 0)
        self.assertGreater(signature["relative_master_movement"], 0)
        self.assertEqual(signature["direct_sign_flips"], 0)


if __name__ == "__main__":
    unittest.main()
