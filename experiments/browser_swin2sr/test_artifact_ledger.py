from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact_ledger import build_artifact_ledger


class ArtifactLedgerTest(unittest.TestCase):
    def test_reports_payload_overhead_separately(self) -> None:
        ledger = build_artifact_ledger(
            mode="ternary",
            binary_bytes=250,
            parameter_values=1000,
            state_values=1100,
            quantized_values=800,
            code_bytes=160,
            scale_bytes=20,
            offset_bytes=4,
            raw_bytes=60,
        )

        self.assertAlmostEqual(
            ledger["ideal_code_bits_per_quantized_weight"],
            math.log2(3),
        )
        self.assertEqual(ledger["packed_code_bits_per_quantized_weight"], 1.6)
        self.assertEqual(ledger["scale_bits_per_quantized_weight"], 0.2)
        self.assertEqual(ledger["offset_bytes"], 4)
        self.assertEqual(ledger["alignment_bytes"], 6)
        self.assertEqual(ledger["whole_model_data_bits_per_parameter"], 2.0)

    def test_rejects_inconsistent_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_artifact_ledger(
                mode="binary",
                binary_bytes=9,
                parameter_values=10,
                state_values=10,
                quantized_values=10,
                code_bytes=4,
                scale_bytes=4,
                raw_bytes=4,
            )


if __name__ == "__main__":
    unittest.main()
