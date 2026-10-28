from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
import unittest

from clark_quantize.qwen38_ternary_entropy import (
    conservative_type_class_payload_bits,
    empirical_entropy_bits_per_trit,
    exact_type_class_payload_bits_small,
    ternary_type_class_ledger,
)


class Qwen38TernaryEntropyTests(unittest.TestCase):
    def test_qwen_sample_is_below_uniform_base3(self) -> None:
        counts = (266670, 466137, 267193)
        entropy = empirical_entropy_bits_per_trit(counts)
        self.assertAlmostEqual(entropy, 1.5305514115778136)
        self.assertLess(entropy, math.log2(3))
        self.assertGreater(math.log2(3) - entropy, 0.05)

    def test_conservative_bound_covers_exact_type_class(self) -> None:
        for counts in ((1, 1, 1), (2, 3, 5), (0, 4, 7), (8, 1, 1), (30, 40, 30)):
            self.assertGreaterEqual(
                conservative_type_class_payload_bits(counts),
                exact_type_class_payload_bits_small(counts),
            )

    def test_header_and_payload_are_counted(self) -> None:
        ledger = ternary_type_class_ledger((30, 40, 30))
        self.assertEqual(ledger.trits, 100)
        self.assertEqual(ledger.header_bytes, 16)
        self.assertEqual(
            ledger.section_bytes_before_tensor_alignment,
            16 + (ledger.conservative_payload_bits + 7) // 8,
        )
        self.assertGreater(ledger.effective_bits_per_trit, ledger.entropy_bits_per_trit)

    def test_degenerate_source_still_pays_finite_block_metadata(self) -> None:
        ledger = ternary_type_class_ledger((0, 1000, 0))
        self.assertEqual(ledger.entropy_bits_per_trit, 0.0)
        self.assertEqual(ledger.conservative_payload_bits, 64)
        self.assertEqual(ledger.section_bytes_before_tensor_alignment, 24)

    def test_rejects_invalid_counts_or_header(self) -> None:
        with self.assertRaises(ValueError):
            empirical_entropy_bits_per_trit((1, 2))
        with self.assertRaises(ValueError):
            empirical_entropy_bits_per_trit((0, 0, 0))
        with self.assertRaises(ValueError):
            ternary_type_class_ledger((1, 1, 1), header_bytes=8)

    def test_rate_preregistration_freezes_complete_qwen38_denominators(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        path = repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_ternary_type_class_rate_preregistration_v1.json"
        )
        preregistration = json.loads(path.read_text())
        scope = preregistration["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertEqual(scope["mapped_ternary_codes"], 26869760000)
        self.assertEqual(scope["complete_parameters"], 27781427952)
        self.assertLessEqual(
            preregistration["gate"]["maximum_projected_complete_bundle_bpw"], 1.59
        )
        for descriptor in preregistration["frozen_inputs"].values():
            source = repo / descriptor["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), descriptor["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
