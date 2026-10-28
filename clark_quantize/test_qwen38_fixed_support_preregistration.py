from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38FixedSupportPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_fixed_support_codec_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_and_parent_are_frozen_to_qwen38(self) -> None:
        self.assertEqual(self.prereg["scope"]["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(self.prereg["scope"]["qwen_4b_compute_forbidden"])
        self.assertEqual(self.prereg["gate"]["required_parameters"], 442368)

    def test_candidate_widths_match_exact_enumerative_space(self) -> None:
        self.assertEqual(
            self.prereg["codec"]["candidate_widths"],
            {
                "hash128_sparse64": 189,
                "hash64_sparse60": 184,
                "hash32_sparse54": 176,
            },
        )

    def test_rate_gate_remains_structural_and_conservative(self) -> None:
        gate = self.prereg["gate"]
        self.assertLessEqual(gate["maximum_structural_projected_bits_per_weight"], 1.56)
        self.assertGreaterEqual(gate["minimum_bits_per_weight_reduction_vs_parent_projection"], 0.015)
        self.assertIn("not a complete-layer", self.prereg["scope"]["claim_boundary"])

    def test_frozen_sources_match(self) -> None:
        for descriptor in self.prereg["frozen_inputs"].values():
            source = self.repo / descriptor["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), descriptor["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
