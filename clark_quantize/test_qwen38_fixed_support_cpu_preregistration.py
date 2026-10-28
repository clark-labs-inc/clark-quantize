from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38FixedSupportCPUPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_fixed_support_cpu_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_is_qwen38_27b_only(self) -> None:
        scope = self.prereg["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(scope["qwen_4b_compute_forbidden"])

    def test_all_frozen_cases_are_qwen_input_aligned(self) -> None:
        self.assertEqual(len(self.prereg["cases"]), 4)
        self.assertEqual(
            {case["columns"] for case in self.prereg["cases"]},
            {5120, 17408},
        )
        self.assertEqual(
            {case["nonzero_per_group"] for case in self.prereg["cases"]},
            {54, 60, 64},
        )

    def test_gate_is_load_time_only(self) -> None:
        gate = self.prereg["gate"]
        self.assertGreater(gate["minimum_unpack_mweights_per_second"], 0)
        self.assertLessEqual(gate["maximum_code_bits_per_weight"], 1.48)
        self.assertIn("not a fused GEMV", self.prereg["scope"]["claim_boundary"])

    def test_every_frozen_source_hash_matches(self) -> None:
        for descriptor in self.prereg["frozen_inputs"].values():
            source = self.repo / descriptor["path"]
            self.assertNotEqual(descriptor["sha256"], "PLACEHOLDER")
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                descriptor["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
