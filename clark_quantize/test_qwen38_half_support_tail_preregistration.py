from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38HalfSupportTailPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_half_support_tail_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_is_qwen38_27b_only(self) -> None:
        scope = self.prereg["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(scope["qwen_4b_compute_forbidden"])
        self.assertEqual(scope["affected_checkpoint_tensors"], 27)
        self.assertEqual(scope["affected_parameters"], 133871616)

    def test_real_vision_tail_geometry_is_frozen(self) -> None:
        case = self.prereg["case"]
        self.assertEqual((case["rows"], case["columns"]), (1152, 4304))
        self.assertEqual(case["tail_width"], 80)
        self.assertEqual(case["tail_nonzero_count"], 40)
        self.assertEqual(case["padded_control_columns"], 4352)
        self.assertEqual(self.prereg["representation"]["row_bytes"], 807)

    def test_gate_requires_exact_rate_correctness_and_speed(self) -> None:
        gate = self.prereg["gate"]
        self.assertEqual(gate["maximum_raw_code_bits_per_weight"], 1.5)
        self.assertLessEqual(gate["maximum_absolute_error"], 0.0005)
        self.assertGreaterEqual(
            gate["minimum_throughput_fraction_of_padded_aligned2"], 0.5
        )
        self.assertGreater(gate["minimum_avx2_speedup_vs_scalar"], 1.0)

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
