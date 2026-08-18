from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38GainHashLayerTranscodePreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_gain_hash_layer_transcode_preregistration_v1.json"
        )
        cls.payload = json.loads(cls.path.read_text())

    def test_scope_and_claim_boundary_are_qwen_27b_only(self) -> None:
        self.assertEqual(
            self.payload["scope"]["active_model_only"], "Qwen/Qwen3.8-27B"
        )
        self.assertTrue(self.payload["scope"]["qwen_4b_compute_forbidden"])
        self.assertEqual(self.payload["scope"]["primary_shape"], [17408, 5120])
        self.assertEqual(self.payload["scope"]["also_covers_shape"], [5120, 17408])
        self.assertIn("not real weights", self.payload["scope"]["claim_boundary"])
        self.assertIn(
            "not evidence of a speed improvement",
            self.payload["disclosed_non_authoritative_arm64_pilot"][
                "scientific_status"
            ],
        )

    def test_gate_preserves_exactness_and_memory_latency_tradeoff(self) -> None:
        gate = self.payload["gate"]
        self.assertTrue(gate["require_direct_output_hash_exact"])
        self.assertTrue(gate["require_dense_control_output_hash_exact"])
        self.assertEqual(gate["minimum_peak_rss_saving_bytes"], 64 * 1024 * 1024)
        self.assertEqual(gate["maximum_direct_peak_rss_bytes"], 256 * 1024 * 1024)
        self.assertEqual(gate["maximum_direct_median_seconds"], 5.0)
        self.assertEqual(
            gate["maximum_direct_slowdown_ratio_vs_dense_control"], 1.75
        )
        self.assertNotIn("minimum_speedup", gate)

    def test_every_frozen_input_hash_matches(self) -> None:
        for name, record in self.payload["frozen_inputs"].items():
            path = self.repo / record["path"]
            self.assertTrue(path.is_file(), msg=name)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["sha256"],
                msg=name,
            )

    def test_resource_envelope_is_single_core_zero_swap_and_disk_bounded(self) -> None:
        resources = self.payload["resources"]
        self.assertEqual(resources["authoritative_architecture"], "x86_64")
        self.assertEqual(resources["cpu_threads"], 1)
        self.assertEqual(resources["cpu_quota_percent"], 100)
        self.assertEqual(resources["memory_max_gib"], 1)
        self.assertTrue(resources["swap_must_remain_zero"])
        self.assertGreaterEqual(resources["minimum_free_disk_gib"], 10)


if __name__ == "__main__":
    unittest.main()
