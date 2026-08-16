from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/qwen38_27b_spot/layer_sensitivity_gate.py"
SPEC = importlib.util.spec_from_file_location("qwen38_layer_sensitivity_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LayerSensitivityTests(unittest.TestCase):
    def test_selects_best_recovery_per_gib_then_absolute(self) -> None:
        rows = [
            {"layer": 0, "teacher_kl_recovery_per_gib": 0.2, "teacher_kl_recovery": 0.1},
            {"layer": 3, "teacher_kl_recovery_per_gib": 0.4, "teacher_kl_recovery": 0.05},
            {"layer": 31, "teacher_kl_recovery_per_gib": 0.4, "teacher_kl_recovery": 0.08},
        ]
        self.assertEqual(MODULE.select_layer(rows)["layer"], 31)

    def test_rejects_empty_ranking(self) -> None:
        with self.assertRaisesRegex(ValueError, "no layer"):
            MODULE.select_layer([])

    def test_sensitivity_candidates_and_rule_are_sealed(self) -> None:
        prereg = json.loads((Path(__file__).resolve().parents[1] / "experiments/qwen38_27b_spot/qat_preregistration.json").read_text())
        candidates = MODULE.validate_sensitivity_preregistration(prereg, [0, 3, 31, 63])
        self.assertEqual(set(candidates), {0, 3, 31, 63})
        changed = json.loads(json.dumps(prereg))
        changed["sensitivity_candidates"][1]["layer"] = 4
        with self.assertRaisesRegex(ValueError, "differ"):
            MODULE.validate_sensitivity_preregistration(changed, [0, 3, 31, 63])


if __name__ == "__main__":
    unittest.main()
