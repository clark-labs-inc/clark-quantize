from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/qwen38_27b_spot/one_layer_qat_100.py"
SPEC = importlib.util.spec_from_file_location("one_layer_qat_100", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OneLayerQAT100Tests(unittest.TestCase):
    def test_second_rung_is_sealed(self) -> None:
        prereg = json.loads((ROOT / "experiments/qwen38_27b_spot/qat_preregistration.json").read_text())
        second = MODULE.validate_second_rung(prereg)
        self.assertEqual(second["evaluation_steps"], list(range(20, 101, 10)))
        changed = json.loads(json.dumps(prereg))
        changed["second_rung"]["optimizer_steps_total"] = 99
        with self.assertRaisesRegex(ValueError, "contract changed"):
            MODULE.validate_second_rung(changed)

    def test_regression_streak_requires_consecutive_worsening(self) -> None:
        streak = MODULE.update_regression_streak(1.0, 1.1, 0)
        self.assertEqual(streak, 1)
        streak = MODULE.update_regression_streak(1.1, 1.2, streak)
        self.assertEqual(streak, 2)
        self.assertEqual(MODULE.update_regression_streak(1.2, 1.1, streak), 0)
        with self.assertRaisesRegex(ValueError, "invalid"):
            MODULE.update_regression_streak(float("nan"), 1.0, 0)


if __name__ == "__main__":
    unittest.main()
