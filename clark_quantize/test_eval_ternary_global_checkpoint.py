from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from clark_quantize.qwen_global_qat import (
    GLOBAL_QAT_STATE_SCHEMA,
    capture_replay_state,
    save_global_qat_checkpoint,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval_ternary_ckpt.py"
SPEC = importlib.util.spec_from_file_location("eval_ternary_ckpt", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvalGlobalCheckpointTests(unittest.TestCase):
    def test_loads_atomic_global_checkpoint_for_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            model = {
                "layer.weight_master": torch.arange(6).reshape(2, 3).float(),
                "norm.weight": torch.ones(3),
            }
            replay = capture_replay_state(
                arm_rng_states={"math": {"state": 1}},
                bandit_kl={"math": 0.5},
                bandit_rng_state=(1, 2, 3),
                servo_state={"lr_mult": 1.0},
                kl_history=[],
            )
            checkpoint = save_global_qat_checkpoint(
                root,
                step=5,
                model_state=model,
                optimizer_state={},
                replay_state=replay,
                run_identity={"model": "test"},
            )
            payload, state, metadata = MODULE.load_checkpoint_model_state(
                checkpoint
            )
            self.assertEqual(payload["step"], 5)
            self.assertEqual(metadata["checkpoint_format"], GLOBAL_QAT_STATE_SCHEMA)
            self.assertEqual(state.keys(), model.keys())
            for name in model:
                torch.testing.assert_close(state[name], model[name])


if __name__ == "__main__":
    unittest.main()
