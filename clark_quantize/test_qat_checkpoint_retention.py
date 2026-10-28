from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from clark_quantize.qwen_qat import save_qat_checkpoint


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/spot/prune_qat_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("prune_qat_checkpoints", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CheckpointRetentionTests(unittest.TestCase):
    def complete(self, root: Path, step: int) -> Path:
        path = root / f"checkpoint-{step}"
        path.mkdir()
        (path / "weights.pt").write_bytes(b"checkpoint")
        (path / "checkpoint.json").write_text(json.dumps({
            "schema": MODULE.SCHEMA,
            "complete": True,
            "step": step,
        }))
        return path

    def test_keeps_newest_two_complete_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (100, 200, 300, 400):
                self.complete(root, step)
            partial = root / "checkpoint-500"
            partial.mkdir()
            receipt = MODULE.prune(root, keep=2)
            self.assertEqual(receipt["removed"], ["checkpoint-100", "checkpoint-200"])
            self.assertEqual(receipt["retained"], ["checkpoint-300", "checkpoint-400"])
            self.assertEqual(receipt["ignored_incomplete"], ["checkpoint-500"])
            self.assertTrue(partial.is_dir())
            self.assertFalse((root / "checkpoint-100").exists())

    def test_dry_run_and_invalid_manifest_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.complete(root, 1)
            self.complete(root, 2)
            self.complete(root, 3)
            receipt = MODULE.prune(root, keep=2, dry_run=True)
            self.assertEqual(receipt["removed"], ["checkpoint-1"])
            self.assertTrue((root / "checkpoint-1").is_dir())
            (root / "checkpoint-2/checkpoint.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "invalid completed-checkpoint"):
                MODULE.prune(root, keep=2)

    def test_rejects_relative_root_and_symlink(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute"):
            MODULE.prune(Path("relative"), keep=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "elsewhere"
            target.mkdir()
            (root / "checkpoint-1").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                MODULE.prune(root, keep=2)

    def test_prunes_atomic_trainer_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for step in (10, 20, 30):
                masters = {"weight": torch.full((2, 3), float(step))}
                save_qat_checkpoint(
                    root,
                    step=step,
                    state={"masters": masters},
                    tensor_metadata={"weight": {"shape": [2, 3], "dtype": "torch.float32"}},
                    metadata={"test": True},
                )
            receipt = MODULE.prune(root, keep=2)
            self.assertEqual(receipt["removed"], ["checkpoint-10"])
            self.assertEqual(receipt["retained"], ["checkpoint-20", "checkpoint-30"])


if __name__ == "__main__":
    unittest.main()
