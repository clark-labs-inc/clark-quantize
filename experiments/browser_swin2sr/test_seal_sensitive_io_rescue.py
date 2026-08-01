import unittest
from argparse import Namespace
from pathlib import Path

from seal_sensitive_io_rescue import training_command


class SealSensitiveIoRescueTests(unittest.TestCase):
    def test_resume_command_preserves_preregistered_controls(self):
        args = Namespace(
            fork_checkpoint=Path("/study/fork.pt"),
            validation_data=Path("/promos"),
            study_root=Path("/study"),
            arm="io-g2-lr025",
            steps=19000,
            schedule_steps=24000,
            batch_size=8,
            crop_size=128,
            learning_rate=1e-6,
            loss_recipe="balanced",
            validate_every=500,
            checkpoint_every=1000,
            hard_shadow_weight_start=0.05,
            hard_shadow_weight_end=0.25,
            hard_shadow_every=2,
            device="mps",
            seed=20260730,
            data=[Path("/ui"), Path("/text")],
        )

        command = training_command(args)

        self.assertEqual(command[command.index("--steps") + 1], "19000")
        self.assertEqual(
            command[command.index("--base-checkpoint") + 1],
            "/study/fork.pt",
        )
        self.assertNotIn("--initialization", command)
        self.assertEqual(command.count("--data"), 2)


if __name__ == "__main__":
    unittest.main()
