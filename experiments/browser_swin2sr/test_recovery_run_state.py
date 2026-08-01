import unittest

from recovery_run_state import (
    checkpoint_recovery_config,
    recovery_offsets,
)


class RecoveryRunStateTests(unittest.TestCase):
    def test_new_recovery_starts_at_zero_despite_base_global_step(self):
        self.assertEqual(recovery_offsets({"step": 5000}), (5000, 0))

    def test_checkpoint_tracks_cumulative_recovery_steps(self):
        config = {"initial_global_step": 5000}
        recovery = checkpoint_recovery_config(config, 5500)
        payload = {"step": 5500, "recovery": recovery}

        self.assertEqual(recovery_offsets(payload), (5000, 500))

    def test_inconsistent_step_accounting_fails_closed(self):
        payload = {
            "step": 5600,
            "recovery": {
                "initial_global_step": 5000,
                "recovery_steps_completed": 500,
            },
        }

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            recovery_offsets(payload)


if __name__ == "__main__":
    unittest.main()
