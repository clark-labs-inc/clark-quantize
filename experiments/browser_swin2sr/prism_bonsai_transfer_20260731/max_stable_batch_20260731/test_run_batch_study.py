import unittest

from run_batch_study import accepted, refinement_candidate


class BatchStudyTests(unittest.TestCase):
    def test_acceptance_requires_explicit_probe_acceptance(self):
        self.assertTrue(accepted({"status": "accepted"}))
        self.assertFalse(accepted({"status": "rejected"}))
        self.assertFalse(accepted(None))

    def test_refinement_uses_sealed_quantum(self):
        self.assertEqual(refinement_candidate(64, 128, 8), 96)
        self.assertEqual(refinement_candidate(96, 128, 8), 112)
        self.assertIsNone(refinement_candidate(120, 128, 8))


if __name__ == "__main__":
    unittest.main()
