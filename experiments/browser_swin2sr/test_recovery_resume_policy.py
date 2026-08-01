import unittest

from recovery_resume_policy import sampler_resume_mode


class RecoveryResumePolicyTests(unittest.TestCase):
    def test_changed_corpus_requires_explicit_reset(self):
        previous = {"corpus": "old"}
        current = {"corpus": "glyph-safe"}

        with self.assertRaisesRegex(ValueError, "without an explicit"):
            sampler_resume_mode(previous, current, False)
        self.assertEqual(
            sampler_resume_mode(previous, current, True),
            "reset",
        )

    def test_same_corpus_resumes_exactly(self):
        sampler = {"corpus": "glyph-safe"}

        self.assertEqual(sampler_resume_mode(sampler, sampler, False), "exact")

    def test_new_run_is_fresh(self):
        self.assertEqual(
            sampler_resume_mode(None, {"corpus": "new"}, False),
            "fresh",
        )


if __name__ == "__main__":
    unittest.main()
