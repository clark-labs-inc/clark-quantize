import unittest

from PIL import Image

from render_sensitive_io_comparison import labelled_montage


class RenderSensitiveIoComparisonTests(unittest.TestCase):
    def test_montage_accepts_arbitrary_candidate_columns(self):
        image = Image.new("RGB", (16, 8), "black")

        result = labelled_montage([
            ("input", image),
            ("g9", image),
            ("g2", image),
            ("g1", image),
            ("truth", image),
        ])

        self.assertEqual(result.size, (80, 32))

    def test_montage_rejects_mismatched_images(self):
        with self.assertRaisesRegex(ValueError, "equal dimensions"):
            labelled_montage([
                ("a", Image.new("RGB", (16, 8))),
                ("b", Image.new("RGB", (8, 8))),
            ])


if __name__ == "__main__":
    unittest.main()
