"""Regression tests for balanced image discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from train_swin2sr_qat import discover_images


class DataDiscoveryTest(unittest.TestCase):
    def test_small_valid_png_is_not_silently_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small = root / "small-script.png"
            Image.new("RGB", (64, 64), "white").save(small)

            self.assertLess(small.stat().st_size, 4096)
            self.assertEqual(discover_images([root]), [small])


if __name__ == "__main__":
    unittest.main()
