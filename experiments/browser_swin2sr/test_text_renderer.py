import tempfile
import unittest
from pathlib import Path

from PIL import Image

from text_renderer import colorize_mask


class TextRendererTests(unittest.TestCase):
    def test_colorize_mask_converts_black_to_opaque_foreground(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = root / "mask.png"
            output = root / "output.png"
            Image.new("L", (2, 1), 255).save(mask)
            with Image.open(mask) as source:
                pixels = source.load()
                pixels[0, 0] = 0
                source.save(mask)
            colorize_mask(mask, output, "#123456")
            with Image.open(output) as rendered:
                self.assertEqual(rendered.getpixel((0, 0)), (18, 52, 86, 255))
                self.assertEqual(rendered.getpixel((1, 0)), (18, 52, 86, 0))


if __name__ == "__main__":
    unittest.main()
