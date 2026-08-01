import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_training_corpus import validate


class TrainingCorpusValidationTests(unittest.TestCase):
    def build_corpus(self, root: Path) -> tuple[Path, Path]:
        font = root / "font.ttf"
        config = root / "fonts.conf"
        font.write_bytes(b"font")
        config.write_text("<fontconfig/>")
        records = []
        for index, language in enumerate(("eng_Latn", "fra_Latn")):
            image = root / f"{language}.png"
            image.write_bytes(f"image-{index}".encode())
            records.append({
                "language": language,
                "local_path": image.name,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "width": 320,
                "height": 240,
                "hf_source": "flores",
                "hf_revision": "revision",
            })
        (root / "manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        (root / "summary.json").write_text(json.dumps({
            "images": 2,
            "languages": 2,
            "images_per_language": 1,
            "resolutions": [[320, 240]],
            "source": "flores",
            "source_revision": "revision",
            "text_renderer": "pango-view",
            "reject_unknown_glyphs": True,
            "font_assets": {
                "font": str(font),
                "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
                "config": str(config),
            },
        }))
        heldout_manifest = root / "heldout.jsonl"
        heldout_manifest.write_text(
            json.dumps({"source_sha256": "heldout"}) + "\n"
        )
        bundle = root / "bundle.json"
        bundle.write_text(json.dumps({
            "manifest": str(heldout_manifest),
            "manifest_sha256": hashlib.sha256(
                heldout_manifest.read_bytes()
            ).hexdigest(),
        }))
        return bundle, root / "manifest.jsonl"

    def test_seals_equal_disjoint_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, _ = self.build_corpus(root)
            receipt = validate(root, 2, 2, 1, [bundle])

            self.assertEqual(receipt["images"], 2)
            self.assertEqual(receipt["languages"], 2)
            self.assertEqual(receipt["heldout_hash_overlap"], 0)
            self.assertEqual(
                receipt["images_per_language_min"],
                receipt["images_per_language_max"],
            )

    def test_rejects_heldout_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = self.build_corpus(root)
            training_hash = json.loads(manifest.read_text().splitlines()[0])[
                "sha256"
            ]
            heldout_manifest = root / "heldout.jsonl"
            heldout_manifest.write_text(
                json.dumps({"source_sha256": training_hash}) + "\n"
            )
            bundle.write_text(json.dumps({
                "manifest": str(heldout_manifest),
                "manifest_sha256": hashlib.sha256(
                    heldout_manifest.read_bytes()
                ).hexdigest(),
            }))
            with self.assertRaisesRegex(ValueError, "overlaps held-out"):
                validate(root, 2, 2, 1, [bundle])


if __name__ == "__main__":
    unittest.main()
