import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from recovery_data import BalancedSourceSampler, SAMPLER_FORMAT


class RecoveryDataTests(unittest.TestCase):
    def corpus(self, root: Path):
        rows = []
        for source, languages, count in (
            ("rico", (None,), 7),
            ("aria", (None,), 3),
            ("websight", (None,), 9),
            ("pd12m", (None,), 2),
            ("flores", ("aa", "bb"), 8),
        ):
            for index in range(count):
                language = languages[index % len(languages)]
                relative = f"{source}/{language or 'none'}-{index}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"png")
                rows.append({
                    "local_path": relative,
                    "source": source,
                    "language": language,
                })
        (root / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (root / "summary.json").write_text(
            json.dumps({"images": len(rows), "source": "fixture"}) + "\n"
        )

    def test_epoch_is_source_and_language_balanced(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.corpus(root)
            sampler = BalancedSourceSampler.from_roots([root], 17)
            paths = sampler.paths(0)

        sources = Counter(path.parent.name for path in paths)
        languages = Counter(
            path.name.split("-", 1)[0]
            for path in paths
            if path.parent.name == "flores"
        )
        self.assertLessEqual(max(sources.values()) - min(sources.values()), 1)
        self.assertLessEqual(max(languages.values()) - min(languages.values()), 1)
        self.assertEqual(len(paths), 29)

    def test_product_sources_prioritize_real_ui(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = []
            for source in (
                "rico",
                "aria_ui",
                "websight",
                "pd12m",
                "flores200_balanced_synthetic_ui",
            ):
                relative = f"{source}/sample.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"png")
                rows.append({
                    "local_path": relative,
                    "source": source,
                    "language": (
                        "eng_Latn"
                        if source == "flores200_balanced_synthetic_ui"
                        else None
                    ),
                })
            (root / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (root / "summary.json").write_text(
                json.dumps({"images": len(rows), "source": "fixture"}) + "\n"
            )
            sampler = BalancedSourceSampler.from_roots([root], 17)

        self.assertEqual(sampler.source_quotas["rico"], 2)
        self.assertEqual(sampler.source_quotas["pd12m"], 0)
        self.assertAlmostEqual(
            sampler.metadata["source_weights"]["rico"],
            0.35,
        )

    def test_epoch_is_reproducible_and_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.corpus(root)
            sampler = BalancedSourceSampler.from_roots([root], 17)
            first = sampler.paths(0)
            self.assertEqual(first, sampler.paths(0))
            self.assertNotEqual(first, sampler.paths(1))

    def test_base_checkpoint_intentionally_starts_new_sampler(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.corpus(root)
            sampler = BalancedSourceSampler.from_roots([root], 17)
            self.assertEqual(
                sampler.restore({}, random.Random(17), 100),
                (0, 0, False),
            )
            self.assertEqual(sampler.metadata["format"], SAMPLER_FORMAT)

    def test_recovery_resume_fails_closed_on_sampler_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.corpus(root)
            sampler = BalancedSourceSampler.from_roots([root], 17)
            checkpoint = {
                "recovery": {
                    "sampler": {
                        **sampler.metadata,
                        "epoch_images": sampler.total + 1,
                    },
                },
            }
            with self.assertRaisesRegex(ValueError, "changed without"):
                sampler.restore(checkpoint, random.Random(17), 100)

    def test_changed_corpus_requires_reset_even_when_shape_is_identical(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.corpus(root)
            original = BalancedSourceSampler.from_roots([root], 17)
            checkpoint = {"recovery": {"sampler": original.metadata}}
            manifest = root / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rows[0]["revision"] = "glyph-safe"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            changed = BalancedSourceSampler.from_roots([root], 17)

            with self.assertRaisesRegex(ValueError, "changed without"):
                changed.restore(checkpoint, random.Random(17), 100)
            self.assertEqual(
                changed.restore(
                    checkpoint,
                    random.Random(17),
                    100,
                    reset_data_stream=True,
                ),
                (0, 0, False),
            )


if __name__ == "__main__":
    unittest.main()
