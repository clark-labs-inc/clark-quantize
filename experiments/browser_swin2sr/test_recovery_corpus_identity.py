import tempfile
import unittest
from pathlib import Path

from recovery_corpus_identity import recovery_corpus_identity


class RecoveryCorpusIdentityTests(unittest.TestCase):
    def corpus(self, root: Path, record: str = '{"sha256":"a"}\n') -> None:
        root.mkdir(parents=True)
        (root / "summary.json").write_text('{"images":1}\n')
        (root / "manifest.jsonl").write_text(record)

    def test_identity_changes_with_manifest_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            self.corpus(root)
            first = recovery_corpus_identity([root])
            (root / "manifest.jsonl").write_text('{"sha256":"b"}\n')
            second = recovery_corpus_identity([root])

            self.assertNotEqual(first, second)

    def test_identity_is_content_based_when_root_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first_root = parent / "first"
            second_root = parent / "second"
            self.corpus(first_root)
            self.corpus(second_root)

            self.assertEqual(
                recovery_corpus_identity([first_root]),
                recovery_corpus_identity([second_root]),
            )


if __name__ == "__main__":
    unittest.main()
