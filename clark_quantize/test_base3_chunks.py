from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from clark_quantize.base3_chunks import ChunkedBase3Store, encode_chunk


class ChunkedBase3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "packed"
        self.identity = {"model_revision": "fixed", "quantizer": "optimal-128"}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_interrupted_prefix_resumes_exactly_and_finalizes(self) -> None:
        rng = np.random.default_rng(20260815)
        chunks = [rng.integers(-1, 2, 999 + index, dtype=np.int8) for index in range(4)]
        store = ChunkedBase3Store(self.root, self.identity)
        for index, codes in enumerate(chunks[:2]):
            self.assertEqual(store.commit_encoded(encode_chunk(index, codes, {"part": index})), "written")

        resumed = ChunkedBase3Store(self.root, self.identity)
        for index, codes in enumerate(chunks):
            if index < 2:
                self.assertTrue(resumed.reuse_if_committed(index, codes, {"part": index}))
            else:
                self.assertFalse(resumed.reuse_if_committed(index, codes, {"part": index}))
                self.assertEqual(resumed.commit_encoded(encode_chunk(index, codes, {"part": index})), "written")
        source_sha = hashlib.sha256(b"".join(chunk.tobytes() for chunk in chunks)).hexdigest()
        manifest = resumed.finalize(4, sum(map(len, chunks)), source_sha)
        self.assertTrue(manifest["complete"])
        decoded = list(ChunkedBase3Store(self.root, self.identity).decoded_chunks())
        np.testing.assert_array_equal(np.concatenate(decoded), np.concatenate(chunks))

    def test_changed_source_or_identity_is_rejected(self) -> None:
        codes = np.asarray([-1, 0, 1] * 100, dtype=np.int8)
        store = ChunkedBase3Store(self.root, self.identity)
        store.commit_encoded(encode_chunk(0, codes, {"part": 0}))
        changed = codes.copy()
        changed[0] = 1
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            ChunkedBase3Store(self.root, self.identity).commit_encoded(encode_chunk(0, changed, {"part": 0}))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            ChunkedBase3Store(self.root, {"model_revision": "other"})

    def test_corrupt_committed_chunk_is_rejected(self) -> None:
        codes = np.asarray([-1, 0, 1] * 100, dtype=np.int8)
        store = ChunkedBase3Store(self.root, self.identity)
        store.commit_encoded(encode_chunk(0, codes))
        path = store.chunk_path(0)
        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        with self.assertRaisesRegex(ValueError, "frame SHA-256"):
            ChunkedBase3Store(self.root, self.identity).validate_prefix()

    def test_manifest_replace_leaves_parseable_json(self) -> None:
        store = ChunkedBase3Store(self.root, self.identity)
        store.commit_encoded(encode_chunk(0, np.asarray([0], dtype=np.int8)))
        parsed = json.loads(store.partial_path.read_text())
        self.assertEqual(parsed["chunks"][0]["index"], 0)

    def test_unverified_encoder_result_is_rejected(self) -> None:
        encoded = encode_chunk(0, np.asarray([-1, 0, 1], dtype=np.int8))
        encoded["verified_roundtrip"] = False
        store = ChunkedBase3Store(self.root, self.identity)
        with self.assertRaisesRegex(ValueError, "not round-trip verified"):
            store.commit_encoded(encoded)


if __name__ == "__main__":
    unittest.main()
