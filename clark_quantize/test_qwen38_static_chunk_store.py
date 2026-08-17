from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from clark_quantize.qwen38_static_chunk_store import StaticChunkStore, sha256_bytes
from clark_quantize.qwen38_static_ternary_codec import frame_metadata, pack


def encoded(index: int, tensor: str, values: list[int]) -> dict:
    codes = np.asarray(values, dtype=np.int8)
    frame = pack(codes)
    metadata = frame_metadata(frame)
    return {
        "index": index,
        "tensor": tensor,
        "trits": codes.size,
        "counts": [int((codes == value).sum()) for value in (-1, 0, 1)],
        "raw_sha256": sha256_bytes(codes.tobytes()),
        "source_frame_bytes": 100 + index,
        "source_frame_sha256": f"{index + 1:064x}",
        "static_frame_sha256": sha256_bytes(frame),
        "static_payload_bits": metadata["payload_bits"],
        "frequencies": metadata["frequencies"],
        "encode_seconds": 0.01,
        "decode_seconds": 0.02,
        "frame": frame,
    }


class Qwen38StaticChunkStoreTests(unittest.TestCase):
    def test_commit_resume_validate_and_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"model": "Qwen/Qwen3.8-27B", "source": "abc"}
            store = StaticChunkStore(root, identity)
            first = encoded(0, "a", [-1, 0, 1, 0])
            second = encoded(1, "b", [0, 0, 1])
            self.assertEqual(store.commit(first), "written")
            self.assertEqual(store.commit(second), "written")
            resumed = StaticChunkStore(root, identity)
            self.assertEqual(resumed.next_index, 2)
            self.assertEqual(resumed.validate_prefix()["trits"], 7)
            self.assertEqual(resumed.commit(first), "reused")
            final = resumed.finalize(expected_chunks=2, expected_trits=7)
            self.assertTrue(final["complete"])
            self.assertEqual(json.loads((root / "manifest.json").read_text()), final)

    def test_rejects_identity_order_and_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StaticChunkStore(root, {"source": "one"})
            with self.assertRaisesRegex(ValueError, "out of order"):
                store.commit(encoded(1, "a", [0]))
            store.commit(encoded(0, "a", [-1, 1]))
            with self.assertRaisesRegex(ValueError, "identity"):
                StaticChunkStore(root, {"source": "two"})
            frame = store.chunk_path(0)
            content = bytearray(frame.read_bytes())
            content[-1] ^= 1
            frame.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "digest"):
                store.validate_prefix()


if __name__ == "__main__":
    unittest.main()
