from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from clark_quantize.base3 import frame_metadata, pack_base3
from clark_quantize.base3_block203 import pack_rows, unpack_rows
from clark_quantize.qwen38_cq3_export import (
    atomic_code_spool,
    exact_tensor_bytes,
    pack_verified_rows,
    transcode_mapped_record,
)


class Qwen38CQ3ExportTests(unittest.TestCase):
    def test_byte_aligned_batches_equal_one_canonical_matrix_stream(self) -> None:
        codes = (np.arange(9 * 129, dtype=np.int32) % 3 - 1).astype(np.int8).reshape(9, 129)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tensor.codes"
            receipt = atomic_code_spool(
                path,
                (pack_verified_rows(codes[:8]), pack_verified_rows(codes[8:])),
                expected_rows=9,
                expected_columns=129,
            )
            self.assertEqual(path.read_bytes(), pack_rows(codes))
            np.testing.assert_array_equal(unpack_rows(path.read_bytes(), 9, 129), codes)
            self.assertEqual(receipt["trits"], codes.size)
            self.assertEqual(receipt["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_nonfinal_unaligned_batch_fails_without_publishing(self) -> None:
        codes = np.zeros((2, 129), dtype=np.int8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.codes"
            with self.assertRaisesRegex(ValueError, "not byte-aligned"):
                atomic_code_spool(
                    path,
                    (pack_verified_rows(codes[:1]), pack_verified_rows(codes[1:])),
                    expected_rows=2,
                    expected_columns=129,
                )
            self.assertFalse(path.exists())

    def test_historical_frame_transcodes_with_authentication(self) -> None:
        codes = (np.arange(3 * 137, dtype=np.int32) % 3 - 1).astype(np.int8)
        frame = pack_base3(codes)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks = root / "codes.cq3d" / "chunks"
            chunks.mkdir(parents=True)
            (chunks / "00000000.cq3a").write_bytes(frame)
            record = {
                "index": 0,
                "descriptor": {
                    "row_start": 0,
                    "row_stop": 3,
                    "tensor_shape": [3, 137],
                },
                "frame_sha256": hashlib.sha256(frame).hexdigest(),
                "raw_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
                "frame_metadata": frame_metadata(frame),
            }
            packed = transcode_mapped_record(root, record)
            np.testing.assert_array_equal(
                unpack_rows(packed.payload, 3, 137).reshape(-1), codes
            )
            record["raw_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "raw-code"):
                transcode_mapped_record(root, record)

    def test_exact_retained_bytes_preserve_bfloat16_bits(self) -> None:
        tensor = torch.asarray([1.0, -2.0, 0.125], dtype=torch.bfloat16)
        self.assertEqual(exact_tensor_bytes(tensor), tensor.view(torch.uint8).numpy().tobytes())


if __name__ == "__main__":
    unittest.main()
