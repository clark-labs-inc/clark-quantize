from __future__ import annotations

import unittest

import numpy as np

from clark_quantize.base3_radix5 import HEADER, pack_radix5, radix5_metadata, unpack_radix5


class Radix5Tests(unittest.TestCase):
    def test_exact_roundtrip_across_partial_groups(self) -> None:
        generator = np.random.default_rng(20260815)
        for count in range(0, 33):
            codes = generator.integers(-1, 2, size=count, dtype=np.int8)
            frame = pack_radix5(codes)
            np.testing.assert_array_equal(unpack_radix5(frame), codes)
            metadata = radix5_metadata(frame)
            self.assertEqual(metadata["payload_bytes"], (count + 4) // 5)

    def test_rejects_corruption_invalid_codes_and_noncanonical_padding(self) -> None:
        with self.assertRaisesRegex(ValueError, "ternary"):
            pack_radix5(np.asarray([-2, 0, 1], dtype=np.int8))
        frame = bytearray(pack_radix5(np.asarray([-1, 0, 1, -1, 0], dtype=np.int8)))
        frame[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            unpack_radix5(bytes(frame))

        frame = bytearray(pack_radix5(np.asarray([-1, 0, 1, -1, 0], dtype=np.int8)))
        frame[12] ^= 1  # Change count from five to four without changing payload length.
        with self.assertRaisesRegex(ValueError, "checksum"):
            unpack_radix5(bytes(frame))

        # Rebuild a valid CRC frame whose final unused digit is nonzero.
        import struct
        import zlib
        payload = bytes([81])  # digits [0, 0, 0, 0, 1], but count says one trit.
        count = 1
        checksum = zlib.crc32(struct.pack(">Q", count) + payload) & 0xFFFFFFFF
        bad_padding = struct.pack(">5sQI", b"CQ3R5", count, checksum) + payload
        with self.assertRaisesRegex(ValueError, "padding"):
            unpack_radix5(bad_padding)

    def test_rejects_noncanonical_payload_byte(self) -> None:
        import struct
        import zlib
        payload = bytes([243])
        count = 5
        checksum = zlib.crc32(struct.pack(">Q", count) + payload) & 0xFFFFFFFF
        frame = struct.pack(">5sQI", b"CQ3R5", count, checksum) + payload
        self.assertEqual(len(frame), HEADER.size + 1)
        with self.assertRaisesRegex(ValueError, "non-canonical byte"):
            unpack_radix5(frame)


if __name__ == "__main__":
    unittest.main()
