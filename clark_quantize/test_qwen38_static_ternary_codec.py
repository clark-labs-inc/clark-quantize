from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from clark_quantize import base3, qwen38_static_ternary_codec as static


class Qwen38StaticTernaryCodecTests(unittest.TestCase):
    def test_exhaustive_short_roundtrip(self) -> None:
        for length in range(7):
            for values in itertools.product((-1, 0, 1), repeat=length):
                source = np.asarray(values, dtype=np.int8)
                frame = static.pack(source)
                np.testing.assert_array_equal(static.unpack(frame), source)
                self.assertEqual(static.frame_metadata(frame)["header_bytes"], 23)

    def test_frequency_apportionment_is_deterministic_and_exact(self) -> None:
        source = np.asarray([-1] * 7 + [0] * 19 + [1] * 5, dtype=np.int8)
        first = static.quantized_frequencies(source)
        second = static.quantized_frequencies(source.copy())
        self.assertEqual(first, second)
        self.assertEqual(sum(first), static.FREQUENCY_TOTAL)
        self.assertTrue(all(value > 0 for value in first))
        self.assertEqual(static.quantized_frequencies(np.zeros(100, dtype=np.int8)), (0, 32768, 0))

    def test_qwen_like_source_beats_uniform_base3(self) -> None:
        generator = np.random.default_rng(20260817)
        source = generator.choice(
            np.asarray([-1, 0, 1], dtype=np.int8),
            size=200_000,
            p=(0.267, 0.466, 0.267),
        )
        static_frame = static.pack(source)
        uniform_frame = base3.pack_base3(source)
        np.testing.assert_array_equal(static.unpack(static_frame), source)
        static_bits = int(static.frame_metadata(static_frame)["payload_bits"])
        uniform_bits = int(base3.frame_metadata(uniform_frame)["payload_bits"])
        self.assertLess(static_bits, uniform_bits - int(0.04 * source.size))
        self.assertLess(static_bits / source.size, math.log2(3) - 0.04)

    def test_degenerate_source_roundtrips_in_constant_payload(self) -> None:
        source = np.zeros(100_000, dtype=np.int8)
        frame = static.pack(source)
        np.testing.assert_array_equal(static.unpack(frame), source)
        self.assertLessEqual(static.frame_metadata(frame)["payload_bits"], 2)

    def test_header_and_payload_corruption_are_rejected(self) -> None:
        source = np.asarray([-1, 0, 1] * 100, dtype=np.int8)
        for offset in (6, -1):
            frame = bytearray(static.pack(source))
            frame[offset] ^= 1
            with self.assertRaisesRegex(ValueError, "checksum|length"):
                static.unpack(bytes(frame))

    def test_rejects_zero_mass_for_observed_symbol(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero mass"):
            static.encode_payload(np.asarray([-1, 0], dtype=np.int8), (0, 16384, 16384))

    @unittest.skipUnless(static.native_decoder_available(), "static native decoder is not built")
    def test_native_and_python_decoders_match(self) -> None:
        generator = np.random.default_rng(311)
        for count in (0, 1, 31, 128, 1025, 65537):
            source = generator.choice(
                np.asarray([-1, 0, 1], dtype=np.int8), count, p=(0.21, 0.57, 0.22)
            )
            payload, bit_length, frequencies = static.encode_payload(source)
            native = static.decode_payload(payload, bit_length, count, frequencies)
            python = static._decode_payload_python(payload, bit_length, count, frequencies)
            np.testing.assert_array_equal(native, python)
            np.testing.assert_array_equal(native, source)

    @unittest.skipUnless(static.native_encoder_available(), "static native encoder is not built")
    def test_native_and_python_encoders_are_byte_exact(self) -> None:
        generator = np.random.default_rng(877)
        for count in (0, 1, 31, 128, 1025, 65537):
            source = generator.choice(
                np.asarray([-1, 0, 1], dtype=np.int8), count, p=(0.19, 0.59, 0.22)
            )
            frequencies = static.quantized_frequencies(source)
            native = static._encode_payload_native(source, frequencies)
            python = static._encode_payload_python(source, frequencies)
            self.assertEqual(native, python)


if __name__ == "__main__":
    unittest.main()
