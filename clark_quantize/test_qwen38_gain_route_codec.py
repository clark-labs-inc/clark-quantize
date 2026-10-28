from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from clark_quantize.qwen38_gain_hash_residual import gain_hash_storage_projection
from clark_quantize.qwen38_gain_route_codec import (
    deserialize_gain_hash_route_body,
    expand_gain_hash_route_compute,
    gain_route_payload_bytes,
    pack_gain_hash_route_body,
    serialize_gain_hash_route_body,
    unpack_gain_hash_route_body,
)


class Qwen38GainRouteCodecTests(unittest.TestCase):
    @staticmethod
    def random_route(
        generator: np.random.Generator,
        input_dim: int,
        sketch_dim: int,
        gain_bits: int,
    ):
        buckets = generator.integers(0, sketch_dim, size=input_dim, dtype=np.uint16)
        signs = generator.choice(np.array([-1, 1], dtype=np.int8), size=input_dim)
        gains = generator.integers(
            0, 1 << gain_bits, size=input_dim, dtype=np.uint8
        )
        codebook = np.sort(
            generator.uniform(0.0, 2.0, size=1 << gain_bits).astype(np.float16)
        )
        return buckets, signs, gains, codebook

    def assert_roundtrip(
        self, input_dim: int, sketch_dim: int, gain_bits: int, seed: int
    ) -> None:
        generator = np.random.default_rng(seed)
        source = self.random_route(generator, input_dim, sketch_dim, gain_bits)
        packed = pack_gain_hash_route_body(
            *source, sketch_dim=sketch_dim, gain_bits=gain_bits
        )
        body = serialize_gain_hash_route_body(packed)
        restored = deserialize_gain_hash_route_body(
            body,
            input_dim=input_dim,
            sketch_dim=sketch_dim,
            gain_bits=gain_bits,
        )
        decoded = unpack_gain_hash_route_body(restored)
        for observed, expected in zip(decoded, source[:3]):
            np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(restored.gain_codebook, source[3])
        self.assertFalse(restored.gain_codebook.flags.writeable)
        expanded = expand_gain_hash_route_compute(restored)
        self.assertEqual(
            sum(array.nbytes for array in expanded), restored.compute_layout_bytes
        )

    def test_exact_qwen_gate_and_down_routes_roundtrip(self) -> None:
        self.assert_roundtrip(5120, 40, 3, 38201)
        self.assert_roundtrip(17408, 136, 3, 38202)

    def test_qwen_route_body_bytes_match_complete_storage_ledger(self) -> None:
        for input_dim, rows, sketch_dim in (
            (5120, 17408, 40),
            (17408, 5120, 136),
        ):
            generator = np.random.default_rng(input_dim)
            source = self.random_route(generator, input_dim, sketch_dim, 3)
            packed = pack_gain_hash_route_body(
                *source, sketch_dim=sketch_dim, gain_bits=3
            )
            storage = gain_hash_storage_projection(
                input_dim=input_dim,
                output_rows=rows,
                sketch_dim=sketch_dim,
                gain_bits=3,
                nonzero_per_group=64,
            )
            self.assertEqual(
                packed.route_payload_bytes,
                storage["routing_and_gain_index_bytes"],
            )
            self.assertEqual(packed.codebook_bytes, storage["gain_codebook_bytes"])
            self.assertEqual(
                packed.body_bytes,
                storage["routing_and_gain_index_bytes"]
                + storage["gain_codebook_bytes"],
            )

    def test_known_lsb_first_bit_order(self) -> None:
        codebook = np.arange(8, dtype=np.float16)
        packed = pack_gain_hash_route_body(
            np.array([0, 4], dtype=np.uint16),
            np.array([-1, 1], dtype=np.int8),
            np.array([0, 7], dtype=np.uint8),
            codebook,
            sketch_dim=5,
            gain_bits=3,
        )
        # Seven bits per coordinate: 0000000 then 1111100, LSB first.
        self.assertEqual(packed.route_payload, bytes((0x00, 0x3E)))
        self.assertTrue(codebook.flags.writeable)
        self.assertFalse(packed.gain_codebook.flags.writeable)

    def test_nonzero_padding_and_bad_length_are_rejected(self) -> None:
        generator = np.random.default_rng(38203)
        source = self.random_route(generator, 3, 5, 3)
        packed = pack_gain_hash_route_body(*source, sketch_dim=5, gain_bits=3)
        tampered = bytearray(serialize_gain_hash_route_body(packed))
        tampered[packed.route_payload_bytes - 1] |= 0x80
        with self.assertRaisesRegex(ValueError, "padding"):
            deserialize_gain_hash_route_body(
                tampered, input_dim=3, sketch_dim=5, gain_bits=3
            )
        with self.assertRaisesRegex(ValueError, "length"):
            deserialize_gain_hash_route_body(
                serialize_gain_hash_route_body(packed) + b"\0",
                input_dim=3,
                sketch_dim=5,
                gain_bits=3,
            )

    def test_noncanonical_bucket_code_is_rejected(self) -> None:
        source = (
            np.array([0], dtype=np.uint16),
            np.array([1], dtype=np.int8),
            np.array([0], dtype=np.uint8),
            np.arange(8, dtype=np.float16),
        )
        packed = pack_gain_hash_route_body(*source, sketch_dim=5, gain_bits=3)
        # Bucket field occupies the low three bits; 7 is outside sketch_dim 5.
        tampered = replace(packed, route_payload=bytes((packed.route_payload[0] | 7,)))
        with self.assertRaisesRegex(ValueError, "bucket"):
            unpack_gain_hash_route_body(tampered)

    def test_randomized_all_gain_widths(self) -> None:
        generator = np.random.default_rng(38204)
        for case in range(100):
            input_dim = int(generator.integers(1, 258))
            sketch_dim = int(generator.integers(1, 258))
            gain_bits = int(generator.integers(1, 9))
            source = self.random_route(
                generator, input_dim, sketch_dim, gain_bits
            )
            packed = pack_gain_hash_route_body(
                *source, sketch_dim=sketch_dim, gain_bits=gain_bits
            )
            self.assertEqual(
                packed.route_payload_bytes,
                gain_route_payload_bytes(
                    input_dim, sketch_dim, gain_bits=gain_bits
                ),
                msg=f"case {case}",
            )
            decoded = unpack_gain_hash_route_body(packed)
            for observed, expected in zip(decoded, source[:3]):
                np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
