from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from clark_quantize.qwen38_scale_index_codec import (
    HEADER,
    decode_scale_indices_huffman,
    encode_scale_indices_huffman,
    scale_index_huffman_ledger,
    scale_index_huffman_ledger_from_frequencies,
)


class Qwen38ScaleIndexCodecTests(unittest.TestCase):
    def test_skewed_indices_round_trip_and_beat_fixed_width(self) -> None:
        values = np.concatenate(
            (
                np.full(9000, 7, dtype=np.uint8),
                np.tile(np.arange(16, dtype=np.uint8), 62),
                np.arange(8, dtype=np.uint8),
            )
        )
        encoded = encode_scale_indices_huffman(values)
        decoded = decode_scale_indices_huffman(encoded)
        np.testing.assert_array_equal(decoded, values)
        ledger = scale_index_huffman_ledger(values)
        self.assertEqual(len(encoded), ledger.section_bytes)
        self.assertLess(ledger.section_bytes, ledger.fixed_width_bytes)
        self.assertEqual(encoded, encode_scale_indices_huffman(values.copy()))

    def test_uniform_indices_use_four_bits_plus_header(self) -> None:
        values = np.tile(np.arange(16, dtype=np.uint8), 100)
        ledger = scale_index_huffman_ledger(values)
        self.assertEqual(set(ledger.code_lengths), {4})
        self.assertEqual(ledger.payload_bits, values.size * 4)
        self.assertEqual(ledger.section_bytes, HEADER.size + values.size // 2)
        np.testing.assert_array_equal(
            decode_scale_indices_huffman(encode_scale_indices_huffman(values)),
            values,
        )

    def test_single_symbol_uses_header_only(self) -> None:
        values = np.full(12345, 11, dtype=np.uint8)
        encoded = encode_scale_indices_huffman(values)
        ledger = scale_index_huffman_ledger(values)
        self.assertEqual(ledger.payload_bits, 0)
        self.assertEqual(len(encoded), HEADER.size)
        np.testing.assert_array_equal(decode_scale_indices_huffman(encoded), values)

    def test_frequency_only_ledger_matches_materialized_indices(self) -> None:
        values = np.asarray([0, 1, 1, 2, 2, 2, 15, 15], dtype=np.uint8)
        frequencies = np.bincount(values, minlength=16)
        self.assertEqual(
            scale_index_huffman_ledger_from_frequencies(frequencies),
            scale_index_huffman_ledger(values),
        )

    def test_rejects_out_of_alphabet_and_corrupt_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            encode_scale_indices_huffman(np.asarray([16], dtype=np.uint8))
        values = np.asarray([0, 1, 2, 3, 0], dtype=np.uint8)
        encoded = bytearray(encode_scale_indices_huffman(values))
        encoded[-1] |= 1
        with self.assertRaisesRegex(ValueError, "padding"):
            decode_scale_indices_huffman(encoded)

        encoded = bytearray(encode_scale_indices_huffman(values))
        magic, count, bits, mask, *lengths = HEADER.unpack_from(encoded)
        HEADER.pack_into(encoded, 0, magic, count, bits + 1, mask, *lengths)
        with self.assertRaisesRegex(ValueError, "payload|trailing|mid-symbol"):
            decode_scale_indices_huffman(encoded)

    def test_rejects_incomplete_codebook(self) -> None:
        values = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        encoded = bytearray(encode_scale_indices_huffman(values))
        magic, count, bits, mask, *lengths = HEADER.unpack_from(encoded)
        lengths[0] = 2
        lengths[1] = 2
        HEADER.pack_into(encoded, 0, magic, count, bits, mask, *lengths)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            decode_scale_indices_huffman(encoded)

    def test_qwen38_analyzer_authenticates_and_counts_packed_nibbles(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        runtime = repo / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
        name = "qwen38_scale_index_huffman_analyzer_test_module"
        spec = importlib.util.spec_from_file_location(
            name, runtime / "analyze_qwen38_27b_scale_index_huffman_v1.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        values = np.asarray([1, 2, 3, 0, 4], dtype=np.uint8)
        packed = bytes((0x21, 0x03, 0x04))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.bin"
            path.write_bytes(packed)
            frequencies = module.index_frequencies(
                path, values.size, hashlib.sha256(values.tobytes()).hexdigest()
            )
            np.testing.assert_array_equal(
                frequencies, np.bincount(values, minlength=16)
            )
            path.write_bytes(bytes((0x21, 0x03, 0xF4)))
            with self.assertRaisesRegex(RuntimeError, "padding"):
                module.index_frequencies(
                    path, values.size, hashlib.sha256(values.tobytes()).hexdigest()
                )

    def test_qwen38_huffman_preregistration_freezes_all_sources(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        path = (
            repo
            / "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_scale_index_huffman_preregistration_v1.json"
        )
        preregistration = json.loads(path.read_text())
        self.assertEqual(
            preregistration["scope"]["active_model_only"], "Qwen/Qwen3.8-27B"
        )
        self.assertEqual(preregistration["gate"]["required_tensors"], 666)
        self.assertEqual(
            preregistration["gate"]["required_scale_indices"], 217521520
        )
        for descriptor in preregistration["frozen_inputs"].values():
            source = repo / descriptor["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                descriptor["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
