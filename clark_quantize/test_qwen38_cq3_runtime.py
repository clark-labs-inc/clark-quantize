from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import numpy as np

from clark_quantize.base3_block203 import (
    matrix_payload_bits,
    pack_rows,
)
from clark_quantize.qwen38_cq3_container import (
    CQ3ContainerError,
    Qwen38CQ3Reader,
    Qwen38CQ3Writer,
)
from clark_quantize.qwen38_cq3_runtime import (
    CQ3RuntimeError,
    Qwen38CQ3DeltaNet,
    Qwen38CQ3MLP,
    Qwen38CQ3RuntimeImage,
)
from clark_quantize.scale_codebook import pack_nibbles
from clark_quantize.ternary_cpu import (
    causal_conv4_silu_f32,
    gated_delta_parameters_f32,
    native_available,
    pack_group128_2bit,
    quantize_q8,
    recurrent_gated_delta_f32,
    rmsnorm_silu_gate_f32,
    swiglu_f32,
)


class FakeProjection:
    def __init__(self, shape: tuple[int, int], output: np.ndarray) -> None:
        self.shape = shape
        self.rows, self.columns = shape
        self.output = np.ascontiguousarray(output, dtype=np.float32)
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def gemv_q8(
        self,
        activation_q8: np.ndarray,
        activation_scales: np.ndarray,
        activation_sums: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        del kwargs
        self.calls.append(
            (
                activation_q8.copy(),
                activation_scales.copy(),
                activation_sums.copy(),
            )
        )
        return self.output.copy()


class Qwen38CQ3RuntimeTests(unittest.TestCase):
    rows = 8
    columns = 1024

    def fixture(self, root: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
        path = root / "qwen38-row-normalized.cq3"
        codes = (
            np.arange(self.rows * self.columns, dtype=np.int64).reshape(
                self.rows, self.columns
            )
            % 3
            - 1
        ).astype(np.int8)
        groups = self.columns // 128
        indices = (
            np.arange(self.rows * groups, dtype=np.uint8) % 16
        ).astype(np.uint8)
        anchors = np.linspace(0.125, 1.0, self.rows, dtype=np.float16)
        codebook = np.linspace(0.0, 1.0, 16, dtype=np.float16)
        packed_indices = pack_nibbles(indices)
        exact_raw = bytes(range(8))
        with Qwen38CQ3Writer(
            path,
            model={"id": "Qwen/Qwen3.8-27B", "revision": "runtime-test"},
            strategy={
                "name": "base3_group128_row_normalized_codebook4",
                "codec": "block203",
                "group_size": 128,
                "scale_bits": 4,
                "row_anchor_dtype": "F16",
                "scale_codebook_entries": 16,
                "scale_codebook_dtype": "F16",
            },
            parameter_count=codes.size + 4,
        ) as writer:
            code = writer.add_section(
                "tensor.0000.codes",
                "ternary_block203",
                pack_rows(codes),
                metadata={
                    "rows": self.rows,
                    "columns": self.columns,
                    "trits": codes.size,
                    "payload_bits": matrix_payload_bits(
                        self.rows, self.columns
                    ),
                    "group_trits": 128,
                    "full_group_bits": 203,
                },
            )
            scales = writer.add_section_parts(
                "tensor.0000.scales",
                "scale_indices_u4_anchors_f16_and_codebook_f16",
                (
                    packed_indices,
                    anchors.astype("<f2").tobytes(),
                    codebook.astype("<f2").tobytes(),
                ),
                metadata={
                    "scale_count": indices.size,
                    "index_bytes": len(packed_indices),
                    "codebook_bytes": codebook.nbytes,
                    "codebook_entries": 16,
                    "index_order": "low_nibble_first",
                    "row_count": self.rows,
                    "anchor_bytes": anchors.nbytes,
                    "factor_order": "indices_then_row_anchors_then_codebook",
                    "decode": "float32(fp16_anchor) * float32(fp16_ratio)",
                },
            )
            exact = writer.add_section(
                "tensor.0001.retained",
                "retained_exact_BF16",
                exact_raw,
                metadata={"dtype": "BF16"},
            )
            auxiliary = writer.add_section(
                "aux.0000",
                "auxiliary_file",
                b"{}\n",
                metadata={"name": "config.json"},
            )
            writer.finalize(
                tensors=[
                    {
                        "name": "model.layers.0.mlp.down_proj.weight",
                        "shape": [self.rows, self.columns],
                        "parameters": codes.size,
                        "source_dtype": "BF16",
                        "encoding": (
                            "base3_group128_row_normalized_codebook4"
                        ),
                        "sections": [code["id"], scales["id"]],
                    },
                    {
                        "name": "model.layers.0.input_layernorm.weight",
                        "shape": [4],
                        "parameters": 4,
                        "source_dtype": "BF16",
                        "encoding": "retained_exact_BF16",
                        "sections": [exact["id"]],
                    },
                ],
                auxiliary_files=[
                    {
                        "name": "config.json",
                        "section": auxiliary["id"],
                        "bytes": auxiliary["bytes"],
                        "sha256": auxiliary["sha256"],
                    }
                ],
            )
        return path, codes, indices, anchors

    def mlp_fixture(self, root: Path) -> Path:
        path = root / "qwen38-mlp.cq3"
        rows = columns = 128
        codebook = np.linspace(0.0, 1.0, 16, dtype=np.float16)
        with Qwen38CQ3Writer(
            path,
            model={"id": "Qwen/Qwen3.8-27B", "revision": "mlp-test"},
            strategy={
                "name": "base3_group128_row_normalized_codebook4",
                "codec": "block203",
                "group_size": 128,
                "scale_bits": 4,
                "row_anchor_dtype": "F16",
                "scale_codebook_entries": 16,
                "scale_codebook_dtype": "F16",
            },
            parameter_count=3 * rows * columns,
        ) as writer:
            tensors = []
            for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
                codes = (
                    (np.arange(rows * columns, dtype=np.int64) + index) % 3
                    - 1
                ).astype(np.int8).reshape(rows, columns)
                indices = (
                    (np.arange(rows, dtype=np.uint8) + index) % 16
                ).astype(np.uint8)
                anchors = np.linspace(
                    0.0625 + index * 0.01,
                    0.5 + index * 0.01,
                    rows,
                    dtype=np.float16,
                )
                packed_indices = pack_nibbles(indices)
                prefix = f"tensor.{index:04d}"
                code = writer.add_section(
                    prefix + ".codes",
                    "ternary_block203",
                    pack_rows(codes),
                    metadata={
                        "rows": rows,
                        "columns": columns,
                        "trits": rows * columns,
                        "payload_bits": matrix_payload_bits(rows, columns),
                        "group_trits": 128,
                        "full_group_bits": 203,
                    },
                )
                scales = writer.add_section_parts(
                    prefix + ".scales",
                    "scale_indices_u4_anchors_f16_and_codebook_f16",
                    (
                        packed_indices,
                        anchors.astype("<f2").tobytes(),
                        codebook.astype("<f2").tobytes(),
                    ),
                    metadata={
                        "scale_count": rows,
                        "index_bytes": len(packed_indices),
                        "codebook_bytes": codebook.nbytes,
                        "codebook_entries": 16,
                        "index_order": "low_nibble_first",
                        "row_count": rows,
                        "anchor_bytes": anchors.nbytes,
                        "factor_order": "indices_then_row_anchors_then_codebook",
                        "decode": "float32(fp16_anchor) * float32(fp16_ratio)",
                    },
                )
                tensors.append(
                    {
                        "name": (
                            "model.language_model.layers.0.mlp."
                            f"{projection}.weight"
                        ),
                        "shape": [rows, columns],
                        "parameters": rows * columns,
                        "source_dtype": "BF16",
                        "encoding": (
                            "base3_group128_row_normalized_codebook4"
                        ),
                        "sections": [code["id"], scales["id"]],
                    }
                )
            writer.finalize(tensors=tensors)
        return path

    def test_zero_copy_views_are_authenticated_and_survive_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, codes, indices, anchors = self.fixture(Path(temporary))
            image = Qwen38CQ3RuntimeImage(path)
            view = image.ternary_tensor(
                "model.layers.0.mlp.down_proj.weight"
            )
            self.assertEqual(view.shape, (self.rows, self.columns))
            self.assertEqual(view.packed_weights.nbytes, len(pack_rows(codes)))
            self.assertEqual(view.packed_scale_indices.nbytes, indices.size // 2)
            np.testing.assert_array_equal(view.row_anchors, anchors)
            self.assertFalse(view.packed_weights.flags.writeable)
            self.assertTrue(np.shares_memory(view.packed_weights, image._mapping))
            exact = image.exact_tensor(
                "model.layers.0.input_layernorm.weight"
            )
            self.assertEqual(exact.source_dtype, "BF16")
            self.assertEqual(exact.raw.tobytes(), bytes(range(8)))
            expected_exact = (
                np.frombuffer(bytes(range(8)), dtype="<u2").astype(np.uint32)
                << np.uint32(16)
            ).view(np.float32)
            np.testing.assert_array_equal(exact.as_f32(), expected_exact)
            self.assertEqual(image.auxiliary_file("config.json").tobytes(), b"{}\n")
            receipt = image.verify_all()
            self.assertEqual(receipt["sections_verified"], 4)
            image.close()
            self.assertEqual(view.packed_weights[0], pack_rows(codes)[0])
            with self.assertRaisesRegex(RuntimeError, "closed"):
                image.ternary_tensor(
                    "model.layers.0.mlp.down_proj.weight"
                )

    @unittest.skipUnless(native_available(), "native ternary CPU library is absent")
    def test_direct_mmap_gemv_matches_reference_and_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, codes, indices, anchors = self.fixture(Path(temporary))
            with Qwen38CQ3RuntimeImage(path) as image:
                view = image.ternary_tensor(
                    "model.layers.0.mlp.down_proj.weight"
                )
                activation = np.linspace(
                    -2.0, 3.0, self.columns, dtype=np.float32
                )
                activation_q8, activation_scales, activation_sums = quantize_q8(
                    activation
                )
                one_thread = view.gemv_q8(
                    activation_q8,
                    activation_scales,
                    activation_sums,
                    implementation="scalar_chunk16",
                )
                aligned = view.derive_aligned2()
                np.testing.assert_array_equal(
                    aligned.packed_weights.reshape(-1),
                    pack_group128_2bit(codes),
                )
                aligned_one_thread = aligned.gemv_q8(
                    activation_q8,
                    activation_scales,
                    activation_sums,
                    implementation="scalar",
                )
                with ThreadPoolExecutor(max_workers=4) as executor:
                    four_threads = view.gemv_q8(
                        activation_q8,
                        activation_scales,
                        activation_sums,
                        implementation="scalar_chunk16",
                        threads=4,
                        executor=executor,
                    )
                    aligned_four_threads = aligned.gemv_q8(
                        activation_q8,
                        activation_scales,
                        activation_sums,
                        implementation="scalar",
                        threads=4,
                        executor=executor,
                    )
                groups = self.columns // 128
                dots = np.sum(
                    codes.reshape(self.rows, groups, 128).astype(np.int32)
                    * activation_q8.reshape(groups, 128).astype(np.int32),
                    axis=2,
                )
                ratios = view.ratio_codebook[indices].reshape(self.rows, groups)
                expected = np.sum(
                    dots.astype(np.float32)
                    * anchors.astype(np.float32)[:, None]
                    * ratios.astype(np.float32)
                    * activation_scales[None, :],
                    axis=1,
                    dtype=np.float32,
                )
                np.testing.assert_allclose(one_thread, expected, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(four_threads, one_thread)
                np.testing.assert_array_equal(aligned_one_thread, one_thread)
                np.testing.assert_array_equal(
                    aligned_four_threads, aligned_one_thread
                )

    def test_payload_corruption_and_metadata_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _, _ = self.fixture(Path(temporary))
            with Qwen38CQ3Reader(path) as reader:
                offset = reader.sections["tensor.0000.codes"]["offset"]
            with path.open("r+b") as handle:
                handle.seek(offset)
                value = handle.read(1)
                handle.seek(offset)
                handle.write(bytes([value[0] ^ 1]))
            with Qwen38CQ3RuntimeImage(path) as image:
                with self.assertRaisesRegex(CQ3ContainerError, "SHA-256"):
                    image.ternary_tensor(
                        "model.layers.0.mlp.down_proj.weight"
                    )

    def test_nonternary_and_unknown_lookups_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _, _ = self.fixture(Path(temporary))
            with Qwen38CQ3RuntimeImage(path) as image:
                with self.assertRaisesRegex(CQ3RuntimeError, "not a row-normalized"):
                    image.ternary_tensor(
                        "model.layers.0.input_layernorm.weight"
                    )
                with self.assertRaisesRegex(KeyError, "unknown Qwen3.8"):
                    image.exact_tensor("missing.weight")

    @unittest.skipUnless(native_available(), "native ternary CPU library is absent")
    def test_mlp_path_reuses_input_quantization_and_matches_manual_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.mlp_fixture(Path(temporary))
            with Qwen38CQ3RuntimeImage(path) as image:
                mlp = Qwen38CQ3MLP.from_image(image, 0)
                activation = np.linspace(-1.5, 2.0, 128, dtype=np.float32)
                actual = mlp.forward(
                    activation, implementation="scalar_chunk16"
                )
                q8, scales, sums = quantize_q8(activation)
                gate = mlp.gate.gemv_q8(
                    q8,
                    scales,
                    sums,
                    implementation="scalar_chunk16",
                )
                up = mlp.up.gemv_q8(
                    q8,
                    scales,
                    sums,
                    implementation="scalar_chunk16",
                )
                intermediate = swiglu_f32(gate, up)
                hidden_q8, hidden_scales, hidden_sums = quantize_q8(intermediate)
                expected = mlp.down.gemv_q8(
                    hidden_q8,
                    hidden_scales,
                    hidden_sums,
                    implementation="scalar_chunk16",
                )
                np.testing.assert_array_equal(actual, expected)
                with self.assertRaisesRegex(ValueError, "hidden dimension"):
                    mlp.forward(np.zeros(127, dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "layer index"):
                    Qwen38CQ3MLP.from_image(image, True)

    @unittest.skipUnless(native_available(), "native ternary CPU library is absent")
    def test_exact_shape_deltanet_graph_matches_manual_primitives(self) -> None:
        model = Qwen38CQ3DeltaNet
        index = np.arange(model.CONV_DIM, dtype=np.float32)
        mixed_qkv = np.sin(index * np.float32(0.003)) * np.float32(0.25)
        z_values = np.cos(
            np.arange(model.VALUE_DIM, dtype=np.float32) * np.float32(0.005)
        ) * np.float32(0.2)
        projected_a = np.linspace(-1.0, 1.0, model.VALUE_HEADS, dtype=np.float32)
        projected_b = np.linspace(1.0, -1.0, model.VALUE_HEADS, dtype=np.float32)
        output_value = np.linspace(
            -0.5, 0.5, model.HIDDEN_SIZE, dtype=np.float32
        )
        qkv = FakeProjection((model.CONV_DIM, model.HIDDEN_SIZE), mixed_qkv)
        z = FakeProjection((model.VALUE_DIM, model.HIDDEN_SIZE), z_values)
        a = FakeProjection((model.VALUE_HEADS, model.HIDDEN_SIZE), projected_a)
        b = FakeProjection((model.VALUE_HEADS, model.HIDDEN_SIZE), projected_b)
        out = FakeProjection((model.HIDDEN_SIZE, model.VALUE_DIM), output_value)
        conv_weights = np.zeros((4, model.CONV_DIM), dtype=np.float32)
        conv_weights[3] = np.linspace(
            0.05, 0.25, model.CONV_DIM, dtype=np.float32
        )
        a_log = np.linspace(-3.0, 0.0, model.VALUE_HEADS, dtype=np.float32)
        dt_bias = np.linspace(-0.5, 0.5, model.VALUE_HEADS, dtype=np.float32)
        norm_weights = np.linspace(0.75, 1.25, model.HEAD_DIM, dtype=np.float32)
        layer = model(
            qkv=qkv,
            z=z,
            a=a,
            b=b,
            out=out,
            conv_weights=conv_weights,
            a_log=a_log,
            dt_bias=dt_bias,
            norm_weights=norm_weights,
        )
        activation = np.linspace(
            -1.0, 1.0, model.HIDDEN_SIZE, dtype=np.float32
        )
        initial_conv = np.zeros((3, model.CONV_DIM), dtype=np.float32)
        initial_recurrent = np.zeros(
            (model.VALUE_HEADS, model.HEAD_DIM, model.HEAD_DIM),
            dtype=np.float32,
        )
        actual_conv = initial_conv.copy()
        actual_recurrent = initial_recurrent.copy()
        actual = layer.forward(
            activation,
            actual_conv,
            actual_recurrent,
            gemv_implementation="scalar_chunk16",
            elementwise_implementation="scalar",
            validated_state=True,
        )
        np.testing.assert_array_equal(actual, output_value)

        expected_input_q8 = quantize_q8(activation)
        for projection in (qkv, z, a, b):
            self.assertEqual(len(projection.calls), 1)
            for observed, expected in zip(projection.calls[0], expected_input_q8):
                np.testing.assert_array_equal(observed, expected)

        expected_conv = initial_conv.copy()
        convolved = causal_conv4_silu_f32(
            mixed_qkv,
            expected_conv,
            conv_weights,
            implementation="scalar",
        )
        query = convolved[: model.KEY_DIM].reshape(
            model.KEY_HEADS, model.HEAD_DIM
        )
        key = convolved[model.KEY_DIM : 2 * model.KEY_DIM].reshape(
            model.KEY_HEADS, model.HEAD_DIM
        )
        value = convolved[2 * model.KEY_DIM :].reshape(
            model.VALUE_HEADS, model.HEAD_DIM
        )
        log_decay, beta = gated_delta_parameters_f32(
            projected_a, projected_b, a_log, dt_bias
        )
        expected_recurrent = initial_recurrent.copy()
        recurrent = recurrent_gated_delta_f32(
            query,
            key,
            value,
            log_decay,
            beta,
            expected_recurrent,
            implementation="scalar",
            validated=True,
        )
        gated = rmsnorm_silu_gate_f32(
            recurrent,
            norm_weights,
            z_values.reshape(model.VALUE_HEADS, model.HEAD_DIM),
            implementation="scalar",
        ).reshape(model.VALUE_DIM)
        expected_output_q8 = quantize_q8(gated)
        self.assertEqual(len(out.calls), 1)
        for observed, expected in zip(out.calls[0], expected_output_q8):
            np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(actual_conv, expected_conv)
        np.testing.assert_array_equal(actual_recurrent, expected_recurrent)


if __name__ == "__main__":
    unittest.main()
