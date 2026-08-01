import sys
import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_portable_onnx_artifact as portable
from build_portable_onnx_artifact import (
    inject_hadamard_inputs,
    map_state_to_onnx,
    portable_precision_contract,
    write_deterministic_gzip,
)


class BuildPortableOnnxArtifactTests(unittest.TestCase):
    def test_runtime_package_is_deterministic_and_lossless(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.onnx"
            first = root / "first.onnx.gz"
            second = root / "second.onnx.gz"
            payload = bytes(range(256)) * 100
            source.write_bytes(payload)

            write_deterministic_gzip(source, first)
            write_deterministic_gzip(source, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(gzip.decompress(first.read_bytes()), payload)
            self.assertLess(first.stat().st_size, source.stat().st_size)

    def test_maps_direct_and_transposed_weights_by_exact_teacher_bytes(self):
        direct = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
        matrix = np.asarray(
            [[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6]], dtype=np.float32
        )
        initializers = [
            numpy_helper.from_array(direct, "direct"),
            numpy_helper.from_array(matrix.T, "transposed"),
        ]
        state = {
            "bias": torch.from_numpy(direct),
            "weight": torch.from_numpy(matrix),
        }

        mapping = map_state_to_onnx(state, initializers)

        self.assertEqual(mapping["bias"], ("direct", "direct"))
        self.assertEqual(mapping["weight"], ("transposed", "transpose"))

    def test_rejects_unmapped_initializer(self):
        state = {"bias": torch.tensor([1.0], dtype=torch.float32)}
        initializers = [
            numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), "bias"),
            numpy_helper.from_array(np.asarray([2.0], dtype=np.float32), "extra"),
        ]

        with self.assertRaisesRegex(ValueError, "unmapped initializers"):
            map_state_to_onnx(state, initializers)

    def test_injects_shared_block_hadamard_before_selected_matmul(self):
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["input", "weight"], ["output"])],
            "hadamard",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
            [
                numpy_helper.from_array(
                    np.asarray(
                        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
                        dtype=np.float32,
                    ),
                    "weight",
                )
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        model.ir_version = 6

        count = inject_hadamard_inputs(model, {"weight"}, 4)

        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in model.graph.node], [
            "Shape",
            "Reshape",
            "MatMul",
            "Reshape",
            "MatMul",
        ])
        self.assertEqual(model.graph.node[-1].input[0], "clark_hadamard_0_input")

    def test_loads_canonical_runtime_state_and_group_contract(self):
        expected = {"weight": torch.tensor([-1.0])}
        materialization = {
            "quantization": {
                "mode": "ternary",
                "scope": "all_matrix_and_conv",
                "method": "absmean",
                "group_size": 60,
                "rotation": "hadamard",
                "rotation_block": 4,
            },
            "surfaces": {
                "matrix.weight": {
                    "group_size": 60,
                },
            },
        }
        teacher = Mock()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "materialization.json"
            with patch.object(
                portable,
                "load_materialized_state",
                return_value=(materialization, expected),
            ) as load:
                state, quantization, group_sizes, loaded = (
                    portable.load_runtime_state(
                        teacher,
                        manifest,
                    )
                )

        load.assert_called_once_with(
            manifest,
            "runtime_state",
        )
        teacher.load_state_dict.assert_called_once_with(expected, strict=True)
        self.assertIs(state, expected)
        self.assertEqual(quantization["scope"], "all_matrix_and_conv")
        self.assertEqual(group_sizes, {"matrix.weight": 60})
        self.assertIs(loaded, materialization)

    def test_accepts_disjoint_fp16_and_int7_portable_islands(self):
        state = {
            "ternary.weight": torch.zeros(1),
            "io.weight": torch.ones(1, dtype=torch.float16).float(),
            "rstb.weight": torch.zeros(1, 1, 3, 3),
        }
        materialization = {
            "retained_precision_islands": [
                {
                    "name": "io",
                    "precision": "fp16",
                    "storage_dtype": "float16",
                    "weight_values": 1,
                    "weights": ["io.weight"],
                },
                {
                    "name": "rstb",
                    "precision": "int7",
                    "storage_dtype": "packed_int7",
                    "bit_width": 7,
                    "granularity": "per_output",
                    "weight_values": 9,
                    "weights": ["rstb.weight"],
                },
            ]
        }

        contract = portable_precision_contract(
            materialization,
            state,
            set(state),
            {"ternary.weight"},
        )

        self.assertEqual(contract.fp16_weights, {"io.weight"})
        self.assertEqual(contract.int7_weights, {"rstb.weight"})


if __name__ == "__main__":
    unittest.main()
