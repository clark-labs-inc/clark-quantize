"""Direct native transcode from the storage codec to the fused CPU layout."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from .qwen38_fixed_support_codec import FixedSupportPayload
from .qwen38_fixed_support_mask64_cpu import Mask64Payload
from .qwen38_gain_hash_layer_codec import (
    GainHashLayerComponents,
    Mask64GainHashLayerCompute,
)


_ERRORS = {
    1: "invalid gain-hash layer transcode argument",
    2: "unsupported gain-hash layer transcode shape",
    3: "invalid or non-canonical gain-hash layer ternary code",
    4: "gain-hash layer transcode output capacity is too small",
}


class GainHashLayerTranscodeCPULibrary:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.library = ctypes.CDLL(str(self.path))
        u8p = ctypes.POINTER(ctypes.c_uint8)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        self.library.cq3_ghlt_mask64_size.argtypes = (
            ctypes.c_uint64,
            ctypes.c_uint64,
            u64p,
        )
        self.library.cq3_ghlt_mask64_size.restype = ctypes.c_int
        self.library.cq3_ghlt_transcode_mask64.argtypes = (
            u8p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            u8p,
            ctypes.c_uint64,
        )
        self.library.cq3_ghlt_transcode_mask64.restype = ctypes.c_int


def load_gain_hash_layer_transcode_cpu_library(
    path: str | os.PathLike[str] | None = None,
) -> GainHashLayerTranscodeCPULibrary | None:
    if path is not None:
        candidates = [Path(path)]
    else:
        explicit = os.environ.get("CLARK_GAIN_HASH_LAYER_TRANSCODE_CPU_LIB")
        candidates = [Path(explicit)] if explicit else [
            Path(__file__).with_name("_qwen38_gain_hash_layer_transcode.so"),
            Path(__file__).with_name("_qwen38_gain_hash_layer_transcode.dylib"),
        ]
    for candidate in candidates:
        try:
            return GainHashLayerTranscodeCPULibrary(candidate)
        except (OSError, AttributeError):
            continue
    return None


def _check_status(status: int) -> None:
    if status != 0:
        raise RuntimeError(
            _ERRORS.get(status, f"gain-hash layer transcode error {status}")
        )


def native_transcode_fixed_support_mask64(
    encoded: FixedSupportPayload,
    *,
    library: GainHashLayerTranscodeCPULibrary,
) -> Mask64Payload:
    if (
        encoded.group_size != 128
        or encoded.nonzero_per_group != 64
        or encoded.group_width_bits != 189
        or encoded.rows <= 0
        or encoded.columns <= 0
        or encoded.columns % 128
        or encoded.bit_length
        != encoded.rows * (encoded.columns // 128) * encoded.group_width_bits
        or len(encoded.payload) != (encoded.bit_length + 7) // 8
    ):
        raise ValueError("gain-hash layer transcode metadata is inconsistent")
    output_bytes = ctypes.c_uint64()
    _check_status(
        library.library.cq3_ghlt_mask64_size(
            encoded.rows, encoded.columns, ctypes.byref(output_bytes)
        )
    )
    source = np.frombuffer(encoded.payload, dtype=np.uint8).copy()
    output = np.empty(output_bytes.value, dtype=np.uint8)
    _check_status(
        library.library.cq3_ghlt_transcode_mask64(
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            source.size,
            encoded.bit_length,
            encoded.rows,
            encoded.columns,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            output.size,
        )
    )
    return Mask64Payload(
        payload=output.tobytes(),
        rows=encoded.rows,
        columns=encoded.columns,
    )


def native_materialize_mask64_gain_hash_layer(
    components: GainHashLayerComponents,
    *,
    library: GainHashLayerTranscodeCPULibrary,
) -> Mask64GainHashLayerCompute:
    """Materialize a complete layer without a dense layer-sized code matrix."""
    return Mask64GainHashLayerCompute(
        ternary=native_transcode_fixed_support_mask64(
            components.ternary, library=library
        ),
        scale_indices=np.ascontiguousarray(components.scale_indices, dtype=np.uint8),
        row_anchors=np.ascontiguousarray(components.row_anchors, dtype=np.float16),
        ratio_codebook=np.ascontiguousarray(
            components.ratio_codebook, dtype=np.float16
        ),
        route=components.route,
        correction_codes=np.ascontiguousarray(
            components.correction_codes, dtype=np.int8
        ),
        correction_row_scales=np.ascontiguousarray(
            components.correction_row_scales, dtype=np.float16
        ),
    )
