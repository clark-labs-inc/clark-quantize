"""Native CPU execution for the v37 rank-16 calibration-span residual."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from .qwen38_calibration_span_residual import Int8ResidualAdapter


FEATURE_AVX2 = 1
FEATURE_FMA = 2
FEATURE_F16C = 4
_ERRORS = {
    1: "invalid pointer argument",
    2: "unsupported tensor shape",
    3: "non-finite numeric input",
    4: "requested CPU ISA is unavailable",
}


class CalibrationResidualCPULibrary:
    """Typed handle for the separately versioned rank-16 residual C ABI."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.library = ctypes.CDLL(str(self.path))
        f32p = ctypes.POINTER(ctypes.c_float)
        i8p = ctypes.POINTER(ctypes.c_int8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        self.library.cq3_calibration_residual_cpu_features.argtypes = ()
        self.library.cq3_calibration_residual_cpu_features.restype = ctypes.c_uint32
        arguments = (
            f32p,
            i8p,
            u16p,
            i8p,
            u16p,
            u16p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            f32p,
            f32p,
            ctypes.c_uint32,
        )
        for name in (
            "cq3_calibration_residual_f32_scalar",
            "cq3_calibration_residual_f32_scalar_trusted",
            "cq3_calibration_residual_f32_avx2",
            "cq3_calibration_residual_f32_avx2_trusted",
        ):
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int

    def features(self) -> dict[str, bool]:
        value = int(self.library.cq3_calibration_residual_cpu_features())
        return {
            "avx2": bool(value & FEATURE_AVX2),
            "fma": bool(value & FEATURE_FMA),
            "f16c": bool(value & FEATURE_F16C),
        }


def load_calibration_residual_cpu_library(
    path: str | os.PathLike[str] | None = None,
) -> CalibrationResidualCPULibrary | None:
    if path is not None:
        candidates = [Path(path)]
    else:
        explicit = os.environ.get("CLARK_CALIBRATION_RESIDUAL_CPU_LIB")
        candidates = [Path(explicit)] if explicit else [
            Path(__file__).with_name("_qwen38_calibration_residual_cpu.so"),
            Path(__file__).with_name("_qwen38_calibration_residual_cpu.dylib"),
        ]
    for candidate in candidates:
        try:
            return CalibrationResidualCPULibrary(candidate)
        except (OSError, AttributeError):
            continue
    return None


def _numpy_adapter(
    adapter: Int8ResidualAdapter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.ascontiguousarray(tensor.detach().cpu().numpy())
        for tensor in (
            adapter.left_codes,
            adapter.left_scales,
            adapter.right_codes,
            adapter.right_scales,
            adapter.bias,
        )
    )


def _check_status(status: int) -> None:
    if status != 0:
        raise RuntimeError(_ERRORS.get(status, f"native calibration-residual error {status}"))


def native_calibration_residual_f32(
    values: np.ndarray,
    adapter: Int8ResidualAdapter,
    *,
    library: CalibrationResidualCPULibrary,
    implementation: str = "auto",
    trusted: bool = False,
    base_output: np.ndarray | None = None,
) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float32)
    left_codes, left_scales, right_codes, right_scales, bias = _numpy_adapter(adapter)
    if source.ndim != 1 or source.shape[0] != left_codes.shape[0]:
        raise ValueError("calibration residual input dimension changed")
    if adapter.rank != 16 or right_codes.shape[1] % 8 != 0:
        raise ValueError("native calibration residual requires rank 16 and eight-aligned output")
    if base_output is None:
        output = np.empty(right_codes.shape[1], dtype=np.float32)
        add = 0
    else:
        output = np.array(base_output, dtype=np.float32, order="C", copy=True)
        if output.ndim != 1 or output.shape[0] != right_codes.shape[1]:
            raise ValueError("calibration residual base output dimension changed")
        add = 1
    if implementation not in {"auto", "scalar", "avx2"}:
        raise ValueError("implementation must be auto, scalar, or avx2")
    features = library.features()
    selected = implementation
    if selected == "auto":
        selected = "avx2" if all(features.values()) else "scalar"
    if selected == "avx2" and not all(features.values()):
        raise RuntimeError("requested CPU ISA is unavailable")
    scratch = np.empty(adapter.rank, dtype=np.float32)
    suffix = "_trusted" if trusted else ""
    function = getattr(
        library.library,
        f"cq3_calibration_residual_f32_{selected}{suffix}",
    )
    status = function(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        left_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        left_scales.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        right_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        right_scales.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        bias.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        left_codes.shape[0],
        right_codes.shape[1],
        adapter.rank,
        scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        add,
    )
    _check_status(status)
    return output
