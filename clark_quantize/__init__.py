"""Clark-Quantize: data-free optimal-scale ternary quantization."""
from .ternary import ternarize, quantize_module, DEFAULT_PROTECT

__all__ = ["ternarize", "quantize_module", "DEFAULT_PROTECT"]
__version__ = "0.1.0"
