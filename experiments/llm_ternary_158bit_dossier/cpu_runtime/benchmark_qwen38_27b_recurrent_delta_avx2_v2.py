#!/usr/bin/env python3
"""Run the model-correct Qwen3.8-27B recurrent DeltaNet AVX2 gate."""
from benchmark_qwen38_27b_recurrent_delta_avx2_v1 import main


if __name__ == "__main__":
    main(
        preregistration_filename=(
            "qwen38_27b_recurrent_delta_avx2_preregistration_v2.json"
        ),
        default_receipt=(
            "/workspace/quant-state/qwen38-27b-recurrent-delta-avx2-v2.json"
        ),
        receipt_schema="qwen38_27b_recurrent_delta_avx2_receipt_v2",
    )
