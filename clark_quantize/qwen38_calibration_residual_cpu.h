#ifndef CLARK_QUANTIZE_QWEN38_CALIBRATION_RESIDUAL_CPU_H
#define CLARK_QUANTIZE_QWEN38_CALIBRATION_RESIDUAL_CPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CQ3_CALIBRATION_FEATURE_AVX2 = 1U,
    CQ3_CALIBRATION_FEATURE_FMA = 2U,
    CQ3_CALIBRATION_FEATURE_F16C = 4U,
};

uint32_t cq3_calibration_residual_cpu_features(void);

int cq3_calibration_residual_f32_scalar_trusted(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
);

int cq3_calibration_residual_f32_scalar(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
);

int cq3_calibration_residual_f32_avx2_trusted(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
);

int cq3_calibration_residual_f32_avx2(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
);

#ifdef __cplusplus
}
#endif

#endif
