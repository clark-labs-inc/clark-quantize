// Fused rotation -> A8 quantization -> AFF3_X16 GEMV for Qwen/Qwen3.8-27B.
//
// This is the exact deployed algebra selected by the mixed representation:
// six FP16 Givens stages rotate one attention-output activation, group-128
// max-absolute A8 quantization produces the inputs consumed by the promoted
// AFF3_X16 register-lookup kernel, and the affine ternary matrix is evaluated
// without reconstructing a dense weight or decoded group-scale matrix.

#include "qwen38_affine_avx2_register_lookup_v18.cpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kQ8Maximum = 127;

int rotate_quantize_v47(
        const uint16_t * CQ3_RESTRICT angles,
        const uint16_t * CQ3_RESTRICT pairings,
        const float * CQ3_RESTRICT input,
        int columns,
        int stages,
        float * CQ3_RESTRICT rotated,
        int8_t * CQ3_RESTRICT quantized,
        float * CQ3_RESTRICT scales,
        int32_t * CQ3_RESTRICT sums) {
    if (angles == nullptr || pairings == nullptr || input == nullptr ||
            rotated == nullptr || quantized == nullptr || scales == nullptr ||
            sums == nullptr || columns <= 0 || columns % kGroupSize != 0 ||
            columns > UINT16_MAX || stages <= 0) {
        return 1;
    }
    std::memmove(rotated, input, static_cast<size_t>(columns) * sizeof(float));
    const int pairs = columns / 2;
    for (int stage = 0; stage < stages; ++stage) {
        const uint16_t *stage_pairings = pairings + stage * columns;
        const uint16_t *stage_angles = angles + stage * pairs;
        for (int pair = 0; pair < pairs; ++pair) {
            const uint16_t first = stage_pairings[pair * 2 + 0];
            const uint16_t second = stage_pairings[pair * 2 + 1];
            if (first >= columns || second >= columns || first == second) {
                return 2;
            }
            const float theta = half_to_float(stage_angles[pair]);
            const float cosine = std::cos(theta);
            const float sine = std::sin(theta);
            const float left = rotated[first];
            const float right = rotated[second];
            rotated[first] = cosine * left - sine * right;
            rotated[second] = sine * left + cosine * right;
        }
    }

    const int groups = columns / kGroupSize;
    for (int group = 0; group < groups; ++group) {
        const int start = group * kGroupSize;
        float maximum = 0.0f;
        for (int offset = 0; offset < kGroupSize; ++offset) {
            const float value = rotated[start + offset];
            if (!std::isfinite(value)) {
                return 4;
            }
            maximum = std::max(maximum, std::abs(value));
        }
        if (maximum == 0.0f) {
            std::memset(quantized + start, 0, kGroupSize);
            scales[group] = 0.0f;
            sums[group] = 0;
            continue;
        }
        const float scale = maximum / static_cast<float>(kQ8Maximum);
        const float inverse = 1.0f / scale;
        int32_t sum = 0;
        for (int offset = 0; offset < kGroupSize; ++offset) {
            const long rounded = std::lrintf(rotated[start + offset] * inverse);
            const int value = static_cast<int>(std::max(
                -kQ8Maximum, std::min(kQ8Maximum, static_cast<int>(rounded))));
            quantized[start + offset] = static_cast<int8_t>(value);
            sum += value;
        }
        scales[group] = scale;
        sums[group] = sum;
    }
    return 0;
}

}  // namespace

extern "C" {

size_t cq3_aff3_rotation_float_scratch_bytes_v47(int columns) {
    return columns > 0 ? static_cast<size_t>(columns) * sizeof(float) : 0U;
}

size_t cq3_aff3_activation_q8_scratch_bytes_v47(int columns) {
    return columns > 0 ? static_cast<size_t>(columns) * sizeof(int8_t) : 0U;
}

size_t cq3_aff3_activation_group_scratch_bytes_v47(int columns) {
    return columns > 0 && columns % kGroupSize == 0
        ? static_cast<size_t>(columns / kGroupSize) *
            (sizeof(float) + sizeof(int32_t))
        : 0U;
}

int cq3_aff3_rotate_quantize_v47(
        const uint16_t * CQ3_RESTRICT angles,
        const uint16_t * CQ3_RESTRICT pairings,
        const float * CQ3_RESTRICT input,
        int columns,
        int stages,
        float * CQ3_RESTRICT rotated,
        int8_t * CQ3_RESTRICT quantized,
        float * CQ3_RESTRICT scales,
        int32_t * CQ3_RESTRICT sums) {
    return rotate_quantize_v47(
        angles, pairings, input, columns, stages,
        rotated, quantized, scales, sums);
}

int cq3_aff3_rotated_gemv_x16_v47(
        const void * CQ3_RESTRICT weights,
        const uint16_t * CQ3_RESTRICT alpha_anchors,
        const uint16_t * CQ3_RESTRICT mu_anchors,
        const uint16_t * CQ3_RESTRICT alpha_codebook,
        const uint16_t * CQ3_RESTRICT mu_codebook,
        const uint16_t * CQ3_RESTRICT angles,
        const uint16_t * CQ3_RESTRICT pairings,
        const float * CQ3_RESTRICT input,
        int rows,
        int columns,
        int stages,
        float * CQ3_RESTRICT rotated_scratch,
        int8_t * CQ3_RESTRICT q8_scratch,
        float * CQ3_RESTRICT scale_scratch,
        int32_t * CQ3_RESTRICT sum_scratch,
        float * CQ3_RESTRICT output) {
    if (weights == nullptr || alpha_anchors == nullptr || mu_anchors == nullptr ||
            alpha_codebook == nullptr || mu_codebook == nullptr || output == nullptr ||
            rows <= 0 || rows % 16 != 0) {
        return 1;
    }
    const int status = rotate_quantize_v47(
        angles, pairings, input, columns, stages,
        rotated_scratch, q8_scratch, scale_scratch, sum_scratch);
    if (status != 0) {
        return status;
    }
    return cq3_aff3_gemv_x16_register_v18(
        weights,
        alpha_anchors,
        mu_anchors,
        alpha_codebook,
        mu_codebook,
        q8_scratch,
        scale_scratch,
        sum_scratch,
        rows,
        columns,
        output);
}

}  // extern "C"
