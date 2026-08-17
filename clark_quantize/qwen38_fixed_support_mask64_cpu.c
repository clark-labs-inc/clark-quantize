#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CQ3_HSM64_X86 1
#include <immintrin.h>
#else
#define CQ3_HSM64_X86 0
#endif

enum {
    CQ3_HSM64_OK = 0,
    CQ3_HSM64_ERROR_ARGUMENT = 1,
    CQ3_HSM64_ERROR_SHAPE = 2,
    CQ3_HSM64_ERROR_CODE = 3,
    CQ3_HSM64_ERROR_NONFINITE = 4,
    CQ3_HSM64_ERROR_ISA = 5,
};

enum {
    CQ3_HSM64_FEATURE_AVX2 = 1U,
    CQ3_HSM64_FEATURE_FMA = 2U,
    CQ3_HSM64_FEATURE_F16C = 4U,
    CQ3_HSM64_FEATURE_BMI2 = 8U,
};

#define CQ3_HSM64_GROUP 128U
#define CQ3_HSM64_GROUP_BYTES 24U
#define CQ3_HSM64_MAX_GROUPS_PER_ROW 256U

uint32_t cq3_hsm64_cpu_features(void) {
    uint32_t features = 0U;
#if CQ3_HSM64_X86 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    features |= __builtin_cpu_supports("avx2") ? CQ3_HSM64_FEATURE_AVX2 : 0U;
    features |= __builtin_cpu_supports("fma") ? CQ3_HSM64_FEATURE_FMA : 0U;
    features |= __builtin_cpu_supports("f16c") ? CQ3_HSM64_FEATURE_F16C : 0U;
    features |= __builtin_cpu_supports("bmi2") ? CQ3_HSM64_FEATURE_BMI2 : 0U;
#endif
    return features;
}

static float cq3_hsm64_half_to_float(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000U) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1fU;
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            exponent = 127U - 15U + 1U;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                exponent -= 1U;
            }
            mantissa &= 0x03ffU;
            bits = sign | (exponent << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + (127U - 15U)) << 23U) |
            (mantissa << 13U);
    }
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static uint64_t cq3_hsm64_load_le64(const uint8_t *source) {
    uint64_t value = 0U;
    for (uint32_t index = 0U; index < 8U; ++index) {
        value |= (uint64_t)source[index] << (index * 8U);
    }
    return value;
}

static void cq3_hsm64_store_le64(uint8_t *target, uint64_t value) {
    for (uint32_t index = 0U; index < 8U; ++index) {
        target[index] = (uint8_t)(value >> (index * 8U));
    }
}

int cq3_hsm64_matrix_size(
    uint64_t rows,
    uint64_t columns,
    uint64_t *payload_bytes
) {
    if (payload_bytes == NULL) {
        return CQ3_HSM64_ERROR_ARGUMENT;
    }
    if (rows == 0U || columns == 0U || columns % CQ3_HSM64_GROUP != 0U) {
        return CQ3_HSM64_ERROR_SHAPE;
    }
    const uint64_t groups = columns / CQ3_HSM64_GROUP;
    if (rows > UINT64_MAX / groups ||
        rows * groups > UINT64_MAX / CQ3_HSM64_GROUP_BYTES) {
        return CQ3_HSM64_ERROR_SHAPE;
    }
    *payload_bytes = rows * groups * CQ3_HSM64_GROUP_BYTES;
    return CQ3_HSM64_OK;
}

int cq3_hsm64_pack_matrix(
    const int8_t *codes,
    uint64_t rows,
    uint64_t columns,
    uint8_t *payload,
    uint64_t payload_capacity_bytes
) {
    if (codes == NULL || payload == NULL) {
        return CQ3_HSM64_ERROR_ARGUMENT;
    }
    uint64_t required = 0U;
    const int size_status = cq3_hsm64_matrix_size(rows, columns, &required);
    if (size_status != CQ3_HSM64_OK) {
        return size_status;
    }
    if (payload_capacity_bytes < required) {
        return CQ3_HSM64_ERROR_SHAPE;
    }
    const uint64_t groups = rows * (columns / CQ3_HSM64_GROUP);
    for (uint64_t group = 0U; group < groups; ++group) {
        uint64_t support_low = 0U;
        uint64_t support_high = 0U;
        uint64_t signs = 0U;
        uint32_t observed = 0U;
        const int8_t *source = codes + group * CQ3_HSM64_GROUP;
        for (uint32_t position = 0U; position < CQ3_HSM64_GROUP; ++position) {
            const int code = source[position];
            if (code < -1 || code > 1) {
                return CQ3_HSM64_ERROR_CODE;
            }
            if (code == 0) {
                continue;
            }
            if (observed >= 64U) {
                return CQ3_HSM64_ERROR_CODE;
            }
            if (position < 64U) {
                support_low |= UINT64_C(1) << position;
            } else {
                support_high |= UINT64_C(1) << (position - 64U);
            }
            if (code > 0) {
                signs |= UINT64_C(1) << observed;
            }
            observed += 1U;
        }
        if (observed != 64U) {
            return CQ3_HSM64_ERROR_CODE;
        }
        uint8_t *target = payload + group * CQ3_HSM64_GROUP_BYTES;
        cq3_hsm64_store_le64(target, support_low);
        cq3_hsm64_store_le64(target + 8U, support_high);
        cq3_hsm64_store_le64(target + 16U, signs);
    }
    return CQ3_HSM64_OK;
}

static int cq3_hsm64_validate_payload(
    const uint8_t *payload,
    uint64_t groups
) {
    for (uint64_t group = 0U; group < groups; ++group) {
        const uint8_t *source = payload + group * CQ3_HSM64_GROUP_BYTES;
        const uint64_t low = cq3_hsm64_load_le64(source);
        const uint64_t high = cq3_hsm64_load_le64(source + 8U);
        if ((uint32_t)__builtin_popcountll(low) +
                (uint32_t)__builtin_popcountll(high) != 64U) {
            return CQ3_HSM64_ERROR_CODE;
        }
    }
    return CQ3_HSM64_OK;
}

int cq3_hsm64_unpack_matrix(
    const uint8_t *payload,
    uint64_t payload_bytes,
    uint64_t rows,
    uint64_t columns,
    int8_t *codes
) {
    if (payload == NULL || codes == NULL) {
        return CQ3_HSM64_ERROR_ARGUMENT;
    }
    uint64_t required = 0U;
    const int size_status = cq3_hsm64_matrix_size(rows, columns, &required);
    if (size_status != CQ3_HSM64_OK) {
        return size_status;
    }
    if (payload_bytes != required) {
        return CQ3_HSM64_ERROR_SHAPE;
    }
    const uint64_t groups = rows * (columns / CQ3_HSM64_GROUP);
    const int validation = cq3_hsm64_validate_payload(payload, groups);
    if (validation != CQ3_HSM64_OK) {
        return validation;
    }
    for (uint64_t group = 0U; group < groups; ++group) {
        const uint8_t *source = payload + group * CQ3_HSM64_GROUP_BYTES;
        const uint64_t support_low = cq3_hsm64_load_le64(source);
        const uint64_t support_high = cq3_hsm64_load_le64(source + 8U);
        const uint64_t signs = cq3_hsm64_load_le64(source + 16U);
        int8_t *target = codes + group * CQ3_HSM64_GROUP;
        memset(target, 0, CQ3_HSM64_GROUP * sizeof(int8_t));
        uint32_t sign_index = 0U;
        for (uint32_t position = 0U; position < CQ3_HSM64_GROUP; ++position) {
            const uint64_t support = position < 64U ? support_low : support_high;
            const uint32_t bit = position < 64U ? position : position - 64U;
            if (((support >> bit) & 1U) != 0U) {
                target[position] = ((signs >> sign_index) & 1U) != 0U ? 1 : -1;
                sign_index += 1U;
            }
        }
    }
    return CQ3_HSM64_OK;
}

static int cq3_hsm64_validate_gemv(
    const uint8_t *weights,
    uint64_t weight_bytes,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    float *output
) {
    if (weights == NULL || packed_scale_indices == NULL ||
        row_anchors == NULL || ratio_codebook == NULL || activation == NULL ||
        activation_scales == NULL || activation_sums == NULL || output == NULL) {
        return CQ3_HSM64_ERROR_ARGUMENT;
    }
    uint64_t required = 0U;
    const int size_status = cq3_hsm64_matrix_size(rows, columns, &required);
    if (size_status != CQ3_HSM64_OK || weight_bytes != required) {
        return CQ3_HSM64_ERROR_SHAPE;
    }
    const uint64_t groups_per_row = columns / CQ3_HSM64_GROUP;
    if (groups_per_row > CQ3_HSM64_MAX_GROUPS_PER_ROW ||
        rows > UINT64_MAX / groups_per_row) {
        return CQ3_HSM64_ERROR_SHAPE;
    }
    const uint64_t groups = rows * groups_per_row;
    const int weight_status = cq3_hsm64_validate_payload(weights, groups);
    if (weight_status != CQ3_HSM64_OK) {
        return weight_status;
    }
    for (uint32_t index = 0U; index < 16U; ++index) {
        if (!isfinite(cq3_hsm64_half_to_float(ratio_codebook[index]))) {
            return CQ3_HSM64_ERROR_NONFINITE;
        }
    }
    for (uint64_t row = 0U; row < rows; ++row) {
        if (!isfinite(cq3_hsm64_half_to_float(row_anchors[row]))) {
            return CQ3_HSM64_ERROR_NONFINITE;
        }
    }
    for (uint64_t group = 0U; group < groups_per_row; ++group) {
        if (!isfinite(activation_scales[group])) {
            return CQ3_HSM64_ERROR_NONFINITE;
        }
    }
    return CQ3_HSM64_OK;
}

int cq3_hsm64_gemv_q8_row_scale4_scalar_trusted(
    const uint8_t *weights,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    float *output
) {
    (void)activation_sums;
    const uint64_t groups = columns / CQ3_HSM64_GROUP;
    for (uint64_t row = 0U; row < rows; ++row) {
        const float anchor = cq3_hsm64_half_to_float(row_anchors[row]);
        float total = 0.0f;
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint8_t *source = weights +
                (row * groups + group) * CQ3_HSM64_GROUP_BYTES;
            uint64_t support_low = cq3_hsm64_load_le64(source);
            uint64_t support_high = cq3_hsm64_load_le64(source + 8U);
            uint64_t signs = cq3_hsm64_load_le64(source + 16U);
            const int8_t *x = activation + group * CQ3_HSM64_GROUP;
            int32_t dot = 0;
            while (support_low != 0U) {
                const uint32_t position = (uint32_t)__builtin_ctzll(support_low);
                dot += (signs & 1U) != 0U ? (int32_t)x[position] : -(int32_t)x[position];
                signs >>= 1U;
                support_low &= support_low - 1U;
            }
            while (support_high != 0U) {
                const uint32_t position = (uint32_t)__builtin_ctzll(support_high);
                dot += (signs & 1U) != 0U
                    ? (int32_t)x[64U + position]
                    : -(int32_t)x[64U + position];
                signs >>= 1U;
                support_high &= support_high - 1U;
            }
            const uint64_t scale_index = row * groups + group;
            const uint8_t ratio_index = (packed_scale_indices[scale_index / 2U] >>
                ((scale_index & 1U) * 4U)) & 0x0fU;
            const float scale = anchor *
                cq3_hsm64_half_to_float(ratio_codebook[ratio_index]) *
                activation_scales[group];
            total += (float)dot * scale;
        }
        output[row] = total;
    }
    return CQ3_HSM64_OK;
}

int cq3_hsm64_gemv_q8_row_scale4_scalar(
    const uint8_t *weights,
    uint64_t weight_bytes,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    float *output
) {
    const int status = cq3_hsm64_validate_gemv(
        weights, weight_bytes, packed_scale_indices, row_anchors,
        ratio_codebook, activation, activation_scales, activation_sums,
        rows, columns, output
    );
    if (status != CQ3_HSM64_OK) {
        return status;
    }
    return cq3_hsm64_gemv_q8_row_scale4_scalar_trusted(
        weights, packed_scale_indices, row_anchors, ratio_codebook,
        activation, activation_scales, activation_sums,
        rows, columns, output
    );
}

#if CQ3_HSM64_X86 && (defined(__GNUC__) || defined(__clang__))
static uint32_t cq3_hsm64_digit_lut[256];
static atomic_int cq3_hsm64_digit_lut_state = ATOMIC_VAR_INIT(0);

static void cq3_hsm64_initialize_digit_lut(void) {
    int expected = 0;
    if (atomic_compare_exchange_strong_explicit(
            &cq3_hsm64_digit_lut_state, &expected, 1,
            memory_order_acq_rel, memory_order_acquire)) {
        for (uint32_t positive = 0U; positive < 16U; ++positive) {
            for (uint32_t support = 0U; support < 16U; ++support) {
                uint32_t digits = 0U;
                for (uint32_t lane = 0U; lane < 4U; ++lane) {
                    const uint32_t supported = (support >> lane) & 1U;
                    const uint32_t is_positive = (positive >> lane) & 1U;
                    const uint32_t digit = 1U - supported + 2U * is_positive;
                    digits |= digit << (lane * 8U);
                }
                cq3_hsm64_digit_lut[support | (positive << 4U)] = digits;
            }
        }
        atomic_store_explicit(&cq3_hsm64_digit_lut_state, 2, memory_order_release);
        return;
    }
    while (atomic_load_explicit(
            &cq3_hsm64_digit_lut_state, memory_order_acquire) != 2) {
    }
}

__attribute__((target("avx2,bmi2,f16c,fma")))
static int cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_impl(
    const uint8_t *weights,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    float *output
) {
    cq3_hsm64_initialize_digit_lut();
    float ratios[16];
    _mm256_storeu_ps(
        ratios, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)ratio_codebook))
    );
    _mm256_storeu_ps(
        ratios + 8U,
        _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(ratio_codebook + 8U)))
    );
    const __m256i ones = _mm256_set1_epi16(1);
    const uint64_t groups = columns / CQ3_HSM64_GROUP;
    for (uint64_t row = 0U; row < rows; ++row) {
        const float anchor = _cvtsh_ss(row_anchors[row]);
        float total = 0.0f;
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint8_t *source = weights +
                (row * groups + group) * CQ3_HSM64_GROUP_BYTES;
            const uint64_t support_low = cq3_hsm64_load_le64(source);
            const uint64_t support_high = cq3_hsm64_load_le64(source + 8U);
            const uint64_t signs = cq3_hsm64_load_le64(source + 16U);
            const uint32_t low_count = (uint32_t)__builtin_popcountll(support_low);
            const uint64_t low_signs = low_count == 64U
                ? signs
                : signs & ((UINT64_C(1) << low_count) - 1U);
            const uint64_t high_signs = low_count == 64U ? 0U : signs >> low_count;
            const uint64_t positive_low = _pdep_u64(low_signs, support_low);
            const uint64_t positive_high = _pdep_u64(high_signs, support_high);
            __m256i combined = _mm256_setzero_si256();
            const int8_t *x = activation + group * CQ3_HSM64_GROUP;
            for (uint32_t segment = 0U; segment < 4U; ++segment) {
                const uint32_t shift = (segment & 1U) * 32U;
                const uint32_t support = (uint32_t)(
                    (segment < 2U ? support_low : support_high) >> shift
                );
                const uint32_t positive = (uint32_t)(
                    (segment < 2U ? positive_low : positive_high) >> shift
                );
                uint32_t digits[8];
                for (uint32_t nibble = 0U; nibble < 8U; ++nibble) {
                    const uint32_t offset = nibble * 4U;
                    const uint32_t index = ((support >> offset) & 0x0fU) |
                        (((positive >> offset) & 0x0fU) << 4U);
                    digits[nibble] = cq3_hsm64_digit_lut[index];
                }
                const __m256i digit_vector = _mm256_loadu_si256(
                    (const __m256i *)digits
                );
                const __m256i values = _mm256_loadu_si256(
                    (const __m256i *)(x + segment * 32U)
                );
                combined = _mm256_add_epi16(
                    combined, _mm256_maddubs_epi16(digit_vector, values)
                );
            }
            const __m256i lanes = _mm256_madd_epi16(combined, ones);
            __m128i horizontal = _mm_add_epi32(
                _mm256_castsi256_si128(lanes),
                _mm256_extracti128_si256(lanes, 1)
            );
            horizontal = _mm_hadd_epi32(horizontal, horizontal);
            horizontal = _mm_hadd_epi32(horizontal, horizontal);
            const int32_t dot = _mm_cvtsi128_si32(horizontal) - activation_sums[group];
            const uint64_t scale_index = row * groups + group;
            const uint8_t ratio_index = (packed_scale_indices[scale_index / 2U] >>
                ((scale_index & 1U) * 4U)) & 0x0fU;
            total += (float)dot * anchor * ratios[ratio_index] *
                activation_scales[group];
        }
        output[row] = total;
    }
    return CQ3_HSM64_OK;
}
#endif

int cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_trusted(
    const uint8_t *weights,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    float *output
) {
#if CQ3_HSM64_X86 && (defined(__GNUC__) || defined(__clang__))
    return cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_impl(
        weights, packed_scale_indices, row_anchors, ratio_codebook,
        activation, activation_scales, activation_sums,
        rows, columns, output
    );
#else
    (void)weights;
    (void)packed_scale_indices;
    (void)row_anchors;
    (void)ratio_codebook;
    (void)activation;
    (void)activation_scales;
    (void)activation_sums;
    (void)rows;
    (void)columns;
    (void)output;
    return CQ3_HSM64_ERROR_ISA;
#endif
}

int cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2(
    const uint8_t *weights,
    uint64_t weight_bytes,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    float *output
) {
    const int status = cq3_hsm64_validate_gemv(
        weights, weight_bytes, packed_scale_indices, row_anchors,
        ratio_codebook, activation, activation_scales, activation_sums,
        rows, columns, output
    );
    if (status != CQ3_HSM64_OK) {
        return status;
    }
    const uint32_t required = CQ3_HSM64_FEATURE_AVX2 |
        CQ3_HSM64_FEATURE_FMA | CQ3_HSM64_FEATURE_F16C |
        CQ3_HSM64_FEATURE_BMI2;
    if ((cq3_hsm64_cpu_features() & required) != required) {
        return CQ3_HSM64_ERROR_ISA;
    }
    return cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_trusted(
        weights, packed_scale_indices, row_anchors, ratio_codebook,
        activation, activation_scales, activation_sums,
        rows, columns, output
    );
}
