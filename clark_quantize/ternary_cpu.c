#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CQ3_X86 1
#include <immintrin.h>
#else
#define CQ3_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define CQ3_ARM64 1
#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#if defined(__has_include)
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
#endif
#endif
#else
#define CQ3_ARM64 0
#endif

#define CQ3_MAX_GROUPS_PER_ROW 256U
#define CQ3_RADIX5_BYTES_PER_GROUP 26U
#define CQ3_RADIX5_SEGMENTS_PADDED 28U
#define CQ3_RADIX5_ACTIVATION_STRIDE (CQ3_RADIX5_SEGMENTS_PADDED * 8U)
#define CQ3_RADIX5_PLANE_WIDTH 32U
#define CQ3_RADIX5_PLANES 5U
#define CQ3_RADIX5_PLANE_ACTIVATION_STRIDE \
    (CQ3_RADIX5_PLANE_WIDTH * CQ3_RADIX5_PLANES)

#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
#define CQ3_RADIX5_DIGITS(value) ( \
    ((uint64_t)(((value) / 1U) % 3U) << 0U) | \
    ((uint64_t)(((value) / 3U) % 3U) << 8U) | \
    ((uint64_t)(((value) / 9U) % 3U) << 16U) | \
    ((uint64_t)(((value) / 27U) % 3U) << 24U) | \
    ((uint64_t)(((value) / 81U) % 3U) << 32U) \
)
#define CQ3_RADIX5_ROW16(base) \
    CQ3_RADIX5_DIGITS((base) + 0U), CQ3_RADIX5_DIGITS((base) + 1U), \
    CQ3_RADIX5_DIGITS((base) + 2U), CQ3_RADIX5_DIGITS((base) + 3U), \
    CQ3_RADIX5_DIGITS((base) + 4U), CQ3_RADIX5_DIGITS((base) + 5U), \
    CQ3_RADIX5_DIGITS((base) + 6U), CQ3_RADIX5_DIGITS((base) + 7U), \
    CQ3_RADIX5_DIGITS((base) + 8U), CQ3_RADIX5_DIGITS((base) + 9U), \
    CQ3_RADIX5_DIGITS((base) + 10U), CQ3_RADIX5_DIGITS((base) + 11U), \
    CQ3_RADIX5_DIGITS((base) + 12U), CQ3_RADIX5_DIGITS((base) + 13U), \
    CQ3_RADIX5_DIGITS((base) + 14U), CQ3_RADIX5_DIGITS((base) + 15U)

static const uint64_t cq3_radix5_digit_lookup_static[256] = {
    CQ3_RADIX5_ROW16(0U), CQ3_RADIX5_ROW16(16U),
    CQ3_RADIX5_ROW16(32U), CQ3_RADIX5_ROW16(48U),
    CQ3_RADIX5_ROW16(64U), CQ3_RADIX5_ROW16(80U),
    CQ3_RADIX5_ROW16(96U), CQ3_RADIX5_ROW16(112U),
    CQ3_RADIX5_ROW16(128U), CQ3_RADIX5_ROW16(144U),
    CQ3_RADIX5_ROW16(160U), CQ3_RADIX5_ROW16(176U),
    CQ3_RADIX5_ROW16(192U), CQ3_RADIX5_ROW16(208U),
    CQ3_RADIX5_ROW16(224U), CQ3_RADIX5_ROW16(240U),
};

#undef CQ3_RADIX5_ROW16
#undef CQ3_RADIX5_DIGITS
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CQ3_RESTRICT __restrict__
#else
#define CQ3_RESTRICT
#endif

enum {
    CQ3_OK = 0,
    CQ3_ERROR_ARGUMENT = 1,
    CQ3_ERROR_SHAPE = 2,
    CQ3_ERROR_CODE = 3,
    CQ3_ERROR_NONFINITE = 4,
    CQ3_ERROR_ISA = 5,
};

enum {
    CQ3_FEATURE_AVX2 = 1,
    CQ3_FEATURE_F16C = 2,
    CQ3_FEATURE_FMA = 4,
    CQ3_FEATURE_NEON = 8,
    CQ3_FEATURE_DOTPROD = 16,
};

#if CQ3_ARM64
static int cq3_arm_dotprod_available(void) {
#if defined(__APPLE__)
    int available = 0;
    size_t size = sizeof(available);
    if (sysctlbyname(
            "hw.optional.arm.FEAT_DotProd", &available, &size, NULL, 0
        ) == 0) {
        return available != 0;
    }
    return 0;
#elif defined(__linux__) && defined(AT_HWCAP) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0UL;
#elif defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}
#endif

uint32_t cq3_cpu_features(void) {
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    uint32_t features = 0U;
    features |= __builtin_cpu_supports("avx2") ? CQ3_FEATURE_AVX2 : 0U;
    features |= __builtin_cpu_supports("f16c") ? CQ3_FEATURE_F16C : 0U;
    features |= __builtin_cpu_supports("fma") ? CQ3_FEATURE_FMA : 0U;
    return features;
#elif CQ3_ARM64
    uint32_t features = CQ3_FEATURE_NEON;
    features |= cq3_arm_dotprod_available() ? CQ3_FEATURE_DOTPROD : 0U;
    return features;
#else
    return 0U;
#endif
}

static float cq3_half_to_float(uint16_t value) {
    const uint32_t sign = ((uint32_t)value & 0x8000U) << 16U;
    uint32_t exponent = ((uint32_t)value >> 10U) & 0x1fU;
    uint32_t mantissa = (uint32_t)value & 0x03ffU;
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
        exponent += 127U - 15U;
        bits = sign | (exponent << 23U) | (mantissa << 13U);
    }
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

#if CQ3_ARM64 && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
static inline float cq3_half_to_float_arm(uint16_t value) {
    float16_t half;
    memcpy(&half, &value, sizeof(half));
    return (float)half;
}
#endif

int cq3_pack_aligned2(
    const int8_t *codes,
    uint64_t count,
    uint8_t *packed
) {
    if ((count != 0U && (codes == NULL || packed == NULL))) {
        return CQ3_ERROR_ARGUMENT;
    }
    const uint64_t packed_count = (count + 3U) / 4U;
    for (uint64_t byte_index = 0U; byte_index < packed_count; ++byte_index) {
        uint8_t value = 0U;
        for (uint32_t lane = 0U; lane < 4U; ++lane) {
            const uint64_t index = byte_index * 4U + lane;
            const int code = index < count ? (int)codes[index] : 0;
            if (code < -1 || code > 1) {
                return CQ3_ERROR_CODE;
            }
            value |= (uint8_t)(code + 1) << (lane * 2U);
        }
        packed[byte_index] = value;
    }
    return CQ3_OK;
}

int cq3_unpack_aligned2(
    const uint8_t *packed,
    uint64_t count,
    int8_t *codes
) {
    if ((count != 0U && (packed == NULL || codes == NULL))) {
        return CQ3_ERROR_ARGUMENT;
    }
    for (uint64_t index = 0U; index < count; ++index) {
        const uint8_t digit = (packed[index / 4U] >> ((index % 4U) * 2U)) & 3U;
        if (digit == 3U) {
            return CQ3_ERROR_CODE;
        }
        codes[index] = (int8_t)digit - 1;
    }
    const uint32_t tail = (uint32_t)(count % 4U);
    if (tail != 0U) {
        const uint8_t last = packed[count / 4U];
        for (uint32_t lane = tail; lane < 4U; ++lane) {
            if (((last >> (lane * 2U)) & 3U) != 1U) {
                return CQ3_ERROR_CODE;
            }
        }
    }
    return CQ3_OK;
}

int cq3_pack_group128_2bit(
    const int8_t *codes,
    uint64_t count,
    uint8_t *packed
) {
    if (codes == NULL || packed == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (count == 0U || count % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = count / 128U;
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t lane = 0U; lane < 32U; ++lane) {
            uint8_t value = 0U;
            for (uint32_t quarter = 0U; quarter < 4U; ++quarter) {
                const int code = (int)codes[group * 128U + quarter * 32U + lane];
                if (code < -1 || code > 1) {
                    return CQ3_ERROR_CODE;
                }
                value |= (uint8_t)(code + 1) << (quarter * 2U);
            }
            packed[group * 32U + lane] = value;
        }
    }
    return CQ3_OK;
}

int cq3_unpack_group128_2bit(
    const uint8_t *packed,
    uint64_t count,
    int8_t *codes
) {
    if (packed == NULL || codes == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (count == 0U || count % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = count / 128U;
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t lane = 0U; lane < 32U; ++lane) {
            const uint8_t value = packed[group * 32U + lane];
            for (uint32_t quarter = 0U; quarter < 4U; ++quarter) {
                const uint8_t digit = (value >> (quarter * 2U)) & 3U;
                if (digit == 3U) {
                    return CQ3_ERROR_CODE;
                }
                codes[group * 128U + quarter * 32U + lane] = (int8_t)digit - 1;
            }
        }
    }
    return CQ3_OK;
}

int cq3_pack_group128_radix5(
    const int8_t *codes,
    uint64_t count,
    uint8_t *packed
) {
    if (codes == NULL || packed == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (count == 0U || count % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = count / 128U;
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t segment = 0U; segment < CQ3_RADIX5_BYTES_PER_GROUP; ++segment) {
            uint32_t value = 0U;
            uint32_t power = 1U;
            for (uint32_t lane = 0U; lane < 5U; ++lane) {
                const uint32_t position = segment * 5U + lane;
                int code = -1;
                if (position < 128U) {
                    code = (int)codes[group * 128U + position];
                    if (code < -1 || code > 1) {
                        return CQ3_ERROR_CODE;
                    }
                }
                value += (uint32_t)(code + 1) * power;
                power *= 3U;
            }
            packed[group * CQ3_RADIX5_BYTES_PER_GROUP + segment] = (uint8_t)value;
        }
    }
    return CQ3_OK;
}

int cq3_unpack_group128_radix5(
    const uint8_t *packed,
    uint64_t count,
    int8_t *codes
) {
    if (packed == NULL || codes == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (count == 0U || count % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = count / 128U;
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t segment = 0U; segment < CQ3_RADIX5_BYTES_PER_GROUP; ++segment) {
            uint32_t value = packed[group * CQ3_RADIX5_BYTES_PER_GROUP + segment];
            if (value > 242U) {
                return CQ3_ERROR_CODE;
            }
            for (uint32_t lane = 0U; lane < 5U; ++lane) {
                const uint32_t position = segment * 5U + lane;
                const uint32_t digit = value % 3U;
                value /= 3U;
                if (position < 128U) {
                    codes[group * 128U + position] = (int8_t)digit - 1;
                } else if (digit != 0U) {
                    return CQ3_ERROR_CODE;
                }
            }
        }
    }
    return CQ3_OK;
}

int cq3_quantize_f32_q8(
    const float *input,
    uint64_t columns,
    uint32_t group_size,
    int8_t *quantized,
    float *scales,
    int32_t *sums
) {
    if (input == NULL || quantized == NULL || scales == NULL || sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (group_size == 0U || columns == 0U || columns % group_size != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = columns / group_size;
    for (uint64_t group = 0U; group < groups; ++group) {
        const uint64_t offset = group * group_size;
        float maximum = 0.0f;
        for (uint32_t lane = 0U; lane < group_size; ++lane) {
            const float value = input[offset + lane];
            if (!isfinite(value)) {
                return CQ3_ERROR_NONFINITE;
            }
            const float absolute = fabsf(value);
            maximum = absolute > maximum ? absolute : maximum;
        }
        const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
        int32_t sum = 0;
        for (uint32_t lane = 0U; lane < group_size; ++lane) {
            long rounded = lrintf(input[offset + lane] / scale);
            rounded = rounded < -127L ? -127L : rounded;
            rounded = rounded > 127L ? 127L : rounded;
            quantized[offset + lane] = (int8_t)rounded;
            sum += (int32_t)rounded;
        }
        scales[group] = scale;
        sums[group] = sum;
    }
    return CQ3_OK;
}

int cq3_expand_q8_radix5(
    const int8_t *activation,
    uint64_t columns,
    int8_t *expanded
) {
    if (activation == NULL || expanded == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (columns == 0U || columns % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = columns / 128U;
    memset(expanded, 0, groups * CQ3_RADIX5_ACTIVATION_STRIDE);
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t position = 0U; position < 128U; ++position) {
            const uint32_t segment = position / 5U;
            const uint32_t lane = position % 5U;
            expanded[group * CQ3_RADIX5_ACTIVATION_STRIDE + segment * 8U + lane] =
                activation[group * 128U + position];
        }
    }
    return CQ3_OK;
}

int cq3_expand_q8_radix5_planes(
    const int8_t *activation,
    uint64_t columns,
    int8_t *expanded
) {
    if (activation == NULL || expanded == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (columns == 0U || columns % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = columns / 128U;
    memset(expanded, 0, groups * CQ3_RADIX5_PLANE_ACTIVATION_STRIDE);
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t position = 0U; position < 128U; ++position) {
            const uint32_t segment = position / 5U;
            const uint32_t plane = position % 5U;
            expanded[
                group * CQ3_RADIX5_PLANE_ACTIVATION_STRIDE
                + plane * CQ3_RADIX5_PLANE_WIDTH
                + segment
            ] = activation[group * 128U + position];
        }
    }
    return CQ3_OK;
}

static int cq3_validate_gemv(
    const uint8_t *weights,
    const uint16_t *weight_scales,
    const int8_t *activation,
    const float *activation_scales,
    float *output,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size
) {
    if (weights == NULL || weight_scales == NULL || activation == NULL ||
        activation_scales == NULL || output == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (rows == 0U || columns == 0U || group_size != 128U ||
        columns % group_size != 0U || columns % 4U != 0U ||
        columns / group_size > CQ3_MAX_GROUPS_PER_ROW) {
        return CQ3_ERROR_SHAPE;
    }
    return CQ3_OK;
}

int cq3_gemv_q8_group128_2bit_scalar(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = columns / 4U;
    for (uint64_t row = 0U; row < rows; ++row) {
        float total = 0.0f;
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint64_t code_offset = row * row_bytes + group * (group_size / 4U);
            const uint64_t activation_offset = group * group_size;
            int32_t dot = 0;
            for (uint32_t lane = 0U; lane < group_size; ++lane) {
                const uint8_t digit =
                    (weights[code_offset + lane % 32U] >> ((lane / 32U) * 2U)) & 3U;
                if (digit == 3U) {
                    return CQ3_ERROR_CODE;
                }
                dot += ((int32_t)digit - 1) * (int32_t)activation[activation_offset + lane];
            }
            const float weight_scale = cq3_half_to_float(weight_scales[row * groups + group]);
            total += (float)dot * weight_scale * activation_scales[group];
        }
        output[row] = total;
    }
    return CQ3_OK;
}

#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,f16c,fma")))
static int cq3_gemv_q8_group128_2bit_avx2_impl(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = columns / 4U;
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    for (uint64_t row = 0U; row < rows; ++row) {
        float combined_scales[CQ3_MAX_GROUPS_PER_ROW];
        uint64_t scale_group = 0U;
        for (; scale_group + 8U <= groups; scale_group += 8U) {
            const __m128i half_scales = _mm_loadu_si128(
                (const __m128i *)(weight_scales + row * groups + scale_group)
            );
            const __m256 float_scales = _mm256_mul_ps(
                _mm256_cvtph_ps(half_scales),
                _mm256_loadu_ps(activation_scales + scale_group)
            );
            _mm256_storeu_ps(combined_scales + scale_group, float_scales);
        }
        for (; scale_group < groups; ++scale_group) {
            combined_scales[scale_group] =
                _cvtsh_ss(weight_scales[row * groups + scale_group])
                * activation_scales[scale_group];
        }
        __m256 total = _mm256_setzero_ps();
        for (uint64_t group = 0U; group < groups; ++group) {
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(weights + row * row_bytes + group * 32U)
            );
            const int8_t *x = activation + group * 128U;
            const __m256i digits0 = _mm256_and_si256(packed, mask);
            const __m256i digits1 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
            const __m256i digits2 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            const __m256i digits3 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);
            const __m256i values0 = _mm256_loadu_si256((const __m256i *)(x + 0U));
            const __m256i values1 = _mm256_loadu_si256((const __m256i *)(x + 32U));
            const __m256i values2 = _mm256_loadu_si256((const __m256i *)(x + 64U));
            const __m256i values3 = _mm256_loadu_si256((const __m256i *)(x + 96U));
            const __m256i products01 = _mm256_add_epi16(
                _mm256_maddubs_epi16(digits0, values0),
                _mm256_maddubs_epi16(digits1, values1)
            );
            const __m256i products23 = _mm256_add_epi16(
                _mm256_maddubs_epi16(digits2, values2),
                _mm256_maddubs_epi16(digits3, values3)
            );
            const __m256i accumulator = _mm256_madd_epi16(
                _mm256_add_epi16(products01, products23),
                ones
            );
            const __m256i centered = _mm256_sub_epi32(
                accumulator,
                _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, activation_sums[group])
            );
            total = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(centered),
                _mm256_set1_ps(combined_scales[group]),
                total
            );
        }
        __m128 horizontal = _mm_add_ps(
            _mm256_castps256_ps128(total),
            _mm256_extractf128_ps(total, 1)
        );
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        output[row] = _mm_cvtss_f32(horizontal);
    }
    return CQ3_OK;
}
#endif

int cq3_gemv_q8_group128_2bit_avx2(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    const uint32_t required = CQ3_FEATURE_AVX2 | CQ3_FEATURE_F16C | CQ3_FEATURE_FMA;
    if ((cq3_cpu_features() & required) != required) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_2bit_avx2_impl(
        weights, weight_scales, activation, activation_scales,
        activation_sums, rows, columns, group_size, output
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

#if CQ3_ARM64 && defined(__ARM_FEATURE_DOTPROD)
static inline int32x4_t cq3_arm_dot_group128(
    const uint8_t *packed_group,
    uint8x16_t mask,
    int8x16_t x0,
    int8x16_t x1,
    int8x16_t x2,
    int8x16_t x3,
    int8x16_t x4,
    int8x16_t x5,
    int8x16_t x6,
    int8x16_t x7
) {
    const uint8x16_t packed0 = vld1q_u8(packed_group);
    const uint8x16_t packed1 = vld1q_u8(packed_group + 16U);
    int32x4_t dot = vdupq_n_s32(0);
    dot = vdotq_s32(
        dot, vreinterpretq_s8_u8(vandq_u8(packed0, mask)), x0
    );
    dot = vdotq_s32(
        dot, vreinterpretq_s8_u8(vandq_u8(packed1, mask)), x1
    );
    dot = vdotq_s32(
        dot,
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed0, 2), mask)),
        x2
    );
    dot = vdotq_s32(
        dot,
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed1, 2), mask)),
        x3
    );
    dot = vdotq_s32(
        dot,
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed0, 4), mask)),
        x4
    );
    dot = vdotq_s32(
        dot,
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed1, 4), mask)),
        x5
    );
    dot = vdotq_s32(
        dot,
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed0, 6), mask)),
        x6
    );
    return vdotq_s32(
        dot,
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed1, 6), mask)),
        x7
    );
}

static inline float cq3_arm_reduce_f32(float32x4_t value) {
    float32x2_t horizontal = vadd_f32(
        vget_low_f32(value), vget_high_f32(value)
    );
    horizontal = vpadd_f32(horizontal, horizontal);
    return vget_lane_f32(horizontal, 0);
}

static int cq3_gemv_q8_group128_2bit_arm_dotprod_impl(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    (void)group_size;
    const uint64_t groups = columns / 128U;
    const uint64_t row_bytes = columns / 4U;
    const uint8x16_t mask = vdupq_n_u8(3U);
    uint64_t row = 0U;
    for (; row + 4U <= rows; row += 4U) {
        float32x4_t total0 = vdupq_n_f32(0.0f);
        float32x4_t total1 = vdupq_n_f32(0.0f);
        float32x4_t total2 = vdupq_n_f32(0.0f);
        float32x4_t total3 = vdupq_n_f32(0.0f);
        for (uint64_t group = 0U; group < groups; ++group) {
            const int8_t *x = activation + group * 128U;
            const int8x16_t x0 = vld1q_s8(x);
            const int8x16_t x1 = vld1q_s8(x + 16U);
            const int8x16_t x2 = vld1q_s8(x + 32U);
            const int8x16_t x3 = vld1q_s8(x + 48U);
            const int8x16_t x4 = vld1q_s8(x + 64U);
            const int8x16_t x5 = vld1q_s8(x + 80U);
            const int8x16_t x6 = vld1q_s8(x + 96U);
            const int8x16_t x7 = vld1q_s8(x + 112U);
            int32x4_t dot0 = cq3_arm_dot_group128(
                weights + (row + 0U) * row_bytes + group * 32U,
                mask, x0, x1, x2, x3, x4, x5, x6, x7
            );
            int32x4_t dot1 = cq3_arm_dot_group128(
                weights + (row + 1U) * row_bytes + group * 32U,
                mask, x0, x1, x2, x3, x4, x5, x6, x7
            );
            int32x4_t dot2 = cq3_arm_dot_group128(
                weights + (row + 2U) * row_bytes + group * 32U,
                mask, x0, x1, x2, x3, x4, x5, x6, x7
            );
            int32x4_t dot3 = cq3_arm_dot_group128(
                weights + (row + 3U) * row_bytes + group * 32U,
                mask, x0, x1, x2, x3, x4, x5, x6, x7
            );
            const int32_t activation_sum = activation_sums[group];
            dot0 = vsetq_lane_s32(vgetq_lane_s32(dot0, 0) - activation_sum, dot0, 0);
            dot1 = vsetq_lane_s32(vgetq_lane_s32(dot1, 0) - activation_sum, dot1, 0);
            dot2 = vsetq_lane_s32(vgetq_lane_s32(dot2, 0) - activation_sum, dot2, 0);
            dot3 = vsetq_lane_s32(vgetq_lane_s32(dot3, 0) - activation_sum, dot3, 0);
            const float activation_scale = activation_scales[group];
            total0 = vfmaq_n_f32(
                total0, vcvtq_f32_s32(dot0),
                cq3_half_to_float_arm(weight_scales[(row + 0U) * groups + group])
                    * activation_scale
            );
            total1 = vfmaq_n_f32(
                total1, vcvtq_f32_s32(dot1),
                cq3_half_to_float_arm(weight_scales[(row + 1U) * groups + group])
                    * activation_scale
            );
            total2 = vfmaq_n_f32(
                total2, vcvtq_f32_s32(dot2),
                cq3_half_to_float_arm(weight_scales[(row + 2U) * groups + group])
                    * activation_scale
            );
            total3 = vfmaq_n_f32(
                total3, vcvtq_f32_s32(dot3),
                cq3_half_to_float_arm(weight_scales[(row + 3U) * groups + group])
                    * activation_scale
            );
        }
        output[row + 0U] = cq3_arm_reduce_f32(total0);
        output[row + 1U] = cq3_arm_reduce_f32(total1);
        output[row + 2U] = cq3_arm_reduce_f32(total2);
        output[row + 3U] = cq3_arm_reduce_f32(total3);
    }
    for (; row < rows; ++row) {
        float32x4_t total = vdupq_n_f32(0.0f);
        for (uint64_t group = 0U; group < groups; ++group) {
            const int8_t *x = activation + group * 128U;
            int32x4_t dot = cq3_arm_dot_group128(
                weights + row * row_bytes + group * 32U,
                mask,
                vld1q_s8(x), vld1q_s8(x + 16U),
                vld1q_s8(x + 32U), vld1q_s8(x + 48U),
                vld1q_s8(x + 64U), vld1q_s8(x + 80U),
                vld1q_s8(x + 96U), vld1q_s8(x + 112U)
            );
            dot = vsetq_lane_s32(
                vgetq_lane_s32(dot, 0) - activation_sums[group], dot, 0
            );
            total = vfmaq_n_f32(
                total, vcvtq_f32_s32(dot),
                cq3_half_to_float_arm(weight_scales[row * groups + group])
                    * activation_scales[group]
            );
        }
        output[row] = cq3_arm_reduce_f32(total);
    }
    return CQ3_OK;
}
#endif

int cq3_gemv_q8_group128_2bit_arm_dotprod(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_ARM64 && defined(__ARM_FEATURE_DOTPROD)
    if (!cq3_arm_dotprod_available()) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_2bit_arm_dotprod_impl(
        weights, weight_scales, activation, activation_scales,
        activation_sums, rows, columns, group_size, output
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

int cq3_swiglu_f32(
    const float *gate,
    const float *up,
    uint64_t count,
    float *output
) {
    if (gate == NULL || up == NULL || output == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    for (uint64_t index = 0U; index < count; ++index) {
        if (!isfinite(gate[index]) || !isfinite(up[index])) {
            return CQ3_ERROR_NONFINITE;
        }
        output[index] =
            (gate[index] / (1.0f + expf(-gate[index]))) * up[index];
    }
    return CQ3_OK;
}

#if CQ3_ARM64 && defined(__ARM_FEATURE_DOTPROD)
static inline float32x4_t cq3_arm_accumulate_group128(
    float32x4_t total,
    const uint8_t *packed_group,
    uint16_t weight_scale,
    float activation_scale,
    int32_t activation_sum,
    uint8x16_t mask,
    int8x16_t x0,
    int8x16_t x1,
    int8x16_t x2,
    int8x16_t x3,
    int8x16_t x4,
    int8x16_t x5,
    int8x16_t x6,
    int8x16_t x7
) {
    int32x4_t dot = cq3_arm_dot_group128(
        packed_group, mask, x0, x1, x2, x3, x4, x5, x6, x7
    );
    dot = vsetq_lane_s32(
        vgetq_lane_s32(dot, 0) - activation_sum, dot, 0
    );
    return vfmaq_n_f32(
        total, vcvtq_f32_s32(dot),
        cq3_half_to_float_arm(weight_scale) * activation_scale
    );
}

static int cq3_gemv_q8_group128_2bit_swiglu_arm_dotprod_impl(
    const uint8_t *CQ3_RESTRICT gate_weights,
    const uint16_t *CQ3_RESTRICT gate_scales,
    const uint8_t *CQ3_RESTRICT up_weights,
    const uint16_t *CQ3_RESTRICT up_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    float *CQ3_RESTRICT output
) {
    const uint64_t groups = columns / 128U;
    const uint64_t row_bytes = columns / 4U;
    const uint8x16_t mask = vdupq_n_u8(3U);
    uint64_t row = 0U;
    for (; row + 4U <= rows; row += 4U) {
        float32x4_t gate0 = vdupq_n_f32(0.0f);
        float32x4_t gate1 = vdupq_n_f32(0.0f);
        float32x4_t gate2 = vdupq_n_f32(0.0f);
        float32x4_t gate3 = vdupq_n_f32(0.0f);
        float32x4_t up0 = vdupq_n_f32(0.0f);
        float32x4_t up1 = vdupq_n_f32(0.0f);
        float32x4_t up2 = vdupq_n_f32(0.0f);
        float32x4_t up3 = vdupq_n_f32(0.0f);
        for (uint64_t group = 0U; group < groups; ++group) {
            const int8_t *x = activation + group * 128U;
            const int8x16_t x0 = vld1q_s8(x);
            const int8x16_t x1 = vld1q_s8(x + 16U);
            const int8x16_t x2 = vld1q_s8(x + 32U);
            const int8x16_t x3 = vld1q_s8(x + 48U);
            const int8x16_t x4 = vld1q_s8(x + 64U);
            const int8x16_t x5 = vld1q_s8(x + 80U);
            const int8x16_t x6 = vld1q_s8(x + 96U);
            const int8x16_t x7 = vld1q_s8(x + 112U);
            const float activation_scale = activation_scales[group];
            const int32_t activation_sum = activation_sums[group];
            gate0 = cq3_arm_accumulate_group128(
                gate0, gate_weights + (row + 0U) * row_bytes + group * 32U,
                gate_scales[(row + 0U) * groups + group],
                activation_scale, activation_sum, mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
            up0 = cq3_arm_accumulate_group128(
                up0, up_weights + (row + 0U) * row_bytes + group * 32U,
                up_scales[(row + 0U) * groups + group],
                activation_scale, activation_sum, mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
            gate1 = cq3_arm_accumulate_group128(
                gate1, gate_weights + (row + 1U) * row_bytes + group * 32U,
                gate_scales[(row + 1U) * groups + group],
                activation_scale, activation_sum, mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
            up1 = cq3_arm_accumulate_group128(
                up1, up_weights + (row + 1U) * row_bytes + group * 32U,
                up_scales[(row + 1U) * groups + group],
                activation_scale, activation_sum, mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
            gate2 = cq3_arm_accumulate_group128(
                gate2, gate_weights + (row + 2U) * row_bytes + group * 32U,
                gate_scales[(row + 2U) * groups + group],
                activation_scale, activation_sum, mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
            up2 = cq3_arm_accumulate_group128(
                up2, up_weights + (row + 2U) * row_bytes + group * 32U,
                up_scales[(row + 2U) * groups + group],
                activation_scale, activation_sum, mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
            gate3 = cq3_arm_accumulate_group128(
                gate3, gate_weights + (row + 3U) * row_bytes + group * 32U,
                gate_scales[(row + 3U) * groups + group],
                activation_scale, activation_sum, mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
            up3 = cq3_arm_accumulate_group128(
                up3, up_weights + (row + 3U) * row_bytes + group * 32U,
                up_scales[(row + 3U) * groups + group],
                activation_scale, activation_sum, mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
        }
        float gate_value = cq3_arm_reduce_f32(gate0);
        output[row + 0U] =
            (gate_value / (1.0f + expf(-gate_value)))
            * cq3_arm_reduce_f32(up0);
        gate_value = cq3_arm_reduce_f32(gate1);
        output[row + 1U] =
            (gate_value / (1.0f + expf(-gate_value)))
            * cq3_arm_reduce_f32(up1);
        gate_value = cq3_arm_reduce_f32(gate2);
        output[row + 2U] =
            (gate_value / (1.0f + expf(-gate_value)))
            * cq3_arm_reduce_f32(up2);
        gate_value = cq3_arm_reduce_f32(gate3);
        output[row + 3U] =
            (gate_value / (1.0f + expf(-gate_value)))
            * cq3_arm_reduce_f32(up3);
    }
    for (; row < rows; ++row) {
        float32x4_t gate_total = vdupq_n_f32(0.0f);
        float32x4_t up_total = vdupq_n_f32(0.0f);
        for (uint64_t group = 0U; group < groups; ++group) {
            const int8_t *x = activation + group * 128U;
            const int8x16_t x0 = vld1q_s8(x);
            const int8x16_t x1 = vld1q_s8(x + 16U);
            const int8x16_t x2 = vld1q_s8(x + 32U);
            const int8x16_t x3 = vld1q_s8(x + 48U);
            const int8x16_t x4 = vld1q_s8(x + 64U);
            const int8x16_t x5 = vld1q_s8(x + 80U);
            const int8x16_t x6 = vld1q_s8(x + 96U);
            const int8x16_t x7 = vld1q_s8(x + 112U);
            gate_total = cq3_arm_accumulate_group128(
                gate_total, gate_weights + row * row_bytes + group * 32U,
                gate_scales[row * groups + group], activation_scales[group],
                activation_sums[group], mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
            up_total = cq3_arm_accumulate_group128(
                up_total, up_weights + row * row_bytes + group * 32U,
                up_scales[row * groups + group], activation_scales[group],
                activation_sums[group], mask,
                x0, x1, x2, x3, x4, x5, x6, x7
            );
        }
        const float gate_value = cq3_arm_reduce_f32(gate_total);
        output[row] =
            (gate_value / (1.0f + expf(-gate_value)))
            * cq3_arm_reduce_f32(up_total);
    }
    return CQ3_OK;
}
#endif

int cq3_gemv_q8_group128_2bit_swiglu_arm_dotprod(
    const uint8_t *CQ3_RESTRICT gate_weights,
    const uint16_t *CQ3_RESTRICT gate_scales,
    const uint8_t *CQ3_RESTRICT up_weights,
    const uint16_t *CQ3_RESTRICT up_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    int validation = cq3_validate_gemv(
        gate_weights, gate_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation == CQ3_OK) {
        validation = cq3_validate_gemv(
            up_weights, up_scales, activation, activation_scales,
            output, rows, columns, group_size
        );
    }
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_ARM64 && defined(__ARM_FEATURE_DOTPROD)
    if (!cq3_arm_dotprod_available()) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_2bit_swiglu_arm_dotprod_impl(
        gate_weights, gate_scales, up_weights, up_scales,
        activation, activation_scales, activation_sums,
        rows, columns, output
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

int cq3_gemv_q8_group128_radix5_scalar(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = groups * CQ3_RADIX5_BYTES_PER_GROUP;
    for (uint64_t row = 0U; row < rows; ++row) {
        float total = 0.0f;
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint8_t *codes = weights + row * row_bytes + group * CQ3_RADIX5_BYTES_PER_GROUP;
            const int8_t *values = activation + group * CQ3_RADIX5_ACTIVATION_STRIDE;
            int32_t dot = 0;
            for (uint32_t segment = 0U; segment < CQ3_RADIX5_BYTES_PER_GROUP; ++segment) {
                uint32_t packed = codes[segment];
                if (packed > 242U) {
                    return CQ3_ERROR_CODE;
                }
                for (uint32_t lane = 0U; lane < 5U; ++lane) {
                    const int32_t digit = (int32_t)(packed % 3U);
                    packed /= 3U;
                    dot += (digit - 1) * (int32_t)values[segment * 8U + lane];
                }
            }
            const float weight_scale = cq3_half_to_float(weight_scales[row * groups + group]);
            total += (float)dot * weight_scale * activation_scales[group];
        }
        output[row] = total;
    }
    return CQ3_OK;
}

#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,f16c,fma")))
static int cq3_gemv_q8_group128_radix5_avx2_impl(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output,
    int lookup_mode
) {
    uint64_t digit_lookup_local[256];
    const uint64_t *digit_lookup = cq3_radix5_digit_lookup_static;
    if (lookup_mode != 2) {
        for (uint32_t value = 0U; value < 256U; ++value) {
            uint32_t remaining = value;
            uint64_t digits = 0U;
            for (uint32_t lane = 0U; lane < 5U; ++lane) {
                digits |= (uint64_t)(remaining % 3U) << (lane * 8U);
                remaining /= 3U;
            }
            digit_lookup_local[value] = digits;
        }
        digit_lookup = digit_lookup_local;
    }
    const int use_gather = lookup_mode == 1;

    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = groups * CQ3_RADIX5_BYTES_PER_GROUP;
    const __m256i ones = _mm256_set1_epi16(1);
    for (uint64_t row = 0U; row < rows; ++row) {
        float combined_scales[CQ3_MAX_GROUPS_PER_ROW];
        uint64_t scale_group = 0U;
        for (; scale_group + 8U <= groups; scale_group += 8U) {
            const __m128i half_scales = _mm_loadu_si128(
                (const __m128i *)(weight_scales + row * groups + scale_group)
            );
            const __m256 float_scales = _mm256_mul_ps(
                _mm256_cvtph_ps(half_scales),
                _mm256_loadu_ps(activation_scales + scale_group)
            );
            _mm256_storeu_ps(combined_scales + scale_group, float_scales);
        }
        for (; scale_group < groups; ++scale_group) {
            combined_scales[scale_group] =
                _cvtsh_ss(weight_scales[row * groups + scale_group])
                * activation_scales[scale_group];
        }

        __m256 total = _mm256_setzero_ps();
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint8_t *codes = weights + row * row_bytes + group * CQ3_RADIX5_BYTES_PER_GROUP;
            const int8_t *values = activation + group * CQ3_RADIX5_ACTIVATION_STRIDE;
            __m256i accumulator = _mm256_setzero_si256();
            for (uint32_t segment = 0U; segment < 24U; segment += 4U) {
                uint32_t indices;
                memcpy(&indices, codes + segment, sizeof(indices));
                const __m256i digits = use_gather
                    ? _mm256_i32gather_epi64(
                        (const long long int *)digit_lookup,
                        _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)indices)),
                        8
                    )
                    : _mm256_set_epi64x(
                        (long long int)digit_lookup[(indices >> 24U) & 0xffU],
                        (long long int)digit_lookup[(indices >> 16U) & 0xffU],
                        (long long int)digit_lookup[(indices >> 8U) & 0xffU],
                        (long long int)digit_lookup[indices & 0xffU]
                    );
                const __m256i activations = _mm256_loadu_si256(
                    (const __m256i *)(values + segment * 8U)
                );
                accumulator = _mm256_add_epi32(
                    accumulator,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(digits, activations),
                        ones
                    )
                );
            }
            const __m256i tail_digits = use_gather
                ? _mm256_i32gather_epi64(
                    (const long long int *)digit_lookup,
                    _mm_set_epi32(0, 0, codes[25], codes[24]),
                    8
                )
                : _mm256_set_epi64x(
                    0,
                    0,
                    (long long int)digit_lookup[codes[25]],
                    (long long int)digit_lookup[codes[24]]
                );
            const __m256i tail_activations = _mm256_loadu_si256(
                (const __m256i *)(values + 24U * 8U)
            );
            accumulator = _mm256_add_epi32(
                accumulator,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(tail_digits, tail_activations),
                    ones
                )
            );
            const __m256i centered = _mm256_sub_epi32(
                accumulator,
                _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, activation_sums[group])
            );
            total = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(centered),
                _mm256_set1_ps(combined_scales[group]),
                total
            );
        }
        __m128 horizontal = _mm_add_ps(
            _mm256_castps256_ps128(total),
            _mm256_extractf128_ps(total, 1)
        );
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        output[row] = _mm_cvtss_f32(horizontal);
    }
    return CQ3_OK;
}

__attribute__((target("avx2,f16c,fma"), always_inline))
static inline __m256i cq3_radix5_lookup4_nogather(
    const uint64_t *lookup,
    uint32_t indices
) {
    return _mm256_set_epi64x(
        (long long int)lookup[(indices >> 24U) & 0xffU],
        (long long int)lookup[(indices >> 16U) & 0xffU],
        (long long int)lookup[(indices >> 8U) & 0xffU],
        (long long int)lookup[indices & 0xffU]
    );
}

__attribute__((target("avx2,f16c,fma")))
static int cq3_gemv_q8_group128_radix5_pair_avx2_impl(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    uint64_t digit_lookup[256];
    for (uint32_t value = 0U; value < 256U; ++value) {
        uint32_t remaining = value;
        uint64_t digits = 0U;
        for (uint32_t lane = 0U; lane < 5U; ++lane) {
            digits |= (uint64_t)(remaining % 3U) << (lane * 8U);
            remaining /= 3U;
        }
        digit_lookup[value] = digits;
    }

    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = groups * CQ3_RADIX5_BYTES_PER_GROUP;
    const __m256i ones = _mm256_set1_epi16(1);
    uint64_t row = 0U;
    for (; row + 1U < rows; row += 2U) {
        float combined_scales0[CQ3_MAX_GROUPS_PER_ROW];
        float combined_scales1[CQ3_MAX_GROUPS_PER_ROW];
        uint64_t scale_group = 0U;
        for (; scale_group + 8U <= groups; scale_group += 8U) {
            const __m256 activation_scale =
                _mm256_loadu_ps(activation_scales + scale_group);
            const __m128i half_scales0 = _mm_loadu_si128(
                (const __m128i *)(weight_scales + row * groups + scale_group)
            );
            const __m128i half_scales1 = _mm_loadu_si128(
                (const __m128i *)(weight_scales + (row + 1U) * groups + scale_group)
            );
            _mm256_storeu_ps(
                combined_scales0 + scale_group,
                _mm256_mul_ps(_mm256_cvtph_ps(half_scales0), activation_scale)
            );
            _mm256_storeu_ps(
                combined_scales1 + scale_group,
                _mm256_mul_ps(_mm256_cvtph_ps(half_scales1), activation_scale)
            );
        }
        for (; scale_group < groups; ++scale_group) {
            combined_scales0[scale_group] =
                _cvtsh_ss(weight_scales[row * groups + scale_group])
                * activation_scales[scale_group];
            combined_scales1[scale_group] =
                _cvtsh_ss(weight_scales[(row + 1U) * groups + scale_group])
                * activation_scales[scale_group];
        }

        __m256 total0 = _mm256_setzero_ps();
        __m256 total1 = _mm256_setzero_ps();
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint8_t *codes0 =
                weights + row * row_bytes + group * CQ3_RADIX5_BYTES_PER_GROUP;
            const uint8_t *codes1 = codes0 + row_bytes;
            const int8_t *values = activation + group * CQ3_RADIX5_ACTIVATION_STRIDE;
            __m256i accumulator0 = _mm256_setzero_si256();
            __m256i accumulator1 = _mm256_setzero_si256();
            for (uint32_t segment = 0U; segment < 24U; segment += 4U) {
                uint32_t indices0;
                uint32_t indices1;
                memcpy(&indices0, codes0 + segment, sizeof(indices0));
                memcpy(&indices1, codes1 + segment, sizeof(indices1));
                const __m256i activations = _mm256_loadu_si256(
                    (const __m256i *)(values + segment * 8U)
                );
                accumulator0 = _mm256_add_epi32(
                    accumulator0,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            cq3_radix5_lookup4_nogather(digit_lookup, indices0),
                            activations
                        ),
                        ones
                    )
                );
                accumulator1 = _mm256_add_epi32(
                    accumulator1,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            cq3_radix5_lookup4_nogather(digit_lookup, indices1),
                            activations
                        ),
                        ones
                    )
                );
            }
            const __m256i tail_activations = _mm256_loadu_si256(
                (const __m256i *)(values + 24U * 8U)
            );
            const __m256i tail_digits0 = _mm256_set_epi64x(
                0, 0,
                (long long int)digit_lookup[codes0[25]],
                (long long int)digit_lookup[codes0[24]]
            );
            const __m256i tail_digits1 = _mm256_set_epi64x(
                0, 0,
                (long long int)digit_lookup[codes1[25]],
                (long long int)digit_lookup[codes1[24]]
            );
            accumulator0 = _mm256_add_epi32(
                accumulator0,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(tail_digits0, tail_activations), ones
                )
            );
            accumulator1 = _mm256_add_epi32(
                accumulator1,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(tail_digits1, tail_activations), ones
                )
            );
            const __m256i sum = _mm256_set_epi32(
                0, 0, 0, 0, 0, 0, 0, activation_sums[group]
            );
            total0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(accumulator0, sum)),
                _mm256_set1_ps(combined_scales0[group]),
                total0
            );
            total1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(accumulator1, sum)),
                _mm256_set1_ps(combined_scales1[group]),
                total1
            );
        }
        __m128 horizontal0 = _mm_add_ps(
            _mm256_castps256_ps128(total0), _mm256_extractf128_ps(total0, 1)
        );
        __m128 horizontal1 = _mm_add_ps(
            _mm256_castps256_ps128(total1), _mm256_extractf128_ps(total1, 1)
        );
        horizontal0 = _mm_hadd_ps(horizontal0, horizontal0);
        horizontal0 = _mm_hadd_ps(horizontal0, horizontal0);
        horizontal1 = _mm_hadd_ps(horizontal1, horizontal1);
        horizontal1 = _mm_hadd_ps(horizontal1, horizontal1);
        output[row] = _mm_cvtss_f32(horizontal0);
        output[row + 1U] = _mm_cvtss_f32(horizontal1);
    }
    if (row < rows) {
        return cq3_gemv_q8_group128_radix5_avx2_impl(
            weights + row * row_bytes,
            weight_scales + row * groups,
            activation,
            activation_scales,
            activation_sums,
            1U,
            columns,
            group_size,
            output + row,
            0
        );
    }
    return CQ3_OK;
}

__attribute__((target("avx2,f16c,fma")))
static int cq3_gemv_q8_group128_radix5_planes_avx2_impl(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = groups * CQ3_RADIX5_BYTES_PER_GROUP;
    const __m256i divide_by_three = _mm256_set1_epi16(171);
    const __m256i ones = _mm256_set1_epi16(1);
    for (uint64_t row = 0U; row < rows; ++row) {
        float combined_scales[CQ3_MAX_GROUPS_PER_ROW];
        uint64_t scale_group = 0U;
        for (; scale_group + 8U <= groups; scale_group += 8U) {
            const __m128i half_scales = _mm_loadu_si128(
                (const __m128i *)(weight_scales + row * groups + scale_group)
            );
            const __m256 float_scales = _mm256_mul_ps(
                _mm256_cvtph_ps(half_scales),
                _mm256_loadu_ps(activation_scales + scale_group)
            );
            _mm256_storeu_ps(combined_scales + scale_group, float_scales);
        }
        for (; scale_group < groups; ++scale_group) {
            combined_scales[scale_group] =
                _cvtsh_ss(weight_scales[row * groups + scale_group])
                * activation_scales[scale_group];
        }

        __m256 total = _mm256_setzero_ps();
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint8_t *codes =
                weights + row * row_bytes + group * CQ3_RADIX5_BYTES_PER_GROUP;
            const int8_t *values =
                activation + group * CQ3_RADIX5_PLANE_ACTIVATION_STRIDE;
            const __m128i packed_low = _mm_loadu_si128((const __m128i *)codes);
            __m128i packed_high;
            if (group + 1U < groups || row + 1U < rows) {
                packed_high = _mm_loadu_si128((const __m128i *)(codes + 16U));
            } else {
                uint8_t tail[16] = {0U};
                memcpy(tail, codes + 16U, 10U);
                packed_high = _mm_loadu_si128((const __m128i *)tail);
            }
            __m256i remaining_low = _mm256_cvtepu8_epi16(packed_low);
            __m256i remaining_high = _mm256_cvtepu8_epi16(packed_high);
            __m256i products = _mm256_setzero_si256();
            for (uint32_t plane = 0U; plane < CQ3_RADIX5_PLANES; ++plane) {
                const __m256i quotient_low = _mm256_srli_epi16(
                    _mm256_mullo_epi16(remaining_low, divide_by_three), 9
                );
                const __m256i quotient_high = _mm256_srli_epi16(
                    _mm256_mullo_epi16(remaining_high, divide_by_three), 9
                );
                const __m256i digits_low = _mm256_sub_epi16(
                    remaining_low,
                    _mm256_add_epi16(
                        quotient_low, _mm256_slli_epi16(quotient_low, 1)
                    )
                );
                const __m256i digits_high = _mm256_sub_epi16(
                    remaining_high,
                    _mm256_add_epi16(
                        quotient_high, _mm256_slli_epi16(quotient_high, 1)
                    )
                );
                const __m256i digits = _mm256_permute4x64_epi64(
                    _mm256_packus_epi16(digits_low, digits_high), 0xd8
                );
                products = _mm256_add_epi16(
                    products,
                    _mm256_maddubs_epi16(
                        digits,
                        _mm256_loadu_si256(
                            (const __m256i *)(
                                values + plane * CQ3_RADIX5_PLANE_WIDTH
                            )
                        )
                    )
                );
                remaining_low = quotient_low;
                remaining_high = quotient_high;
            }
            const __m256i accumulator = _mm256_madd_epi16(products, ones);
            const __m256i centered = _mm256_sub_epi32(
                accumulator,
                _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, activation_sums[group])
            );
            total = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(centered),
                _mm256_set1_ps(combined_scales[group]),
                total
            );
        }
        __m128 horizontal = _mm_add_ps(
            _mm256_castps256_ps128(total),
            _mm256_extractf128_ps(total, 1)
        );
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        output[row] = _mm_cvtss_f32(horizontal);
    }
    return CQ3_OK;
}
#endif

int cq3_gemv_q8_group128_radix5_avx2(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    const uint32_t required = CQ3_FEATURE_AVX2 | CQ3_FEATURE_F16C | CQ3_FEATURE_FMA;
    if ((cq3_cpu_features() & required) != required) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_radix5_avx2_impl(
        weights, weight_scales, activation, activation_scales,
        activation_sums, rows, columns, group_size, output, 1
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

int cq3_gemv_q8_group128_radix5_avx2_nogather(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    const uint32_t required = CQ3_FEATURE_AVX2 | CQ3_FEATURE_F16C | CQ3_FEATURE_FMA;
    if ((cq3_cpu_features() & required) != required) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_radix5_avx2_impl(
        weights, weight_scales, activation, activation_scales,
        activation_sums, rows, columns, group_size, output, 0
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

int cq3_gemv_q8_group128_radix5_avx2_nogather_static(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    const uint32_t required = CQ3_FEATURE_AVX2 | CQ3_FEATURE_F16C | CQ3_FEATURE_FMA;
    if ((cq3_cpu_features() & required) != required) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_radix5_avx2_impl(
        weights, weight_scales, activation, activation_scales,
        activation_sums, rows, columns, group_size, output, 2
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

int cq3_gemv_q8_group128_radix5_avx2_planes(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    const uint32_t required = CQ3_FEATURE_AVX2 | CQ3_FEATURE_F16C | CQ3_FEATURE_FMA;
    if ((cq3_cpu_features() & required) != required) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_radix5_planes_avx2_impl(
        weights, weight_scales, activation, activation_scales,
        activation_sums, rows, columns, group_size, output
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

int cq3_gemv_q8_group128_radix5_avx2_nogather_pair(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    const uint32_t required = CQ3_FEATURE_AVX2 | CQ3_FEATURE_F16C | CQ3_FEATURE_FMA;
    if ((cq3_cpu_features() & required) != required) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_radix5_pair_avx2_impl(
        weights, weight_scales, activation, activation_scales,
        activation_sums, rows, columns, group_size, output
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

int cq3_sum_u8_scalar(const uint8_t *input, uint64_t count, uint64_t *sum) {
    if (input == NULL || sum == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    uint64_t total = 0U;
    for (uint64_t index = 0U; index < count; ++index) {
        total += input[index];
    }
    *sum = total;
    return CQ3_OK;
}

#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static uint64_t cq3_sum_u8_avx2_impl(const uint8_t *input, uint64_t count) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i accumulator = _mm256_setzero_si256();
    uint64_t index = 0U;
    for (; index + 32U <= count; index += 32U) {
        const __m256i values = _mm256_loadu_si256((const __m256i *)(input + index));
        accumulator = _mm256_add_epi64(accumulator, _mm256_sad_epu8(values, zero));
    }
    uint64_t lanes[4];
    _mm256_storeu_si256((__m256i *)lanes, accumulator);
    uint64_t total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; index < count; ++index) {
        total += input[index];
    }
    return total;
}
#endif

int cq3_sum_u8_avx2(const uint8_t *input, uint64_t count, uint64_t *sum) {
    if (input == NULL || sum == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    if ((cq3_cpu_features() & CQ3_FEATURE_AVX2) == 0U) {
        return CQ3_ERROR_ISA;
    }
    *sum = cq3_sum_u8_avx2_impl(input, count);
    return CQ3_OK;
#else
    (void)count;
    return CQ3_ERROR_ISA;
#endif
}

#if CQ3_ARM64
static uint64_t cq3_sum_u8_neon_impl(const uint8_t *input, uint64_t count) {
    uint64_t total = 0U;
    uint64_t index = 0U;
    while (index + 16U <= count) {
        const uint64_t chunk_stop =
            count - index > (1U << 20U) ? index + (1U << 20U) : count;
        uint32x4_t accumulator = vdupq_n_u32(0U);
        for (; index + 16U <= chunk_stop; index += 16U) {
            const uint8x16_t values = vld1q_u8(input + index);
            accumulator = vpadalq_u16(accumulator, vpaddlq_u8(values));
        }
        total += (uint64_t)vaddvq_u32(accumulator);
    }
    for (; index < count; ++index) {
        total += input[index];
    }
    return total;
}
#endif

int cq3_sum_u8_neon(const uint8_t *input, uint64_t count, uint64_t *sum) {
    if (input == NULL || sum == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
#if CQ3_ARM64
    *sum = cq3_sum_u8_neon_impl(input, count);
    return CQ3_OK;
#else
    (void)count;
    return CQ3_ERROR_ISA;
#endif
}
