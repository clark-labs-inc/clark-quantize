#include "qwen38_cq3_exact_lut_cpu.h"

#include <math.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define CQ3_EXACT_LUT_X86 1
#include <immintrin.h>
#else
#define CQ3_EXACT_LUT_X86 0
#endif

_Static_assert(
    sizeof(cq3_exact_lut_source_block) == 68,
    "CQ3 source block layout changed"
);
_Static_assert(
    sizeof(cq3_exact_lut_tile32) ==
        CQ3_EXACT_LUT_TILE_ROWS * sizeof(cq3_exact_lut_source_block),
    "exact LUT compute image must remain storage-neutral"
);
_Static_assert(
    sizeof(cq3_exact_lut_table) == 32,
    "exact signed-INT16 subset table layout changed"
);

static int cq3_exact_lut_shape_valid(int rows, int columns) {
    return rows > 0 && rows % CQ3_EXACT_LUT_TILE_ROWS == 0 &&
        columns > 0 && columns % CQ3_EXACT_LUT_BLOCK == 0;
}

size_t cq3_exact_lut_compute_image_bytes(int rows, int columns) {
    if (!cq3_exact_lut_shape_valid(rows, columns)) {
        return 0;
    }
    return (size_t)(rows / CQ3_EXACT_LUT_TILE_ROWS) *
        (size_t)(columns / CQ3_EXACT_LUT_BLOCK) *
        sizeof(cq3_exact_lut_tile32);
}

size_t cq3_exact_lut_temporary_bytes(int columns) {
    if (columns <= 0 || columns % CQ3_EXACT_LUT_GROUP != 0) {
        return 0;
    }
    return (size_t)(columns / CQ3_EXACT_LUT_GROUP) *
        CQ3_EXACT_LUT_QUARTETS * sizeof(cq3_exact_lut_table);
}

static uint8_t cq3_exact_lut_digit(
        const cq3_exact_lut_source_block *block,
        int group,
        int position) {
    const int lane = position & 31;
    const int quarter = position >> 5;
    return (uint8_t)((block->qs[group * 32 + lane] >> (2 * quarter)) & 3U);
}

int cq3_exact_lut_repack(
        const cq3_exact_lut_source_block *source,
        cq3_exact_lut_tile32 *destination,
        int rows,
        int columns) {
    if (source == NULL || destination == NULL) {
        return CQ3_EXACT_LUT_ERROR_ARGUMENT;
    }
    if (!cq3_exact_lut_shape_valid(rows, columns)) {
        return CQ3_EXACT_LUT_ERROR_SHAPE;
    }
    const int blocks_per_row = columns / CQ3_EXACT_LUT_BLOCK;
    for (int row = 0; row < rows; row += CQ3_EXACT_LUT_TILE_ROWS) {
        for (int block = 0; block < blocks_per_row; ++block) {
            cq3_exact_lut_tile32 *tile = destination +
                (row / CQ3_EXACT_LUT_TILE_ROWS) * blocks_per_row + block;
            for (int group = 0; group < 2; ++group) {
                for (int tile_row = 0; tile_row < CQ3_EXACT_LUT_TILE_ROWS;
                     ++tile_row) {
                    const cq3_exact_lut_source_block *input = source +
                        (row + tile_row) * blocks_per_row + block;
                    tile->d[group][tile_row] = input->d[group];
                }
                for (int quartet = 0; quartet < CQ3_EXACT_LUT_QUARTETS;
                     ++quartet) {
                    for (int tile_row = 0; tile_row < CQ3_EXACT_LUT_TILE_ROWS;
                         ++tile_row) {
                        const cq3_exact_lut_source_block *input = source +
                            (row + tile_row) * blocks_per_row + block;
                        uint8_t bit0 = 0;
                        uint8_t bit1 = 0;
                        for (int index = 0; index < 4; ++index) {
                            const uint8_t digit = cq3_exact_lut_digit(
                                input, group, quartet * 4 + index
                            );
                            if (digit == 3U) {
                                return CQ3_EXACT_LUT_ERROR_CODE;
                            }
                            bit0 |= (uint8_t)((digit & 1U) << index);
                            bit1 |= (uint8_t)(((digit >> 1U) & 1U) << index);
                        }
                        tile->masks[group][quartet][tile_row] =
                            (uint8_t)(bit0 | (uint8_t)(bit1 << 4U));
                    }
                }
            }
        }
    }
    return CQ3_EXACT_LUT_OK;
}

int cq3_exact_lut_build_tables(
        const cq3_exact_lut_activation *activations,
        cq3_exact_lut_table *tables,
        int columns) {
    if (activations == NULL || tables == NULL) {
        return CQ3_EXACT_LUT_ERROR_ARGUMENT;
    }
    if (columns <= 0 || columns % CQ3_EXACT_LUT_GROUP != 0) {
        return CQ3_EXACT_LUT_ERROR_SHAPE;
    }
    const int groups = columns / CQ3_EXACT_LUT_GROUP;
    for (int group = 0; group < groups; ++group) {
        if (!isfinite(activations[group].d)) {
            return CQ3_EXACT_LUT_ERROR_ARGUMENT;
        }
        for (int quartet = 0; quartet < CQ3_EXACT_LUT_QUARTETS; ++quartet) {
            int16_t sums[CQ3_EXACT_LUT_SUBSETS] = {0};
            const int8_t *values = activations[group].qs + quartet * 4;
            for (int mask = 1; mask < CQ3_EXACT_LUT_SUBSETS; ++mask) {
                int sum = 0;
                for (int index = 0; index < 4; ++index) {
                    if ((mask >> index) & 1) {
                        sum += values[index];
                    }
                }
                sums[mask] = (int16_t)sum;
            }
            cq3_exact_lut_table *table = tables +
                group * CQ3_EXACT_LUT_QUARTETS + quartet;
            for (int mask = 0; mask < CQ3_EXACT_LUT_SUBSETS; ++mask) {
                const uint16_t bits = (uint16_t)sums[mask];
                table->low[mask] = (uint8_t)(bits & 0xffU);
                table->high[mask] = (uint8_t)(bits >> 8U);
            }
        }
    }
    return CQ3_EXACT_LUT_OK;
}

static int16_t cq3_exact_lut_read(
        const cq3_exact_lut_table *table,
        uint8_t index) {
    const uint16_t bits = (uint16_t)table->low[index] |
        (uint16_t)((uint16_t)table->high[index] << 8U);
    int16_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float cq3_exact_lut_half_to_float(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000U) << 16U;
    const uint32_t exponent = (value >> 10U) & 0x1fU;
    const uint32_t fraction = value & 0x03ffU;
    uint32_t bits;
    if (exponent == 0U) {
        if (fraction == 0U) {
            bits = sign;
        } else {
            uint32_t normalized = fraction;
            int shift = 0;
            while ((normalized & 0x0400U) == 0U) {
                normalized <<= 1U;
                ++shift;
            }
            normalized &= 0x03ffU;
            bits = sign | (uint32_t)(113 - shift) << 23U | normalized << 13U;
        }
    } else if (exponent == 31U) {
        bits = sign | 0x7f800000U | fraction << 13U;
    } else {
        bits = sign | (exponent + 112U) << 23U | fraction << 13U;
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int cq3_exact_lut_gemv_scalar_trusted(
        const cq3_exact_lut_tile32 *weights,
        const cq3_exact_lut_activation *activations,
        const cq3_exact_lut_table *tables,
        float *output,
        int rows,
        int columns) {
    if (weights == NULL || activations == NULL || tables == NULL || output == NULL) {
        return CQ3_EXACT_LUT_ERROR_ARGUMENT;
    }
    if (!cq3_exact_lut_shape_valid(rows, columns)) {
        return CQ3_EXACT_LUT_ERROR_SHAPE;
    }
    const int blocks = columns / CQ3_EXACT_LUT_BLOCK;
    for (int row = 0; row < rows; row += CQ3_EXACT_LUT_TILE_ROWS) {
        float sums[CQ3_EXACT_LUT_TILE_ROWS] = {0.0f};
        for (int block = 0; block < blocks; ++block) {
            const cq3_exact_lut_tile32 *tile = weights +
                (row / CQ3_EXACT_LUT_TILE_ROWS) * blocks + block;
            for (int group = 0; group < 2; ++group) {
                const int activation_group = 2 * block + group;
                for (int tile_row = 0; tile_row < CQ3_EXACT_LUT_TILE_ROWS;
                     ++tile_row) {
                    int32_t dot = 0;
                    for (int quartet = 0; quartet < CQ3_EXACT_LUT_QUARTETS;
                         ++quartet) {
                        const uint8_t masks =
                            tile->masks[group][quartet][tile_row];
                        const cq3_exact_lut_table *table = tables +
                            activation_group * CQ3_EXACT_LUT_QUARTETS + quartet;
                        dot += cq3_exact_lut_read(table, masks & 15U) +
                            2 * cq3_exact_lut_read(table, masks >> 4U);
                    }
                    dot -= activations[activation_group].sum;
                    const float scale =
                        cq3_exact_lut_half_to_float(
                            tile->d[group][tile_row]
                        ) *
                        activations[activation_group].d;
                    sums[tile_row] += (float)dot * scale;
                }
            }
        }
        memcpy(output + row, sums, sizeof(sums));
    }
    return CQ3_EXACT_LUT_OK;
}

#if CQ3_EXACT_LUT_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma,f16c")))
static int cq3_exact_lut_gemv_avx2_impl(
        const cq3_exact_lut_tile32 *weights,
        const cq3_exact_lut_activation *activations,
        const cq3_exact_lut_table *tables,
        float *output,
        int rows,
        int columns) {
    const int blocks = columns / CQ3_EXACT_LUT_BLOCK;
    const __m256i nibble_mask = _mm256_set1_epi8(15);
    for (int row = 0; row < rows; row += CQ3_EXACT_LUT_TILE_ROWS) {
        __m256 sums0 = _mm256_setzero_ps();
        __m256 sums1 = _mm256_setzero_ps();
        __m256 sums2 = _mm256_setzero_ps();
        __m256 sums3 = _mm256_setzero_ps();
        for (int block = 0; block < blocks; ++block) {
            const cq3_exact_lut_tile32 *tile = weights +
                (row / CQ3_EXACT_LUT_TILE_ROWS) * blocks + block;
            for (int group = 0; group < 2; ++group) {
                const int activation_group = 2 * block + group;
                __m256i dots_low = _mm256_setzero_si256();
                __m256i dots_high = _mm256_setzero_si256();
                for (int quartet = 0; quartet < CQ3_EXACT_LUT_QUARTETS;
                     ++quartet) {
                    const __m256i masks = _mm256_loadu_si256(
                        (const __m256i *)tile->masks[group][quartet]
                    );
                    const __m256i bit0 = _mm256_and_si256(masks, nibble_mask);
                    const __m256i bit1 = _mm256_and_si256(
                        _mm256_srli_epi16(masks, 4), nibble_mask
                    );
                    const cq3_exact_lut_table *table = tables +
                        activation_group * CQ3_EXACT_LUT_QUARTETS + quartet;
                    const __m256i lows = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128((const __m128i *)table->low)
                    );
                    const __m256i highs = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128((const __m128i *)table->high)
                    );
                    const __m256i bit0_low = _mm256_shuffle_epi8(lows, bit0);
                    const __m256i bit0_high = _mm256_shuffle_epi8(highs, bit0);
                    const __m256i bit1_low = _mm256_shuffle_epi8(lows, bit1);
                    const __m256i bit1_high = _mm256_shuffle_epi8(highs, bit1);
                    const __m256i value0_low =
                        _mm256_unpacklo_epi8(bit0_low, bit0_high);
                    const __m256i value0_high =
                        _mm256_unpackhi_epi8(bit0_low, bit0_high);
                    const __m256i value1_low =
                        _mm256_unpacklo_epi8(bit1_low, bit1_high);
                    const __m256i value1_high =
                        _mm256_unpackhi_epi8(bit1_low, bit1_high);
                    dots_low = _mm256_add_epi16(
                        dots_low,
                        _mm256_add_epi16(
                            value0_low, _mm256_slli_epi16(value1_low, 1)
                        )
                    );
                    dots_high = _mm256_add_epi16(
                        dots_high,
                        _mm256_add_epi16(
                            value0_high, _mm256_slli_epi16(value1_high, 1)
                        )
                    );
                }
                const __m256i activation_sum = _mm256_set1_epi16(
                    activations[activation_group].sum
                );
                dots_low = _mm256_sub_epi16(dots_low, activation_sum);
                dots_high = _mm256_sub_epi16(dots_high, activation_sum);
                const __m256 dot0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm256_castsi256_si128(dots_low)
                ));
                const __m256 dot1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm256_castsi256_si128(dots_high)
                ));
                const __m256 dot2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm256_extracti128_si256(dots_low, 1)
                ));
                const __m256 dot3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm256_extracti128_si256(dots_high, 1)
                ));
                const float activation_scale = activations[activation_group].d;
                const __m256 scale0 = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        (const __m128i *)(tile->d[group] + 0)
                    )),
                    _mm256_set1_ps(activation_scale)
                );
                const __m256 scale1 = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        (const __m128i *)(tile->d[group] + 8)
                    )),
                    _mm256_set1_ps(activation_scale)
                );
                const __m256 scale2 = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        (const __m128i *)(tile->d[group] + 16)
                    )),
                    _mm256_set1_ps(activation_scale)
                );
                const __m256 scale3 = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        (const __m128i *)(tile->d[group] + 24)
                    )),
                    _mm256_set1_ps(activation_scale)
                );
                sums0 = _mm256_fmadd_ps(dot0, scale0, sums0);
                sums1 = _mm256_fmadd_ps(dot1, scale1, sums1);
                sums2 = _mm256_fmadd_ps(dot2, scale2, sums2);
                sums3 = _mm256_fmadd_ps(dot3, scale3, sums3);
            }
        }
        _mm256_storeu_ps(output + row + 0, sums0);
        _mm256_storeu_ps(output + row + 8, sums1);
        _mm256_storeu_ps(output + row + 16, sums2);
        _mm256_storeu_ps(output + row + 24, sums3);
    }
    return CQ3_EXACT_LUT_OK;
}
#endif

int cq3_exact_lut_gemv_avx2_trusted(
        const cq3_exact_lut_tile32 *weights,
        const cq3_exact_lut_activation *activations,
        const cq3_exact_lut_table *tables,
        float *output,
        int rows,
        int columns) {
    if (weights == NULL || activations == NULL || tables == NULL || output == NULL) {
        return CQ3_EXACT_LUT_ERROR_ARGUMENT;
    }
    if (!cq3_exact_lut_shape_valid(rows, columns)) {
        return CQ3_EXACT_LUT_ERROR_SHAPE;
    }
#if CQ3_EXACT_LUT_X86 && (defined(__GNUC__) || defined(__clang__))
    return cq3_exact_lut_gemv_avx2_impl(
        weights, activations, tables, output, rows, columns
    );
#else
    return CQ3_EXACT_LUT_ERROR_ISA;
#endif
}
