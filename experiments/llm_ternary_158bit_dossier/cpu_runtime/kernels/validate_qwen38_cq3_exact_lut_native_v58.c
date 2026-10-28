#include "qwen38_cq3_exact_lut_cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    ROWS = 32,
    COLUMNS = 256,
    GROUPS = 2,
    RANDOM_TRIALS = 512,
    ADVERSARIAL_TRIALS = 4,
    RESERVED_CHECKS = 8,
};

static uint64_t splitmix64(uint64_t *state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

static void set_digit(
        cq3_exact_lut_source_block *block,
        int group,
        int position,
        uint8_t digit) {
    const int lane = position & 31;
    const int quarter = position >> 5;
    const int shift = 2 * quarter;
    block->qs[group * 32 + lane] = (uint8_t)(
        (block->qs[group * 32 + lane] & ~(3U << shift)) |
        ((digit & 3U) << shift)
    );
}

static uint8_t get_digit(
        const cq3_exact_lut_source_block *block,
        int group,
        int position) {
    const int lane = position & 31;
    const int quarter = position >> 5;
    return (uint8_t)((block->qs[group * 32 + lane] >> (2 * quarter)) & 3U);
}

static void fill_random_case(
        cq3_exact_lut_source_block source[ROWS],
        cq3_exact_lut_activation activations[GROUPS],
        uint64_t *random_state) {
    static const uint16_t half_scales[] = {
        UINT16_C(0x2800), UINT16_C(0x2c00), UINT16_C(0x3000),
        UINT16_C(0x3400), UINT16_C(0x3800), UINT16_C(0x3c00),
    };
    static const float activation_scales[] = {
        0.00390625f, 0.0078125f, 0.015625f, 0.03125f,
    };
    memset(source, 0, ROWS * sizeof(*source));
    memset(activations, 0, GROUPS * sizeof(*activations));
    for (int row = 0; row < ROWS; ++row) {
        for (int group = 0; group < GROUPS; ++group) {
            source[row].d[group] =
                half_scales[splitmix64(random_state) % 6U];
            for (int position = 0; position < CQ3_EXACT_LUT_GROUP; ++position) {
                set_digit(
                    source + row,
                    group,
                    position,
                    (uint8_t)(splitmix64(random_state) % 3U)
                );
            }
        }
    }
    for (int group = 0; group < GROUPS; ++group) {
        int sum = 0;
        activations[group].d =
            activation_scales[splitmix64(random_state) & 3U];
        for (int position = 0; position < CQ3_EXACT_LUT_GROUP; ++position) {
            const int8_t value = (int8_t)(splitmix64(random_state) >> 56U);
            activations[group].qs[position] = value;
            sum += value;
        }
        activations[group].sum = (int16_t)sum;
    }
}

static void fill_adversarial_case(
        cq3_exact_lut_source_block source[ROWS],
        cq3_exact_lut_activation activations[GROUPS],
        int trial) {
    memset(source, 0, ROWS * sizeof(*source));
    memset(activations, 0, GROUPS * sizeof(*activations));
    for (int row = 0; row < ROWS; ++row) {
        source[row].d[0] = UINT16_C(0x3800);
        source[row].d[1] = UINT16_C(0x3c00);
        for (int group = 0; group < GROUPS; ++group) {
            for (int position = 0; position < CQ3_EXACT_LUT_GROUP; ++position) {
                const uint8_t digit = (uint8_t)(
                    trial == 0 ? 2 :
                    trial == 1 ? 0 :
                    trial == 2 ? position % 3 :
                    (row + group + position) % 3
                );
                set_digit(source + row, group, position, digit);
            }
        }
    }
    for (int group = 0; group < GROUPS; ++group) {
        int sum = 0;
        activations[group].d = group == 0 ? 0.25f : 0.5f;
        for (int position = 0; position < CQ3_EXACT_LUT_GROUP; ++position) {
            const int8_t value = (int8_t)(
                trial == 0 ? 127 :
                trial == 1 ? -128 :
                trial == 2 ? (position & 1 ? 127 : -128) :
                ((position + group) & 1 ? -127 : 126)
            );
            activations[group].qs[position] = value;
            sum += value;
        }
        activations[group].sum = (int16_t)sum;
    }
}

static float half_to_float(uint16_t value) {
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
        bits = sign | UINT32_C(0x7f800000) | fraction << 13U;
    } else {
        bits = sign | (exponent + 112U) << 23U | fraction << 13U;
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int validate_case(
        const cq3_exact_lut_source_block source[ROWS],
        const cq3_exact_lut_activation activations[GROUPS]) {
    cq3_exact_lut_tile32 compute_image;
    cq3_exact_lut_table tables[GROUPS * CQ3_EXACT_LUT_QUARTETS];
    float direct[ROWS] = {0.0f};
    float scalar[ROWS] = {0.0f};
    float avx2[ROWS] = {0.0f};
    for (int row = 0; row < ROWS; ++row) {
        for (int group = 0; group < GROUPS; ++group) {
            int dot = 0;
            for (int position = 0; position < CQ3_EXACT_LUT_GROUP; ++position) {
                const int code = (int)get_digit(source + row, group, position) - 1;
                dot += code * (int)activations[group].qs[position];
            }
            direct[row] += (float)dot * half_to_float(source[row].d[group]) *
                activations[group].d;
        }
    }
    if (
        cq3_exact_lut_repack(source, &compute_image, ROWS, COLUMNS)
            != CQ3_EXACT_LUT_OK ||
        cq3_exact_lut_build_tables(activations, tables, COLUMNS)
            != CQ3_EXACT_LUT_OK ||
        cq3_exact_lut_gemv_scalar_trusted(
            &compute_image, activations, tables, scalar, ROWS, COLUMNS
        ) != CQ3_EXACT_LUT_OK ||
        cq3_exact_lut_gemv_avx2_trusted(
            &compute_image, activations, tables, avx2, ROWS, COLUMNS
        ) != CQ3_EXACT_LUT_OK ||
        memcmp(direct, scalar, sizeof(direct)) != 0 ||
        memcmp(scalar, avx2, sizeof(scalar)) != 0
    ) {
        return 1;
    }
    return 0;
}

int main(void) {
    cq3_exact_lut_source_block source[ROWS];
    cq3_exact_lut_activation activations[GROUPS];
    uint64_t random_state = UINT64_C(20260819);
    for (int trial = 0; trial < RANDOM_TRIALS; ++trial) {
        fill_random_case(source, activations, &random_state);
        if (validate_case(source, activations) != 0) {
            fprintf(stderr, "v58 randomized exactness failed at trial %d\n", trial);
            return 2;
        }
    }
    for (int trial = 0; trial < ADVERSARIAL_TRIALS; ++trial) {
        fill_adversarial_case(source, activations, trial);
        if (validate_case(source, activations) != 0) {
            fprintf(stderr, "v58 adversarial exactness failed at trial %d\n", trial);
            return 2;
        }
    }
    memset(source, 0, sizeof(source));
    cq3_exact_lut_tile32 compute_image;
    for (int check = 0; check < RESERVED_CHECKS; ++check) {
        set_digit(
            source + (check * 5) % ROWS,
            check & 1,
            (check * 17) % CQ3_EXACT_LUT_GROUP,
            3
        );
        if (cq3_exact_lut_repack(source, &compute_image, ROWS, COLUMNS)
            != CQ3_EXACT_LUT_ERROR_CODE) {
            fprintf(stderr, "v58 reserved-code rejection failed at check %d\n", check);
            return 2;
        }
        memset(source, 0, sizeof(source));
    }
    printf(
        "{\"schema\":\"qwen38_cq3_exact_lut_native_validation_v58\","
        "\"random_groups\":%d,\"adversarial_groups\":%d,"
        "\"reserved_code_checks\":%d,\"direct_scalar_bit_exact\":true,"
        "\"scalar_avx2_bit_exact\":true,\"storage_neutral\":true,"
        "\"complete\":true,\"terminal\":true}\n",
        RANDOM_TRIALS * GROUPS,
        ADVERSARIAL_TRIALS * GROUPS,
        RESERVED_CHECKS
    );
    return 0;
}
