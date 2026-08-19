#define _POSIX_C_SOURCE 200809L

#include "qwen38_cq3_exact_lut_cpu.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    WARMUP_PAIRS = 2,
    MEASURED_PAIRS = 21,
    EVICTION_BYTES = 256 * 1024 * 1024,
};

typedef struct {
    uint16_t d[2][8];
    uint8_t qs[8 * 64];
} cq3_x8_tile;

_Static_assert(
    sizeof(cq3_x8_tile) == 8 * sizeof(cq3_exact_lut_source_block),
    "x8 control must remain storage-neutral"
);

typedef struct {
    const char *name;
    int matrices;
    int rows;
    int columns;
} shape_spec;

typedef struct {
    shape_spec shape;
    cq3_exact_lut_source_block *source;
    cq3_x8_tile *x8;
    cq3_exact_lut_tile32 *lut32;
    cq3_exact_lut_activation *activations;
    cq3_exact_lut_table *tables;
    float *ordinary_output;
    float *x8_output;
    float *lut32_output;
    uint8_t *eviction;
} benchmark_state;

typedef void (*benchmark_function)(benchmark_state *state);

static uint64_t splitmix64(uint64_t *state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

static void fill_source_block(
        cq3_exact_lut_source_block *block,
        uint64_t *random_state) {
    static const uint16_t scales[] = {
        UINT16_C(0x2400), UINT16_C(0x2800), UINT16_C(0x2c00),
        UINT16_C(0x3000), UINT16_C(0x3400), UINT16_C(0x3800),
    };
    for (int byte = 0; byte < 64; ++byte) {
        uint8_t packed = 0;
        for (int quarter = 0; quarter < 4; ++quarter) {
            const uint8_t digit = (uint8_t)(splitmix64(random_state) % 3U);
            packed |= (uint8_t)(digit << (2 * quarter));
        }
        block->qs[byte] = packed;
    }
    block->d[0] = scales[splitmix64(random_state) % 6U];
    block->d[1] = scales[splitmix64(random_state) % 6U];
}

static int repack_x8(
        const cq3_exact_lut_source_block *source,
        cq3_x8_tile *destination,
        int rows,
        int columns) {
    if (rows <= 0 || rows % 8 != 0 || columns <= 0 || columns % 256 != 0) {
        return 1;
    }
    const int blocks = columns / 256;
    for (int row = 0; row < rows; row += 8) {
        for (int block = 0; block < blocks; ++block) {
            cq3_x8_tile *tile = destination + (row / 8) * blocks + block;
            for (int group = 0; group < 2; ++group) {
                for (int lane = 0; lane < 8; ++lane) {
                    const cq3_exact_lut_source_block *input =
                        source + (row + lane) * blocks + block;
                    tile->d[group][lane] = input->d[group];
                }
                for (int chunk = 0; chunk < 8; ++chunk) {
                    for (int lane = 0; lane < 8; ++lane) {
                        const cq3_exact_lut_source_block *input =
                            source + (row + lane) * blocks + block;
                        memcpy(
                            tile->qs + group * 256 + chunk * 32 + lane * 4,
                            input->qs + group * 32 + chunk * 4,
                            4
                        );
                    }
                }
            }
        }
    }
    return 0;
}

static float hsum8(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static void ordinary_gemv(
        const cq3_exact_lut_source_block *weights,
        const cq3_exact_lut_activation *activations,
        float *output,
        int rows,
        int columns) {
    const int blocks = columns / 256;
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int row = 0; row < rows; ++row) {
        const cq3_exact_lut_source_block *row_weights = weights + row * blocks;
        __m256 sumf = _mm256_setzero_ps();
        for (int block = 0; block < blocks; ++block) {
            for (int group = 0; group < 2; ++group) {
                const __m256i packed = _mm256_loadu_si256(
                    (const __m256i *)(row_weights[block].qs + group * 32)
                );
                const __m256i digits0 = _mm256_and_si256(packed, mask);
                const __m256i digits1 = _mm256_and_si256(
                    _mm256_srli_epi16(packed, 2), mask
                );
                const __m256i digits2 = _mm256_and_si256(
                    _mm256_srli_epi16(packed, 4), mask
                );
                const __m256i digits3 = _mm256_and_si256(
                    _mm256_srli_epi16(packed, 6), mask
                );
                const cq3_exact_lut_activation *activation =
                    activations + 2 * block + group;
                const __m256i values0 = _mm256_loadu_si256(
                    (const __m256i *)(activation->qs + 0)
                );
                const __m256i values1 = _mm256_loadu_si256(
                    (const __m256i *)(activation->qs + 32)
                );
                const __m256i values2 = _mm256_loadu_si256(
                    (const __m256i *)(activation->qs + 64)
                );
                const __m256i values3 = _mm256_loadu_si256(
                    (const __m256i *)(activation->qs + 96)
                );
                const __m256i products01 = _mm256_add_epi16(
                    _mm256_maddubs_epi16(digits0, values0),
                    _mm256_maddubs_epi16(digits1, values1)
                );
                const __m256i products23 = _mm256_add_epi16(
                    _mm256_maddubs_epi16(digits2, values2),
                    _mm256_maddubs_epi16(digits3, values3)
                );
                __m256i accumulator = _mm256_madd_epi16(
                    _mm256_add_epi16(products01, products23), ones
                );
                accumulator = _mm256_sub_epi32(
                    accumulator,
                    _mm256_set_epi32(
                        0, 0, 0, 0, 0, 0, 0, activation->sum
                    )
                );
                const __m256 scale = _mm256_set1_ps(
                    _cvtsh_ss(row_weights[block].d[group]) * activation->d
                );
                sumf = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(accumulator), scale, sumf
                );
            }
        }
        output[row] = hsum8(sumf);
    }
}

static void x8_gemv(
        const cq3_x8_tile *weights,
        const cq3_exact_lut_activation *activations,
        float *output,
        int rows,
        int columns) {
    const int blocks = columns / 256;
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int row = 0; row < rows; row += 8) {
        const cq3_x8_tile *row_tiles = weights + (row / 8) * blocks;
        __m256 sumf = _mm256_setzero_ps();
        for (int block = 0; block < blocks; ++block) {
            for (int group = 0; group < 2; ++group) {
                const cq3_exact_lut_activation *activation =
                    activations + 2 * block + group;
                __m256i accumulator = _mm256_setzero_si256();
                for (int chunk = 0; chunk < 8; ++chunk) {
                    const __m256i packed = _mm256_loadu_si256(
                        (const __m256i *)(row_tiles[block].qs +
                            group * 256 + chunk * 32)
                    );
                    const __m256i digits0 = _mm256_and_si256(packed, mask);
                    const __m256i digits1 = _mm256_and_si256(
                        _mm256_srli_epi16(packed, 2), mask
                    );
                    const __m256i digits2 = _mm256_and_si256(
                        _mm256_srli_epi16(packed, 4), mask
                    );
                    const __m256i digits3 = _mm256_and_si256(
                        _mm256_srli_epi16(packed, 6), mask
                    );
                    int32_t word0;
                    int32_t word1;
                    int32_t word2;
                    int32_t word3;
                    memcpy(&word0, activation->qs + chunk * 4 + 0, 4);
                    memcpy(&word1, activation->qs + chunk * 4 + 32, 4);
                    memcpy(&word2, activation->qs + chunk * 4 + 64, 4);
                    memcpy(&word3, activation->qs + chunk * 4 + 96, 4);
                    const __m256i values0 = _mm256_set1_epi32(word0);
                    const __m256i values1 = _mm256_set1_epi32(word1);
                    const __m256i values2 = _mm256_set1_epi32(word2);
                    const __m256i values3 = _mm256_set1_epi32(word3);
                    const __m256i products01 = _mm256_add_epi16(
                        _mm256_maddubs_epi16(digits0, values0),
                        _mm256_maddubs_epi16(digits1, values1)
                    );
                    const __m256i products23 = _mm256_add_epi16(
                        _mm256_maddubs_epi16(digits2, values2),
                        _mm256_maddubs_epi16(digits3, values3)
                    );
                    accumulator = _mm256_add_epi32(
                        accumulator,
                        _mm256_madd_epi16(
                            _mm256_add_epi16(products01, products23), ones
                        )
                    );
                }
                accumulator = _mm256_sub_epi32(
                    accumulator, _mm256_set1_epi32(activation->sum)
                );
                const __m256 scales = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        (const __m128i *)row_tiles[block].d[group]
                    )),
                    _mm256_set1_ps(activation->d)
                );
                sumf = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(accumulator), scales, sumf
                );
            }
        }
        _mm256_storeu_ps(output + row, sumf);
    }
}

static void run_ordinary(benchmark_state *state) {
    const size_t source_per_matrix =
        (size_t)state->shape.rows * state->shape.columns / 256;
    for (int matrix = 0; matrix < state->shape.matrices; ++matrix) {
        ordinary_gemv(
            state->source + matrix * source_per_matrix,
            state->activations,
            state->ordinary_output + (size_t)matrix * state->shape.rows,
            state->shape.rows,
            state->shape.columns
        );
    }
}

static void run_x8(benchmark_state *state) {
    const size_t tiles_per_matrix =
        (size_t)(state->shape.rows / 8) * (state->shape.columns / 256);
    for (int matrix = 0; matrix < state->shape.matrices; ++matrix) {
        x8_gemv(
            state->x8 + matrix * tiles_per_matrix,
            state->activations,
            state->x8_output + (size_t)matrix * state->shape.rows,
            state->shape.rows,
            state->shape.columns
        );
    }
}

static void run_lut32(benchmark_state *state) {
    if (cq3_exact_lut_build_tables(
            state->activations, state->tables, state->shape.columns
        ) != CQ3_EXACT_LUT_OK) {
        fprintf(stderr, "v58 LUT construction failed\n");
        abort();
    }
    const size_t tiles_per_matrix =
        (size_t)(state->shape.rows / 32) * (state->shape.columns / 256);
    for (int matrix = 0; matrix < state->shape.matrices; ++matrix) {
        if (cq3_exact_lut_gemv_avx2_trusted(
                state->lut32 + matrix * tiles_per_matrix,
                state->activations,
                state->tables,
                state->lut32_output + (size_t)matrix * state->shape.rows,
                state->shape.rows,
                state->shape.columns
            ) != CQ3_EXACT_LUT_OK) {
            fprintf(stderr, "v58 LUT32 kernel failed\n");
            abort();
        }
    }
}

static void evict_cache(benchmark_state *state) {
    static uint8_t epoch = 73;
    for (size_t offset = 0; offset < EVICTION_BYTES; offset += 64) {
        state->eviction[offset] = (uint8_t)(state->eviction[offset] + epoch);
    }
    ++epoch;
}

static double timed(benchmark_state *state, benchmark_function function) {
    struct timespec start;
    struct timespec stop;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        abort();
    }
    function(state);
    if (clock_gettime(CLOCK_MONOTONIC, &stop) != 0) {
        abort();
    }
    return (double)(stop.tv_sec - start.tv_sec) * 1.0e6 +
        (double)(stop.tv_nsec - start.tv_nsec) / 1.0e3;
}

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median(const double *values, int count) {
    double copy[MEASURED_PAIRS];
    memcpy(copy, values, (size_t)count * sizeof(double));
    qsort(copy, (size_t)count, sizeof(double), compare_double);
    return copy[count / 2];
}

static double mad_fraction(const double *values, int count, double center) {
    double deviations[MEASURED_PAIRS];
    for (int index = 0; index < count; ++index) {
        deviations[index] = fabs(values[index] - center);
    }
    return median(deviations, count) / center;
}

static void paired_ratios(
        benchmark_state *state,
        benchmark_function control,
        benchmark_function candidate,
        double ratios[MEASURED_PAIRS]) {
    for (int pair = -WARMUP_PAIRS; pair < MEASURED_PAIRS; ++pair) {
        evict_cache(state);
        const double a1 = timed(state, control);
        evict_cache(state);
        const double b1 = timed(state, candidate);
        evict_cache(state);
        const double b2 = timed(state, candidate);
        evict_cache(state);
        const double a2 = timed(state, control);
        if (pair >= 0) {
            ratios[pair] = (a1 + a2) / (b1 + b2);
        }
    }
}

static void print_ratios(const double values[MEASURED_PAIRS]) {
    putchar('[');
    for (int index = 0; index < MEASURED_PAIRS; ++index) {
        printf(index == 0 ? "%.9f" : ",%.9f", values[index]);
    }
    putchar(']');
}

static int initialize(benchmark_state *state, shape_spec shape) {
    memset(state, 0, sizeof(*state));
    state->shape = shape;
    const size_t source_blocks_per_matrix =
        (size_t)shape.rows * shape.columns / 256;
    const size_t source_blocks = source_blocks_per_matrix * shape.matrices;
    const size_t x8_tiles =
        (size_t)(shape.rows / 8) * (shape.columns / 256) * shape.matrices;
    const size_t lut32_tiles =
        (size_t)(shape.rows / 32) * (shape.columns / 256) * shape.matrices;
    const size_t activation_groups = (size_t)shape.columns / 128;
    const size_t table_count = activation_groups * 32;
    const size_t outputs = (size_t)shape.matrices * shape.rows;
    state->source = calloc(source_blocks, sizeof(*state->source));
    state->x8 = malloc(x8_tiles * sizeof(*state->x8));
    state->lut32 = malloc(lut32_tiles * sizeof(*state->lut32));
    state->activations = calloc(activation_groups, sizeof(*state->activations));
    state->tables = malloc(table_count * sizeof(*state->tables));
    state->ordinary_output = malloc(outputs * sizeof(float));
    state->x8_output = malloc(outputs * sizeof(float));
    state->lut32_output = malloc(outputs * sizeof(float));
    state->eviction = calloc(EVICTION_BYTES, 1);
    if (state->source == NULL || state->x8 == NULL || state->lut32 == NULL ||
        state->activations == NULL || state->tables == NULL ||
        state->ordinary_output == NULL || state->x8_output == NULL ||
        state->lut32_output == NULL || state->eviction == NULL) {
        return 1;
    }
    uint64_t random_state = UINT64_C(20260819);
    for (size_t block = 0; block < source_blocks; ++block) {
        fill_source_block(state->source + block, &random_state);
    }
    static const float activation_scales[] = {
        0.00390625f, 0.0078125f, 0.015625f, 0.03125f,
    };
    for (size_t group = 0; group < activation_groups; ++group) {
        int sum = 0;
        state->activations[group].d = activation_scales[group & 3U];
        for (int index = 0; index < 128; ++index) {
            const int8_t value = (int8_t)(splitmix64(&random_state) >> 56U);
            state->activations[group].qs[index] = value;
            sum += value;
        }
        state->activations[group].sum = (int16_t)sum;
    }
    for (int matrix = 0; matrix < shape.matrices; ++matrix) {
        if (repack_x8(
                state->source + (size_t)matrix * source_blocks_per_matrix,
                state->x8 + (size_t)matrix * x8_tiles / shape.matrices,
                shape.rows,
                shape.columns
            ) != 0 ||
            cq3_exact_lut_repack(
                state->source + (size_t)matrix * source_blocks_per_matrix,
                state->lut32 + (size_t)matrix * lut32_tiles / shape.matrices,
                shape.rows,
                shape.columns
            ) != CQ3_EXACT_LUT_OK) {
            return 1;
        }
    }
    return 0;
}

static void destroy(benchmark_state *state) {
    free(state->source);
    free(state->x8);
    free(state->lut32);
    free(state->activations);
    free(state->tables);
    free(state->ordinary_output);
    free(state->x8_output);
    free(state->lut32_output);
    free(state->eviction);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--shape") != 0) {
        fprintf(stderr, "usage: %s --shape gate_up|down\n", argv[0]);
        return 2;
    }
    shape_spec shape;
    if (strcmp(argv[2], "gate_up") == 0) {
        shape = (shape_spec){"gate_up", 2, 17408, 5120};
    } else if (strcmp(argv[2], "down") == 0) {
        shape = (shape_spec){"down", 1, 5120, 17408};
    } else {
        fprintf(stderr, "unknown v58 shape\n");
        return 2;
    }
    benchmark_state state;
    if (initialize(&state, shape) != 0) {
        fprintf(stderr, "v58 benchmark initialization failed\n");
        destroy(&state);
        return 2;
    }
    run_ordinary(&state);
    run_x8(&state);
    run_lut32(&state);
    const size_t output_count = (size_t)shape.matrices * shape.rows;
    if (memcmp(state.x8_output, state.lut32_output,
               output_count * sizeof(float)) != 0) {
        fprintf(stderr, "v58 full-shape LUT32 output differs from x8 bits\n");
        destroy(&state);
        return 2;
    }
    float maximum_scaled_delta = 0.0f;
    for (size_t index = 0; index < output_count; ++index) {
        const float delta = fabsf(
            state.ordinary_output[index] - state.lut32_output[index]
        ) / fmaxf(1.0f, fabsf(state.ordinary_output[index]));
        maximum_scaled_delta = fmaxf(maximum_scaled_delta, delta);
    }
    if (maximum_scaled_delta > 0.00005f) {
        fprintf(stderr, "v58 ordinary numerical bound failed: %.9g\n",
                maximum_scaled_delta);
        destroy(&state);
        return 2;
    }

    double x8_ratios[MEASURED_PAIRS];
    double ordinary_ratios[MEASURED_PAIRS];
    paired_ratios(&state, run_x8, run_lut32, x8_ratios);
    paired_ratios(&state, run_ordinary, run_lut32, ordinary_ratios);
    const double x8_median = median(x8_ratios, MEASURED_PAIRS);
    const double ordinary_median = median(ordinary_ratios, MEASURED_PAIRS);
    printf(
        "{\"schema\":\"qwen38_cq3_exact_lut_native_benchmark_v58\","
        "\"shape\":\"%s\",\"matrices\":%d,\"rows\":%d,"
        "\"columns\":%d,\"warmup_pairs\":%d,\"measured_pairs\":%d,"
        "\"cache_eviction_bytes\":%d,\"x8_lut32_bit_exact\":true,"
        "\"maximum_scaled_delta_vs_ordinary\":%.9g,"
        "\"source_bytes\":%zu,\"x8_bytes\":%zu,\"lut32_bytes\":%zu,"
        "\"temporary_lut_bytes\":%zu,\"x8_over_lut32_ratios\":",
        shape.name,
        shape.matrices,
        shape.rows,
        shape.columns,
        WARMUP_PAIRS,
        MEASURED_PAIRS,
        EVICTION_BYTES,
        maximum_scaled_delta,
        output_count * (size_t)shape.columns / 256U *
            sizeof(cq3_exact_lut_source_block),
        output_count / 8U * (size_t)shape.columns / 256U * sizeof(cq3_x8_tile),
        output_count / 32U * (size_t)shape.columns / 256U *
            sizeof(cq3_exact_lut_tile32),
        cq3_exact_lut_temporary_bytes(shape.columns)
    );
    print_ratios(x8_ratios);
    printf(
        ",\"ordinary_over_lut32_ratios\":"
    );
    print_ratios(ordinary_ratios);
    printf(
        ",\"x8_over_lut32_median\":%.9f,"
        "\"ordinary_over_lut32_median\":%.9f,"
        "\"x8_ratio_mad_fraction\":%.9f,"
        "\"ordinary_ratio_mad_fraction\":%.9f,"
        "\"complete\":true,\"terminal\":true}\n",
        x8_median,
        ordinary_median,
        mad_fraction(x8_ratios, MEASURED_PAIRS, x8_median),
        mad_fraction(ordinary_ratios, MEASURED_PAIRS, ordinary_median)
    );
    destroy(&state);
    return 0;
}
