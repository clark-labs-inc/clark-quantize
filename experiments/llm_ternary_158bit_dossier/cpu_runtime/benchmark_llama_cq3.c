#define _POSIX_C_SOURCE 200809L

#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

struct measurement {
    const char * name;
    const char * cache_mode;
    double median_seconds;
    double best_seconds;
    double weight_gbps;
    double checksum;
};

static volatile uint64_t eviction_sink;

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        abort();
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9;
}

static int compare_double(const void * lhs, const void * rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static void fill_codes(uint8_t * qs, size_t bytes, uint64_t seed) {
    for (size_t j = 0; j < bytes; ++j) {
        uint8_t packed = 0;
        for (int quarter = 0; quarter < 4; ++quarter) {
            const uint64_t mixed = seed + 17u*j + 29u*(uint64_t)quarter;
            packed |= (uint8_t)(mixed % 3u) << (2*quarter);
        }
        qs[j] = packed;
    }
}

static void evict_last_level_cache(const uint8_t * data, size_t bytes) {
    uint64_t sum = 0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t offset = 0; offset < bytes; offset += 64) {
        sum += data[offset];
    }
    eviction_sink += sum;
}

static struct measurement benchmark(
        const char * name,
        dot_fn fn,
        const void * weights,
        size_t row_bytes,
        const void * activation,
        int rows,
        int columns,
        int iterations,
        int trials,
        float * output,
        const uint8_t * eviction,
        size_t eviction_bytes) {
    double * samples = calloc((size_t)trials, sizeof(double));
    if (samples == NULL) {
        abort();
    }

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        fn(columns, &output[row], 0, (const uint8_t *)weights + (size_t)row*row_bytes,
           0, activation, 0, 1);
    }

    double checksum = 0.0;
    for (int trial = 0; trial < trials; ++trial) {
        double elapsed = 0.0;
        if (eviction == NULL) {
            const double start = now_seconds();
            for (int iteration = 0; iteration < iterations; ++iteration) {
                #pragma omp parallel for schedule(static)
                for (int row = 0; row < rows; ++row) {
                    fn(columns, &output[row], 0, (const uint8_t *)weights + (size_t)row*row_bytes,
                       0, activation, 0, 1);
                }
            }
            elapsed = now_seconds() - start;
        } else {
            for (int iteration = 0; iteration < iterations; ++iteration) {
                evict_last_level_cache(eviction, eviction_bytes);
                const double start = now_seconds();
                #pragma omp parallel for schedule(static)
                for (int row = 0; row < rows; ++row) {
                    fn(columns, &output[row], 0, (const uint8_t *)weights + (size_t)row*row_bytes,
                       0, activation, 0, 1);
                }
                elapsed += now_seconds() - start;
            }
        }
        samples[trial] = elapsed/(double)iterations;
        for (int row = 0; row < rows; ++row) {
            checksum += output[row]*(double)(row + 1);
        }
    }
    qsort(samples, (size_t)trials, sizeof(double), compare_double);
    const double median = samples[trials/2];
    const double best = samples[0];
    const double gbps = ((double)row_bytes*(double)rows)/median/1e9;
    free(samples);
    return (struct measurement) {
        name, eviction == NULL ? "hot" : "evicted", median, best, gbps, checksum
    };
}

int main(int argc, char ** argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 17408;
    const int columns = argc > 2 ? atoi(argv[2]) : 5120;
    const int iterations = argc > 3 ? atoi(argv[3]) : 20;
    const int trials = argc > 4 ? atoi(argv[4]) : 7;
    const int threads = argc > 5 ? atoi(argv[5]) : 4;
    if (rows <= 0 || columns <= 0 || columns % QK_CQ3_0 != 0 ||
        iterations <= 0 || trials <= 0 || (trials % 2) == 0 || threads <= 0) {
        fprintf(stderr, "usage: %s [rows] [columns_multiple_of_256] [iterations] [odd_trials] [threads]\n", argv[0]);
        return 2;
    }

    ggml_cpu_init();
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    const int blocks = columns/QK_CQ3_0;
    const size_t cq3_row_bytes = (size_t)blocks*sizeof(block_cq3_0);
    const size_t tq2_row_bytes = (size_t)blocks*sizeof(block_tq2_0);
    block_cq3_0 * cq3 = aligned_alloc(64, cq3_row_bytes*(size_t)rows);
    block_tq2_0 * tq2 = aligned_alloc(64, tq2_row_bytes*(size_t)rows);
    float * input = aligned_alloc(64, (size_t)columns*sizeof(float));
    block_cq8_0 * cq8 = aligned_alloc(64, (size_t)(columns/QK_CQ8_0)*sizeof(block_cq8_0));
    block_q8_K * q8k = aligned_alloc(64, (size_t)blocks*sizeof(block_q8_K));
    float * output = aligned_alloc(64, (size_t)rows*sizeof(float));
    float * reference = aligned_alloc(64, (size_t)rows*sizeof(float));
    const size_t eviction_bytes = 128u*1024u*1024u;
    uint8_t * eviction = aligned_alloc(64, eviction_bytes);
    if (cq3 == NULL || tq2 == NULL || input == NULL || cq8 == NULL || q8k == NULL ||
        output == NULL || reference == NULL || eviction == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 3;
    }

    for (int j = 0; j < columns; ++j) {
        input[j] = 2.25f*sinf(0.017f*(float)j) + 0.375f*cosf(0.071f*(float)j);
    }
    quantize_row_cq8_0_ref(input, cq8, columns);
    quantize_row_q8_K_ref(input, q8k, columns);
    for (size_t j = 0; j < eviction_bytes; ++j) {
        eviction[j] = (uint8_t)(j*37u + 11u);
    }

    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks; ++block) {
            const uint64_t seed = (uint64_t)row*1315423911u + (uint64_t)block*2654435761u;
            fill_codes(cq3[(size_t)row*blocks + block].qs, sizeof(cq3[0].qs), seed);
            fill_codes(tq2[(size_t)row*blocks + block].qs, sizeof(tq2[0].qs), seed);
            cq3[(size_t)row*blocks + block].d[0] = 0x3c00;
            cq3[(size_t)row*blocks + block].d[1] = 0x3c00;
            tq2[(size_t)row*blocks + block].d = 0x3c00;
        }
    }

    const struct measurement generic = benchmark(
        "cq3_generic", ggml_vec_dot_cq3_0_cq8_0_generic, cq3, cq3_row_bytes,
        cq8, rows, columns, iterations, trials, reference, NULL, 0);
    const struct measurement native_hot = benchmark(
        "cq3_native_hot", ggml_vec_dot_cq3_0_cq8_0, cq3, cq3_row_bytes,
        cq8, rows, columns, iterations, trials, output, NULL, 0);

    float max_abs_error = 0.0f;
    for (int row = 0; row < rows; ++row) {
        max_abs_error = fmaxf(max_abs_error, fabsf(output[row] - reference[row]));
    }
    const struct measurement tq2_hot = benchmark(
        "tq2_native_hot", ggml_vec_dot_tq2_0_q8_K, tq2, tq2_row_bytes,
        q8k, rows, columns, iterations, trials, output, NULL, 0);
    const struct measurement native_cold = benchmark(
        "cq3_native_evicted", ggml_vec_dot_cq3_0_cq8_0, cq3, cq3_row_bytes,
        cq8, rows, columns, 1, trials, output, eviction, eviction_bytes);
    const struct measurement tq2_cold = benchmark(
        "tq2_native_evicted", ggml_vec_dot_tq2_0_q8_K, tq2, tq2_row_bytes,
        q8k, rows, columns, 1, trials, output, eviction, eviction_bytes);

    printf("{\n");
    printf("  \"schema\": \"clark.llama-cq3-benchmark.v1\",\n");
    printf("  \"shape\": {\"rows\": %d, \"columns\": %d},\n", rows, columns);
    printf("  \"iterations_per_trial\": %d,\n", iterations);
    printf("  \"trials\": %d,\n", trials);
    printf("  \"threads\": %d,\n", threads);
    printf("  \"eviction_bytes\": %zu,\n", eviction_bytes);
    printf("  \"cq3_stored_bits_per_weight\": %.9f,\n", 8.0*(double)sizeof(block_cq3_0)/(double)QK_CQ3_0);
    printf("  \"max_abs_native_vs_generic\": %.9g,\n", max_abs_error);
    printf("  \"measurements\": [\n");
    const struct measurement values[] = { generic, native_hot, tq2_hot, native_cold, tq2_cold };
    for (size_t i = 0; i < sizeof(values)/sizeof(values[0]); ++i) {
        printf("    {\"name\": \"%s\", \"cache_mode\": \"%s\", \"median_seconds\": %.9g, \"best_seconds\": %.9g, \"weight_gbps\": %.9g, \"checksum\": %.17g}%s\n",
               values[i].name, values[i].cache_mode, values[i].median_seconds, values[i].best_seconds,
               values[i].weight_gbps, values[i].checksum,
               i + 1 == sizeof(values)/sizeof(values[0]) ? "" : ",");
    }
    printf("  ]\n}\n");

    free(eviction);
    free(reference);
    free(output);
    free(q8k);
    free(cq8);
    free(input);
    free(tq2);
    free(cq3);
    return max_abs_error <= 1e-3f ? 0 : 4;
}
