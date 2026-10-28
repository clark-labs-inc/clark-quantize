#include "repack.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct WidthResult {
    int64_t width;
    int calls_per_sample;
    std::vector<double> parent_ns;
    std::vector<double> dual_ns;
};

volatile uint64_t output_guard = 0;

uint64_t sample_bytes(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    uint64_t value = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < size; i += 257) {
        value = (value ^ bytes[i]) * 0x100000001b3ULL;
    }
    return value;
}

template <typename Function>
double measure(Function && function, int calls) {
    const auto started = Clock::now();
    for (int call = 0; call < calls; ++call) {
        function();
    }
    const auto stopped = Clock::now();
    return std::chrono::duration<double, std::nano>(stopped - started).count()
        / calls;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }
    return values[middle];
}

void print_values(const std::vector<double> & values) {
    std::putchar('[');
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::putchar(',');
        }
        std::printf("%.3f", values[i]);
    }
    std::putchar(']');
}

WidthResult run_width(int64_t width, uint32_t seed) {
    const int nb = static_cast<int>(width / QK_K);
    const int calls = std::max(16, static_cast<int>(262144 / width));
    std::vector<float> input(static_cast<size_t>(4 * width));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-19.0f, 23.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = distribution(rng);
    }

    std::vector<block_q8_Kx4> parent_native(nb);
    std::vector<block_q8_K> parent_rows(4 * nb);
    std::vector<block_q8_Kx4> dual_native(nb);
    std::vector<block_q8_K> dual_rows(4 * nb);

    const auto parent = [&] {
        ggml_quantize_mat_q8_K_4x8(input.data(), parent_native.data(), width);
        ggml_repack_q8_K_4x8_to_q8_K_rows(
            parent_native.data(), parent_rows.data(), width);
    };
    const auto dual = [&] {
        ggml_quantize_mat_q8_K_4x8_dual(
            input.data(), dual_native.data(), dual_rows.data(), width);
    };

    parent();
    dual();
    if (std::memcmp(parent_native.data(), dual_native.data(),
                    parent_native.size() * sizeof(parent_native[0])) != 0 ||
        std::memcmp(parent_rows.data(), dual_rows.data(),
                    parent_rows.size() * sizeof(parent_rows[0])) != 0) {
        std::fprintf(stderr, "benchmark setup produced non-identical outputs at width %lld\n",
                     static_cast<long long>(width));
        std::exit(1);
    }

    for (int warmup = 0; warmup < 8; ++warmup) {
        parent();
        dual();
    }

    WidthResult result{width, calls, {}, {}};
    result.parent_ns.reserve(31);
    result.dual_ns.reserve(31);
    for (int pair = 0; pair < 31; ++pair) {
        if ((pair & 1) == 0) {
            result.parent_ns.push_back(measure(parent, calls));
            result.dual_ns.push_back(measure(dual, calls));
        } else {
            result.dual_ns.push_back(measure(dual, calls));
            result.parent_ns.push_back(measure(parent, calls));
        }
    }

    output_guard ^= sample_bytes(dual_native.data(),
                                 dual_native.size() * sizeof(dual_native[0]));
    output_guard ^= sample_bytes(dual_rows.data(),
                                 dual_rows.size() * sizeof(dual_rows[0]));
    return result;
}

} // namespace

int main() {
    static constexpr int64_t widths[] = {2048, 4096, 8192, 16384};
    std::vector<WidthResult> results;
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i) {
        results.push_back(run_width(widths[i], 0x382724u + static_cast<uint32_t>(i)));
    }

    double log_ratio_sum = 0.0;
    std::printf(
        "{\"schema\":\"q8_k_dual_layout_producer_microbenchmark_v24\","
        "\"paired_samples_per_width\":31,\"widths\":[");
    for (size_t i = 0; i < results.size(); ++i) {
        if (i != 0) {
            std::putchar(',');
        }
        const double parent_median = median(results[i].parent_ns);
        const double dual_median = median(results[i].dual_ns);
        const double ratio = parent_median / dual_median;
        log_ratio_sum += std::log(ratio);
        std::printf(
            "{\"width\":%lld,\"calls_per_sample\":%d,"
            "\"parent_ns_per_call\":",
            static_cast<long long>(results[i].width), results[i].calls_per_sample);
        print_values(results[i].parent_ns);
        std::printf(",\"dual_ns_per_call\":");
        print_values(results[i].dual_ns);
        std::printf(
            ",\"parent_median_ns\":%.3f,\"dual_median_ns\":%.3f,"
            "\"parent_over_dual\":%.9f}",
            parent_median, dual_median, ratio);
    }
    std::printf(
        "],\"parent_over_dual_geomean\":%.9f,\"output_guard\":%llu}\n",
        std::exp(log_ratio_sum / results.size()),
        static_cast<unsigned long long>(output_guard));
    return 0;
}
