#include "repack.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string mode = "verify";
    std::string output;
    int64_t m = 10240;
    int64_t k = 5120;
    int warmups = 2;
    int repetitions = 31;
    size_t eviction_bytes = 256ULL * 1024ULL * 1024ULL;
};

uint64_t fnv1a(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void fill_inputs(
    std::vector<block_q4_Kx8> & weights,
    std::vector<block_q8_K> & activations,
    uint32_t seed,
    int special_case = 0) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    std::uniform_int_distribution<int> signed_distribution(-127, 127);

    for (auto & block : weights) {
        for (int column = 0; column < 8; ++column) {
            block.d[column] = static_cast<ggml_half>(0x2800 + column * 17);
            block.dmin[column] = static_cast<ggml_half>(0x2400 + column * 13);
        }
        for (auto & value : block.scales) {
            value = static_cast<uint8_t>(byte_distribution(generator));
        }
        for (auto & value : block.qs) {
            value = static_cast<uint8_t>(byte_distribution(generator));
        }
    }

    const size_t blocks_per_row = activations.size() / 4;
    for (size_t block_index = 0; block_index < activations.size(); ++block_index) {
        block_q8_K & block = activations[block_index];
        const int row = static_cast<int>(block_index / blocks_per_row);
        block.d = 0.001f * static_cast<float>(row + 1);
        for (int index = 0; index < QK_K; ++index) {
            int value = signed_distribution(generator);
            if (special_case == 1) {
                value = 0;
            } else if (special_case == 2) {
                value = index % 2 == 0 ? -128 : 127;
            } else if (special_case == 3) {
                value = index % 2 == 0 ? -73 : 73;
            } else if (special_case == 4) {
                value = (row + 1) * ((index % 15) - 7);
            } else if (special_case == 5) {
                value = ((index / 16) + 3 * row) % 17 - 8;
            }
            block.qs[index] = static_cast<int8_t>(value);
        }
        for (int group = 0; group < QK_K / 16; ++group) {
            int sum = 0;
            for (int offset = 0; offset < 16; ++offset) {
                sum += block.qs[group * 16 + offset];
            }
            block.bsums[group] = static_cast<int16_t>(sum);
        }
    }
}

void call_kernel(
    int64_t m,
    int64_t k,
    const std::vector<block_q4_Kx8> & weights,
    const std::vector<block_q8_K> & activations,
    std::vector<float> & output) {
    ggml_gemv_q4_K_8x8_q8_K(
        static_cast<int>(k),
        output.data(),
        static_cast<size_t>(m),
        weights.data(),
        activations.data(),
        4,
        static_cast<int>(m));
}

void evict_cache(std::vector<uint8_t> & eviction) {
    static volatile uint64_t sink = 0;
    uint64_t sum = sink;
    for (size_t index = 0; index < eviction.size(); index += 64) {
        eviction[index] = static_cast<uint8_t>(eviction[index] + 1);
        sum += eviction[index];
    }
    sink = sum;
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            return argv[index];
        };
        if (argument == "--mode") {
            options.mode = value();
        } else if (argument == "--output") {
            options.output = value();
        } else if (argument == "--m") {
            options.m = std::stoll(value());
        } else if (argument == "--k") {
            options.k = std::stoll(value());
        } else if (argument == "--warmups") {
            options.warmups = std::stoi(value());
        } else if (argument == "--repetitions") {
            options.repetitions = std::stoi(value());
        } else if (argument == "--eviction-bytes") {
            options.eviction_bytes = static_cast<size_t>(std::stoull(value()));
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.m <= 0 || options.m % 8 != 0 || options.k <= 0 || options.k % QK_K != 0) {
        throw std::invalid_argument("m must be divisible by 8 and k by QK_K");
    }
    return options;
}

int run_verification(const Options & options) {
    if (options.output.empty()) {
        throw std::invalid_argument("verification requires --output");
    }
    constexpr int64_t m = 8;
    constexpr int64_t k = QK_K;
    std::vector<float> all_outputs;
    all_outputs.reserve((1024 + 5) * 4 * m);
    for (int test_case = 0; test_case < 1029; ++test_case) {
        std::vector<block_q4_Kx8> weights((m / 8) * (k / QK_K));
        std::vector<block_q8_K> activations(4 * (k / QK_K));
        std::vector<float> output(4 * m, 0.0f);
        const int special_case = test_case < 5 ? test_case + 1 : 0;
        fill_inputs(weights, activations, 0x9e3779b9U + static_cast<uint32_t>(test_case), special_case);
        const uint64_t weights_before = fnv1a(weights.data(), weights.size() * sizeof(weights[0]));
        const uint64_t activations_before = fnv1a(activations.data(), activations.size() * sizeof(activations[0]));
        call_kernel(m, k, weights, activations, output);
        if (weights_before != fnv1a(weights.data(), weights.size() * sizeof(weights[0]))
            || activations_before != fnv1a(activations.data(), activations.size() * sizeof(activations[0]))) {
            throw std::runtime_error("kernel modified a source buffer");
        }
        all_outputs.insert(all_outputs.end(), output.begin(), output.end());
    }
    std::ofstream stream(options.output, std::ios::binary | std::ios::trunc);
    stream.write(
        reinterpret_cast<const char *>(all_outputs.data()),
        static_cast<std::streamsize>(all_outputs.size() * sizeof(all_outputs[0])));
    if (!stream) {
        throw std::runtime_error("failed to write verification output");
    }
    std::cout << "{\"mode\":\"verify\",\"seeded_cases\":1024,\"special_cases\":5,"
              << "\"source_unchanged\":true,\"output_floats\":" << all_outputs.size() << "}\n";
    return 0;
}

int run_benchmark(const Options & options) {
    const int64_t blocks_per_row = options.k / QK_K;
    std::vector<block_q4_Kx8> weights((options.m / 8) * blocks_per_row);
    std::vector<block_q8_K> activations(4 * blocks_per_row);
    std::vector<float> output(4 * options.m, 0.0f);
    std::vector<uint8_t> eviction(options.eviction_bytes, 1);
    fill_inputs(weights, activations, 0x51f15eU);
    for (int index = 0; index < options.warmups; ++index) {
        evict_cache(eviction);
        call_kernel(options.m, options.k, weights, activations, output);
    }
    std::vector<double> microseconds;
    for (int index = 0; index < options.repetitions; ++index) {
        evict_cache(eviction);
        const auto start = std::chrono::steady_clock::now();
        call_kernel(options.m, options.k, weights, activations, output);
        const auto stop = std::chrono::steady_clock::now();
        microseconds.push_back(
            std::chrono::duration<double, std::micro>(stop - start).count());
    }
    std::cout << "{\"mode\":\"benchmark\",\"m\":" << options.m
              << ",\"n\":4,\"k\":" << options.k
              << ",\"warmups\":" << options.warmups
              << ",\"repetitions\":" << options.repetitions
              << ",\"eviction_bytes\":" << options.eviction_bytes
              << ",\"output_fnv1a\":" << fnv1a(output.data(), output.size() * sizeof(output[0]))
              << ",\"samples_us\":[";
    for (size_t index = 0; index < microseconds.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << microseconds[index];
    }
    std::cout << "]}\n";
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.mode == "verify") {
            return run_verification(options);
        }
        if (options.mode == "benchmark") {
            return run_benchmark(options);
        }
        throw std::invalid_argument("--mode must be verify or benchmark");
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
