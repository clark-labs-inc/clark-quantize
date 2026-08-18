#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "llamafile/sgemm.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

struct shape {
    int64_t m;
    int64_t k;
};

constexpr std::array<shape, 6> shapes = {{{10240, 160},
                                          {48, 160},
                                          {6144, 160},
                                          {5120, 192},
                                          {12288, 160},
                                          {1024, 160}}};

uint32_t mix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

void fill_blocks(std::vector<block_q8_0> &blocks, uint32_t stream) {
    for (size_t index = 0; index < blocks.size(); ++index) {
        block_q8_0 &block = blocks[index];
        const uint32_t scale_bits = mix32(static_cast<uint32_t>(index) ^ stream);
        const float scale = 0.0009765625f * static_cast<float>(1 + scale_bits % 31);
        block.d = GGML_FP32_TO_FP16(scale);
        for (int lane = 0; lane < QK8_0; ++lane) {
            const uint32_t bits = mix32(
                static_cast<uint32_t>(index * QK8_0 + lane) ^ (stream * 17U));
            block.qs[lane] = static_cast<int8_t>(static_cast<int>(bits % 255) - 127);
        }
    }
}

bool write_all(FILE *output, const void *data, size_t bytes) {
    return std::fwrite(data, 1, bytes, output) == bytes;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s OUTPUT\n", argv[0]);
        return 2;
    }
    FILE *output = std::fopen(argv[1], "wb");
    if (!output) {
        std::perror("fopen");
        return 2;
    }

    constexpr int64_t n = 4;
    constexpr int nth = 4;
    for (size_t shape_index = 0; shape_index < shapes.size(); ++shape_index) {
        const shape current = shapes[shape_index];
        std::vector<block_q8_0> weights(current.m * current.k);
        std::vector<block_q8_0> activations(n * current.k);
        std::vector<float> result(current.m * n, 0.0f);
        fill_blocks(weights, 0x12345678U + static_cast<uint32_t>(shape_index));
        fill_blocks(activations, 0x9abcdef0U + static_cast<uint32_t>(shape_index));

        for (int ith = 0; ith < nth; ++ith) {
            ggml_compute_params params = {};
            params.ith = ith;
            params.nth = nth;
            if (!llamafile_sgemm(&params,
                                 current.m, n, current.k,
                                 weights.data(), current.k,
                                 activations.data(), current.k,
                                 result.data(), current.m,
                                 GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32)) {
                std::fprintf(stderr, "sgemm rejected shape m=%lld n=4 k=%lld\n",
                             static_cast<long long>(current.m),
                             static_cast<long long>(current.k));
                std::fclose(output);
                return 1;
            }
        }

        const int64_t header[3] = {current.m, n, current.k};
        if (!write_all(output, header, sizeof(header)) ||
            !write_all(output, result.data(), result.size() * sizeof(float))) {
            std::perror("fwrite");
            std::fclose(output);
            return 2;
        }
    }
    if (std::fclose(output) != 0) {
        std::perror("fclose");
        return 2;
    }
    return 0;
}
