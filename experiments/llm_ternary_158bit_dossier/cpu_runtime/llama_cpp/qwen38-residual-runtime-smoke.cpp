#include "qwen38-residual-runtime.cpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kFeatures = 5120;
constexpr int kRank = 16;
constexpr size_t kPayloadBytes = 174144;

[[noreturn]] void smoke_fail(const std::string &message) {
    throw std::runtime_error("Qwen3.8 residual v57 smoke: " + message);
}

int parse_positive(const char *text, const char *name) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        smoke_fail(std::string("invalid ") + name);
    }
    return static_cast<int>(value);
}

int parse_layer(const char *text) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value >= 64) {
        smoke_fail("--layer must be in [0, 63]");
    }
    return static_cast<int>(value);
}

struct Options {
    std::string bundle;
    std::string payload;
    std::string training_sha;
    std::string residual_sha;
    std::string prereg_sha;
    int layer = -1;
    int tokens = 0;
    int threads = 0;
};

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto take = [&](const char *name) -> const char * {
            if (index + 1 >= argc) smoke_fail(std::string("missing ") + name);
            return argv[++index];
        };
        if (argument == "--bundle") options.bundle = take("--bundle");
        else if (argument == "--payload") options.payload = take("--payload");
        else if (argument == "--training-sha") options.training_sha = take("--training-sha");
        else if (argument == "--residual-sha") options.residual_sha = take("--residual-sha");
        else if (argument == "--prereg-sha") options.prereg_sha = take("--prereg-sha");
        else if (argument == "--layer") options.layer = parse_layer(take("--layer"));
        else if (argument == "--tokens") options.tokens = parse_positive(take("--tokens"), "--tokens");
        else if (argument == "--threads") options.threads = parse_positive(take("--threads"), "--threads");
        else smoke_fail("unknown argument: " + argument);
    }
    if (options.bundle.empty() || options.payload.empty() ||
        options.training_sha.empty() || options.residual_sha.empty() ||
        options.prereg_sha.empty() || options.layer < 0 ||
        options.tokens <= 0 || options.threads <= 0) {
        smoke_fail("all bundle, payload, identity, layer, token, and thread arguments are required");
    }
    return options;
}

uint64_t splitmix64(uint64_t &state) {
    uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

float deterministic_float(uint64_t &state) {
    const uint32_t bits = static_cast<uint32_t>(splitmix64(state) >> 40U);
    return static_cast<float>(bits) * (2.0f / 16777215.0f) - 1.0f;
}

std::vector<uint8_t> load_payload(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamsize>(kPayloadBytes)) {
        smoke_fail("payload byte count changed");
    }
    stream.seekg(0);
    std::vector<uint8_t> bytes(kPayloadBytes);
    stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    if (!stream) smoke_fail("payload read failed");
    return bytes;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        setenv("QWEN38_RESIDUAL_IMAGE", options.bundle.c_str(), 1);
        setenv("QWEN38_RESIDUAL_TRAINING_SHA256", options.training_sha.c_str(), 1);
        setenv("QWEN38_RESIDUAL_RECEIPT_SHA256", options.residual_sha.c_str(), 1);
        setenv("QWEN38_RESIDUAL_PREREG_SHA256", options.prereg_sha.c_str(), 1);
        const auto bundle = qwen38_residual_load_from_environment(64, kFeatures, kFeatures);
        if (!qwen38_residual_layer_enabled(bundle, options.layer)) {
            smoke_fail("requested layer is not active in bundle");
        }

        const size_t metadata_bytes = 16U * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(16, false) + 65536U;
        ggml_init_params params = {
            /*.mem_size =*/ metadata_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc =*/ true,
        };
        ggml_context *ctx = ggml_init(params);
        if (ctx == nullptr) smoke_fail("ggml context initialization failed");
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend == nullptr) smoke_fail("ggml CPU backend initialization failed");
        ggml_backend_cpu_set_n_threads(backend, options.threads);

        ggml_tensor *input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, kFeatures, options.tokens);
        ggml_tensor *base = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, kFeatures, options.tokens);
        std::vector<qwen38_residual_node_state_ptr> states;
        ggml_tensor *output = qwen38_residual_apply(
            ctx, base, input, bundle, options.layer, states);
        if (states.size() != 1 || output == base) {
            smoke_fail("active graph node denominator changed");
        }
        int bypass_layer = -1;
        for (int layer = 0; layer < 64; ++layer) {
            if (!qwen38_residual_layer_enabled(bundle, layer)) {
                bypass_layer = layer;
                break;
            }
        }
        if (bypass_layer < 0) smoke_fail("smoke bundle has no exact bypass layer");
        std::vector<qwen38_residual_node_state_ptr> bypass_states;
        if (qwen38_residual_apply(
                ctx, base, input, bundle, bypass_layer, bypass_states) != base ||
            !bypass_states.empty()) {
            smoke_fail("exact bypass unexpectedly built a graph node");
        }

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) smoke_fail("ggml tensor allocation failed");
        ggml_cgraph *graph = ggml_new_graph_custom(ctx, 16, false);
        if (graph == nullptr) smoke_fail("ggml graph allocation failed");
        ggml_build_forward_expand(graph, output);
        if (!ggml_backend_supports_op(backend, output)) {
            smoke_fail("ggml CPU backend rejected residual custom node");
        }

        const size_t elements = static_cast<size_t>(kFeatures) * options.tokens;
        std::vector<float> input_values(elements);
        std::vector<float> base_values(elements);
        uint64_t random_state = UINT64_C(20260819);
        for (size_t index = 0; index < elements; ++index) {
            input_values[index] = deterministic_float(random_state);
            base_values[index] = deterministic_float(random_state) * 0.25f;
        }
        ggml_backend_tensor_set(input, input_values.data(), 0, elements * sizeof(float));
        ggml_backend_tensor_set(base, base_values.data(), 0, elements * sizeof(float));
        const ggml_status graph_status = ggml_backend_graph_compute(backend, graph);
        if (graph_status != GGML_STATUS_SUCCESS) {
            smoke_fail(std::string("graph compute failed: ") +
                       ggml_status_to_string(graph_status));
        }
        std::vector<float> graph_output(elements);
        ggml_backend_tensor_get(output, graph_output.data(), 0, elements * sizeof(float));

        const std::vector<uint8_t> payload = load_payload(options.payload);
        const int8_t *left_codes = reinterpret_cast<const int8_t *>(payload.data());
        const uint16_t *left_scales = reinterpret_cast<const uint16_t *>(
            payload.data() + static_cast<size_t>(kFeatures) * kRank);
        const int8_t *right_codes = reinterpret_cast<const int8_t *>(
            reinterpret_cast<const uint8_t *>(left_scales) + kRank * sizeof(uint16_t));
        const uint16_t *right_scales = reinterpret_cast<const uint16_t *>(
            reinterpret_cast<const uint8_t *>(right_codes) +
            static_cast<size_t>(kRank) * kFeatures);
        const uint16_t *bias = right_scales + kRank;
        std::vector<float> direct_output = base_values;
        for (int token = 0; token < options.tokens; ++token) {
            float scratch[kRank];
            const int status = cq3_calibration_residual_f32_avx2_trusted(
                input_values.data() + static_cast<size_t>(token) * kFeatures,
                left_codes, left_scales, right_codes, right_scales, bias,
                kFeatures, kFeatures, kRank, scratch,
                direct_output.data() + static_cast<size_t>(token) * kFeatures,
                1U);
            if (status != 0) smoke_fail("direct native residual failed");
        }
        float maximum_absolute_error = 0.0f;
        for (size_t index = 0; index < elements; ++index) {
            maximum_absolute_error = std::max(
                maximum_absolute_error,
                std::abs(graph_output[index] - direct_output[index]));
        }
        const bool bit_exact = std::memcmp(
            graph_output.data(), direct_output.data(), elements * sizeof(float)) == 0;
        if (!bit_exact) smoke_fail("graph output differs from exact native algebra");

        std::cout << "{\"schema\":\"qwen38_residual_runtime_smoke_v57\","
                  << "\"layer\":" << options.layer << ","
                  << "\"tokens\":" << options.tokens << ","
                  << "\"threads\":" << options.threads << ","
                  << "\"maximum_absolute_error\":" << maximum_absolute_error << ","
                  << "\"bit_exact\":true,\"bypass_exact\":true,"
                  << "\"complete\":true,\"terminal\":true}\n";

        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
