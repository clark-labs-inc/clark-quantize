#define QWEN38_AFF3_DEVELOPMENT_TESTING 1
#include "qwen38-aff3-runtime.cpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kModelLayers = 64;
constexpr int kRows = 5120;
constexpr int kColumns = 6144;
constexpr int kThreads = 4;

[[noreturn]] void smoke_fail(const std::string &message) {
    throw std::runtime_error("Qwen3.8 AFF3 v49 smoke: " + message);
}

int parse_positive(const char *value, const char *name) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
        smoke_fail(std::string("invalid ") + name);
    }
    return static_cast<int>(parsed);
}

int parse_layer(const char *value) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 ||
        parsed >= kModelLayers) {
        smoke_fail("invalid --layer");
    }
    return static_cast<int>(parsed);
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

struct Options {
    int tokens = 0;
    int layer = -1;
    bool validate_only = false;
};

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto take = [&](const char *name) -> const char * {
            if (index + 1 >= argc) {
                smoke_fail(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (argument == "--tokens") {
            options.tokens = parse_positive(take("--tokens"), "--tokens");
        } else if (argument == "--layer") {
            options.layer = parse_layer(take("--layer"));
        } else if (argument == "--validate-only") {
            options.validate_only = true;
        } else {
            smoke_fail("unknown argument: " + argument);
        }
    }
    if ((options.tokens != 1 && options.tokens != 8) ||
        options.layer < 0 || options.layer >= kModelLayers ||
        !options.validate_only) {
        smoke_fail(
            "requires --tokens 1|8, --layer 0..63, and --validate-only; "
            "performance timing is forbidden");
    }
    return options;
}

struct GraphRuntime {
    ggml_context *context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *input = nullptr;
    ggml_tensor *output = nullptr;
    std::vector<qwen38_aff3_node_state_ptr> node_states;

    ~GraphRuntime() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
        if (context != nullptr) {
            ggml_free(context);
        }
    }

    GraphRuntime() = default;
    GraphRuntime(const GraphRuntime &) = delete;
    GraphRuntime &operator=(const GraphRuntime &) = delete;
};

std::unique_ptr<GraphRuntime> make_graph(
        const qwen38_aff3_image_ptr &image,
        int tokens) {
    auto runtime = std::make_unique<GraphRuntime>();
    const size_t metadata_bytes = 16U * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(16, false) + 65536U;
    const ggml_init_params parameters = {
        /*.mem_size =*/ metadata_bytes,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    runtime->context = ggml_init(parameters);
    if (runtime->context == nullptr) {
        smoke_fail("GGML context initialization failed");
    }
    runtime->backend = ggml_backend_cpu_init();
    if (runtime->backend == nullptr) {
        smoke_fail("GGML CPU backend initialization failed");
    }
    ggml_backend_cpu_set_n_threads(runtime->backend, kThreads);
    runtime->input = ggml_new_tensor_2d(
        runtime->context, GGML_TYPE_F32, kColumns, tokens);
    runtime->output = qwen38_aff3_build_projection(
        runtime->context, runtime->input, image, runtime->node_states);
    runtime->buffer = ggml_backend_alloc_ctx_tensors(
        runtime->context, runtime->backend);
    if (runtime->buffer == nullptr) {
        smoke_fail("GGML tensor allocation failed");
    }
    runtime->graph = ggml_new_graph_custom(runtime->context, 16, false);
    if (runtime->graph == nullptr) {
        smoke_fail("GGML graph allocation failed");
    }
    ggml_build_forward_expand(runtime->graph, runtime->output);
    if (!ggml_backend_supports_op(runtime->backend, runtime->output)) {
        smoke_fail("GGML CPU backend rejected the AFF3 custom op");
    }
    return runtime;
}

void compute_direct(
        const qwen38_aff3_image &image,
        const std::vector<float> &input,
        int tokens,
        std::vector<float> &output) {
    output.resize(static_cast<size_t>(tokens) * image.rows);
    ThreadScratch scratch(image.columns);
    for (int token = 0; token < tokens; ++token) {
        const int status = cq3_aff3_rotated_gemv_x16_v47(
            image.weights(),
            image.alpha_anchors(),
            image.mu_anchors(),
            image.alpha_codebook(),
            image.mu_codebook(),
            image.angles(),
            image.pairings(),
            input.data() + static_cast<size_t>(token) * image.columns,
            image.rows,
            image.columns,
            image.stages,
            scratch.rotated.data(),
            scratch.quantized.data(),
            scratch.scales.data(),
            scratch.sums.data(),
            output.data() + static_cast<size_t>(token) * image.rows);
        if (status != 0) {
            smoke_fail("direct v47 kernel failed: " + std::to_string(status));
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto images = qwen38_aff3_load_from_environment(
            kModelLayers, kRows, kColumns);
        if (!qwen38_aff3_any_enabled(images) ||
            images[options.layer] == nullptr ||
            std::count_if(images.begin(), images.end(),
                [](const qwen38_aff3_image_ptr &image) {
                    return image != nullptr;
                }) != 1) {
            smoke_fail("environment must select exactly the requested layer");
        }
        const qwen38_aff3_image_ptr &image = images[options.layer];
        uint64_t random_state = UINT64_C(20260819);
        std::vector<float> input(
            static_cast<size_t>(options.tokens) * kColumns);
        for (float &value : input) {
            value = deterministic_float(random_state);
        }
        std::vector<float> direct;
        compute_direct(*image, input, options.tokens, direct);
        auto graph = make_graph(image, options.tokens);
        ggml_backend_tensor_set(
            graph->input, input.data(), 0, input.size() * sizeof(float));
        const ggml_status status = ggml_backend_graph_compute(
            graph->backend, graph->graph);
        if (status != GGML_STATUS_SUCCESS) {
            smoke_fail(std::string("GGML graph failed: ") +
                       ggml_status_to_string(status));
        }
        std::vector<float> observed(direct.size());
        ggml_backend_tensor_get(
            graph->output, observed.data(), 0,
            observed.size() * sizeof(float));
        double maximum_error = 0.0;
        for (size_t index = 0; index < direct.size(); ++index) {
            maximum_error = std::max(maximum_error, std::abs(
                static_cast<double>(direct[index]) - observed[index]));
        }
        const bool exact = std::memcmp(
            direct.data(), observed.data(), direct.size() * sizeof(float)) == 0;
        if (graph->node_states.size() != 1) {
            smoke_fail("unexpected AFF3 graph state count");
        }
        const uint32_t task_mask =
            graph->node_states[0]->observed_task_mask.load(
                std::memory_order_relaxed);
        if (!exact || maximum_error != 0.0 || task_mask != UINT32_C(15)) {
            smoke_fail("bit parity or four-task execution gate failed");
        }
        std::cout
            << "{\"schema\":\"qwen38_27b_aff3_llamacpp_runtime_smoke_v49\""
            << ",\"model\":\"Qwen/Qwen3.8-27B\""
            << ",\"layer\":" << options.layer
            << ",\"shape\":[" << kRows << ',' << kColumns << ']'
            << ",\"tokens\":" << options.tokens
            << ",\"threads\":" << kThreads
            << ",\"observed_task_mask\":" << task_mask
            << ",\"output_bit_exact_vs_direct_v47\":"
            << (exact ? "true" : "false")
            << ",\"maximum_absolute_error_vs_direct_v47\":"
            << std::setprecision(17) << maximum_error
            << ",\"persistent_dense_weight_bytes\":0"
            << ",\"decoded_group_scale_matrix_bytes\":0"
            << ",\"performance_timing_ran\":false"
            << ",\"passed\":true,\"complete\":true,\"terminal\":true}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
