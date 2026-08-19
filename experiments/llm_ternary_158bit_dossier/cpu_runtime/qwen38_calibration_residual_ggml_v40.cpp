#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "qwen38_calibration_residual_cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kInputDim = 5120;
constexpr uint32_t kOutputDim = 5120;
constexpr uint32_t kRank = 16;
constexpr size_t kLeftCodeBytes = static_cast<size_t>(kInputDim) * kRank;
constexpr size_t kScaleBytes = static_cast<size_t>(kRank) * sizeof(uint16_t);
constexpr size_t kRightCodeBytes = static_cast<size_t>(kRank) * kOutputDim;
constexpr size_t kBiasBytes = static_cast<size_t>(kOutputDim) * sizeof(uint16_t);
constexpr size_t kPayloadBytes =
    kLeftCodeBytes + kScaleBytes + kRightCodeBytes + kScaleBytes + kBiasBytes;

struct FreeDeleter {
    void operator()(void *pointer) const noexcept {
        std::free(pointer);
    }
};

struct Payload {
    std::unique_ptr<void, FreeDeleter> storage;
    const int8_t *left_codes = nullptr;
    const uint16_t *left_scales = nullptr;
    const int8_t *right_codes = nullptr;
    const uint16_t *right_scales = nullptr;
    const uint16_t *bias = nullptr;
};

struct GraphUserData {
    const Payload *payload = nullptr;
    std::atomic<int> status {0};
};

struct Options {
    std::string payload_path;
    int tokens = 1;
    int threads = 4;
    uint64_t seed = 20260819;
    int blocks = 7;
    int warmups = 3;
    int samples = 9;
    int inner = 128;
    bool validate_only = false;
};

struct GraphRuntime {
    ggml_context *ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *input = nullptr;
    ggml_tensor *base = nullptr;
    ggml_tensor *output = nullptr;
    GraphUserData userdata;

    ~GraphRuntime() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    GraphRuntime(const GraphRuntime &) = delete;
    GraphRuntime &operator=(const GraphRuntime &) = delete;
    GraphRuntime() = default;
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

int parse_positive(const char *value, const char *name) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
        fail(std::string("invalid ") + name);
    }
    return static_cast<int>(parsed);
}

uint64_t parse_u64(const char *value, const char *name) {
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        fail(std::string("invalid ") + name);
    }
    return static_cast<uint64_t>(parsed);
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto take = [&](const char *name) -> const char * {
            if (index + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (argument == "--payload") {
            options.payload_path = take("--payload");
        } else if (argument == "--tokens") {
            options.tokens = parse_positive(take("--tokens"), "--tokens");
        } else if (argument == "--threads") {
            options.threads = parse_positive(take("--threads"), "--threads");
        } else if (argument == "--seed") {
            options.seed = parse_u64(take("--seed"), "--seed");
        } else if (argument == "--blocks") {
            options.blocks = parse_positive(take("--blocks"), "--blocks");
        } else if (argument == "--warmups") {
            options.warmups = parse_positive(take("--warmups"), "--warmups");
        } else if (argument == "--samples") {
            options.samples = parse_positive(take("--samples"), "--samples");
        } else if (argument == "--inner") {
            options.inner = parse_positive(take("--inner"), "--inner");
        } else if (argument == "--validate-only") {
            options.validate_only = true;
        } else {
            fail("unknown argument: " + argument);
        }
    }
    if (options.payload_path.empty()) {
        fail("--payload is required");
    }
    if (options.tokens > 64) {
        fail("--tokens must be at most 64");
    }
    return options;
}

Payload load_payload(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        fail("could not open payload: " + path);
    }
    const std::streamsize size = stream.tellg();
    if (size != static_cast<std::streamsize>(kPayloadBytes)) {
        fail("payload byte count changed");
    }
    stream.seekg(0);
    void *raw = nullptr;
    if (posix_memalign(&raw, 64, kPayloadBytes) != 0 || raw == nullptr) {
        fail("aligned payload allocation failed");
    }
    Payload payload;
    payload.storage.reset(raw);
    stream.read(static_cast<char *>(raw), size);
    if (!stream) {
        fail("payload read failed");
    }
    const uint8_t *cursor = static_cast<const uint8_t *>(raw);
    payload.left_codes = reinterpret_cast<const int8_t *>(cursor);
    cursor += kLeftCodeBytes;
    payload.left_scales = reinterpret_cast<const uint16_t *>(cursor);
    cursor += kScaleBytes;
    payload.right_codes = reinterpret_cast<const int8_t *>(cursor);
    cursor += kRightCodeBytes;
    payload.right_scales = reinterpret_cast<const uint16_t *>(cursor);
    cursor += kScaleBytes;
    payload.bias = reinterpret_cast<const uint16_t *>(cursor);
    cursor += kBiasBytes;
    if (cursor != static_cast<const uint8_t *>(raw) + kPayloadBytes) {
        fail("payload parser did not consume exact bytes");
    }
    return payload;
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

void graph_callback(
    ggml_tensor *dst,
    const ggml_tensor *base,
    const ggml_tensor *input,
    int ith,
    int nth,
    void *opaque
) {
    auto *userdata = static_cast<GraphUserData *>(opaque);
    if (userdata == nullptr || userdata->payload == nullptr ||
        dst->type != GGML_TYPE_F32 || base->type != GGML_TYPE_F32 ||
        input->type != GGML_TYPE_F32 || dst->ne[0] != kOutputDim ||
        base->ne[0] != kOutputDim || input->ne[0] != kInputDim ||
        dst->ne[1] != input->ne[1] || base->ne[1] != input->ne[1]) {
        if (userdata != nullptr) {
            userdata->status.store(1, std::memory_order_relaxed);
        }
        return;
    }
    const Payload &payload = *userdata->payload;
    for (int64_t token = ith; token < input->ne[1]; token += nth) {
        const auto *input_row = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(input->data) + token * input->nb[1]);
        const auto *base_row = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(base->data) + token * base->nb[1]);
        auto *output_row = reinterpret_cast<float *>(
            static_cast<uint8_t *>(dst->data) + token * dst->nb[1]);
        std::memcpy(output_row, base_row, kOutputDim * sizeof(float));
        float rank_scratch[kRank];
        const int status = cq3_calibration_residual_f32_avx2_trusted(
            input_row,
            payload.left_codes,
            payload.left_scales,
            payload.right_codes,
            payload.right_scales,
            payload.bias,
            kInputDim,
            kOutputDim,
            kRank,
            rank_scratch,
            output_row,
            1U);
        if (status != 0) {
            userdata->status.store(status, std::memory_order_relaxed);
            return;
        }
    }
}

std::unique_ptr<GraphRuntime> make_graph(
    const Payload &payload,
    int tokens,
    int threads
) {
    auto runtime = std::make_unique<GraphRuntime>();
    runtime->userdata.payload = &payload;
    const size_t metadata_bytes =
        16U * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false) + 65536U;
    ggml_init_params params = {
        /*.mem_size =*/ metadata_bytes,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    runtime->ctx = ggml_init(params);
    if (runtime->ctx == nullptr) {
        fail("ggml context initialization failed");
    }
    runtime->backend = ggml_backend_cpu_init();
    if (runtime->backend == nullptr) {
        fail("ggml CPU backend initialization failed");
    }
    ggml_backend_cpu_set_n_threads(runtime->backend, threads);
    runtime->input = ggml_new_tensor_2d(
        runtime->ctx, GGML_TYPE_F32, kInputDim, tokens);
    runtime->base = ggml_new_tensor_2d(
        runtime->ctx, GGML_TYPE_F32, kOutputDim, tokens);
    runtime->output = ggml_map_custom2(
        runtime->ctx,
        runtime->base,
        runtime->input,
        graph_callback,
        tokens,
        &runtime->userdata);
    ggml_set_name(runtime->input, "residual_input");
    ggml_set_name(runtime->base, "base_output");
    ggml_set_name(runtime->output, "base_plus_compact_residual");
    runtime->buffer = ggml_backend_alloc_ctx_tensors(runtime->ctx, runtime->backend);
    if (runtime->buffer == nullptr) {
        fail("ggml tensor allocation failed");
    }
    runtime->graph = ggml_new_graph_custom(runtime->ctx, 16, false);
    if (runtime->graph == nullptr) {
        fail("ggml graph allocation failed");
    }
    ggml_build_forward_expand(runtime->graph, runtime->output);
    if (!ggml_backend_supports_op(runtime->backend, runtime->output)) {
        fail("ggml CPU backend rejected compact residual custom op");
    }
    return runtime;
}

void set_graph_inputs(
    GraphRuntime &runtime,
    const std::vector<float> &input,
    const std::vector<float> &base
) {
    ggml_backend_tensor_set(
        runtime.input, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(
        runtime.base, base.data(), 0, base.size() * sizeof(float));
}

void compute_graph(GraphRuntime &runtime) {
    runtime.userdata.status.store(0, std::memory_order_relaxed);
    const ggml_status status =
        ggml_backend_graph_compute(runtime.backend, runtime.graph);
    if (status != GGML_STATUS_SUCCESS) {
        fail(std::string("ggml graph compute failed: ") + ggml_status_to_string(status));
    }
    const int callback_status =
        runtime.userdata.status.load(std::memory_order_relaxed);
    if (callback_status != 0) {
        fail("compact residual graph callback failed with status " +
             std::to_string(callback_status));
    }
}

void compute_direct(
    const Payload &payload,
    const std::vector<float> &input,
    const std::vector<float> &base,
    int tokens,
    std::vector<float> &output
) {
    output = base;
    for (int token = 0; token < tokens; ++token) {
        float rank_scratch[kRank];
        const int status = cq3_calibration_residual_f32_avx2_trusted(
            input.data() + static_cast<size_t>(token) * kInputDim,
            payload.left_codes,
            payload.left_scales,
            payload.right_codes,
            payload.right_scales,
            payload.bias,
            kInputDim,
            kOutputDim,
            kRank,
            rank_scratch,
            output.data() + static_cast<size_t>(token) * kOutputDim,
            1U);
        if (status != 0) {
            fail("direct compact residual failed with status " +
                 std::to_string(status));
        }
    }
}

double maximum_absolute_error(
    const std::vector<float> &left,
    const std::vector<float> &right
) {
    if (left.size() != right.size()) {
        fail("comparison vector sizes differ");
    }
    double maximum = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(
            static_cast<double>(left[index]) - static_cast<double>(right[index])));
    }
    return maximum;
}

template <typename Function>
double measure_ns(int inner, Function &&function) {
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < inner; ++iteration) {
        function();
    }
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(stop - start).count() / inner;
}

void print_samples(const std::vector<std::vector<double>> &blocks) {
    std::cout << '[';
    for (size_t block = 0; block < blocks.size(); ++block) {
        if (block != 0) {
            std::cout << ',';
        }
        std::cout << '[';
        for (size_t sample = 0; sample < blocks[block].size(); ++sample) {
            if (sample != 0) {
                std::cout << ',';
            }
            std::cout << std::setprecision(17) << blocks[block][sample];
        }
        std::cout << ']';
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        const uint32_t required_features =
            CQ3_CALIBRATION_FEATURE_AVX2 |
            CQ3_CALIBRATION_FEATURE_FMA |
            CQ3_CALIBRATION_FEATURE_F16C;
        const uint32_t observed_features =
            cq3_calibration_residual_cpu_features();
        if ((observed_features & required_features) != required_features) {
            fail("AVX2/FMA/F16C is unavailable");
        }

        const Payload payload = load_payload(options.payload_path);
        uint64_t random_state = options.seed;
        std::vector<float> input(
            static_cast<size_t>(options.tokens) * kInputDim);
        std::vector<float> base(
            static_cast<size_t>(options.tokens) * kOutputDim);
        for (float &value : input) {
            value = deterministic_float(random_state);
        }
        for (float &value : base) {
            value = deterministic_float(random_state);
        }

        auto graph = make_graph(payload, options.tokens, options.threads);
        set_graph_inputs(*graph, input, base);
        std::vector<float> direct;
        compute_direct(payload, input, base, options.tokens, direct);
        compute_graph(*graph);
        std::vector<float> observed(direct.size());
        ggml_backend_tensor_get(
            graph->output, observed.data(), 0, observed.size() * sizeof(float));
        const double maximum_error = maximum_absolute_error(direct, observed);

        if (options.validate_only) {
            std::cout << "{\"schema\":\"qwen38_calibration_residual_ggml_validation_v40\""
                      << ",\"tokens\":" << options.tokens
                      << ",\"threads\":" << options.threads
                      << ",\"payload_bytes\":" << kPayloadBytes
                      << ",\"rank_scratch_bytes\":" << kRank * sizeof(float)
                      << ",\"maximum_absolute_error_vs_direct\":"
                      << std::setprecision(17) << maximum_error
                      << ",\"validated\":true}\n";
            return 0;
        }

        std::vector<std::vector<double>> direct_blocks(options.blocks);
        std::vector<std::vector<double>> graph_blocks(options.blocks);
        std::vector<float> direct_scratch;
        for (int block = 0; block < options.blocks; ++block) {
            const bool graph_first = block % 2 != 0;
            auto warm_direct = [&]() {
                compute_direct(
                    payload, input, base, options.tokens, direct_scratch);
            };
            auto warm_graph = [&]() { compute_graph(*graph); };
            for (int warmup = 0; warmup < options.warmups; ++warmup) {
                if (graph_first) {
                    warm_graph();
                    warm_direct();
                } else {
                    warm_direct();
                    warm_graph();
                }
            }
            auto collect_direct = [&]() {
                direct_blocks[block].push_back(measure_ns(options.inner, warm_direct));
            };
            auto collect_graph = [&]() {
                graph_blocks[block].push_back(measure_ns(options.inner, warm_graph));
            };
            for (int sample = 0; sample < options.samples; ++sample) {
                if (graph_first) {
                    collect_graph();
                    collect_direct();
                } else {
                    collect_direct();
                    collect_graph();
                }
            }
        }

        std::cout << "{\"schema\":\"qwen38_calibration_residual_ggml_case_v40\""
                  << ",\"tokens\":" << options.tokens
                  << ",\"threads\":" << options.threads
                  << ",\"seed\":" << options.seed
                  << ",\"payload_bytes\":" << kPayloadBytes
                  << ",\"rank_scratch_bytes_per_task\":"
                  << kRank * sizeof(float)
                  << ",\"maximum_absolute_error_vs_direct\":"
                  << std::setprecision(17) << maximum_error
                  << ",\"blocks\":" << options.blocks
                  << ",\"samples_per_block\":" << options.samples
                  << ",\"inner_calls_per_sample\":" << options.inner
                  << ",\"direct_ns_per_batch_blocks\":";
        print_samples(direct_blocks);
        std::cout << ",\"ggml_ns_per_batch_blocks\":";
        print_samples(graph_blocks);
        std::cout << ",\"complete\":true}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "qwen38_calibration_residual_ggml_v40: "
                  << error.what() << '\n';
        return 1;
    }
}
