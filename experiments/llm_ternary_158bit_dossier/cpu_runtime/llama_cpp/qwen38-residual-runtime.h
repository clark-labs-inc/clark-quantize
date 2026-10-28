#pragma once

#include "ggml.h"

#include <memory>
#include <vector>


struct qwen38_residual_bundle;
struct qwen38_residual_node_state;

using qwen38_residual_bundle_ptr = std::shared_ptr<const qwen38_residual_bundle>;
using qwen38_residual_node_state_ptr =
    std::shared_ptr<qwen38_residual_node_state>;

bool qwen38_residual_environment_requested();

qwen38_residual_bundle_ptr qwen38_residual_load_from_environment(
        int n_layers,
        int expected_input_features,
        int expected_output_features);

bool qwen38_residual_any_enabled(
        const qwen38_residual_bundle_ptr &bundle);

bool qwen38_residual_layer_enabled(
        const qwen38_residual_bundle_ptr &bundle,
        int layer);

ggml_tensor * qwen38_residual_apply(
        ggml_context *ctx,
        ggml_tensor *base_output,
        ggml_tensor *block_input,
        const qwen38_residual_bundle_ptr &bundle,
        int layer,
        std::vector<qwen38_residual_node_state_ptr> &node_states);
