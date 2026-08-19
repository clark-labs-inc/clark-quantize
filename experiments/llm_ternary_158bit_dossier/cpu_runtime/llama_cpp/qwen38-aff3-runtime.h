#pragma once

#include "ggml.h"

#include <memory>
#include <vector>

struct qwen38_aff3_image;
struct qwen38_aff3_node_state;

using qwen38_aff3_image_ptr = std::shared_ptr<const qwen38_aff3_image>;
using qwen38_aff3_node_state_ptr = std::shared_ptr<qwen38_aff3_node_state>;

bool qwen38_aff3_environment_requested();

std::vector<qwen38_aff3_image_ptr> qwen38_aff3_load_from_environment(
        int n_layers,
        int expected_rows,
        int expected_columns);

bool qwen38_aff3_any_enabled(
        const std::vector<qwen38_aff3_image_ptr> &images);

ggml_tensor * qwen38_aff3_build_projection(
        ggml_context *ctx,
        ggml_tensor *input,
        const qwen38_aff3_image_ptr &image,
        std::vector<qwen38_aff3_node_state_ptr> &node_states);
