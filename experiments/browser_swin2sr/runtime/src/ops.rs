//! Portable scalar graph primitives for the Swin2SR WASM fallback.

use crate::math::dot_product;

pub fn block_hadamard_core(
    input: &[f32],
    features: usize,
    block_size: usize,
) -> Result<Vec<f32>, &'static str> {
    if features == 0
        || input.len() % features != 0
        || block_size == 0
        || !block_size.is_power_of_two()
        || features % block_size != 0
    {
        return Err("Hadamard transform dimensions are inconsistent");
    }
    let mut output = input.to_vec();
    for row in output.chunks_exact_mut(features) {
        for block in row.chunks_exact_mut(block_size) {
            let mut width = 1;
            while width < block_size {
                for start in (0..block_size).step_by(width * 2) {
                    for offset in 0..width {
                        let left = block[start + offset];
                        let right = block[start + width + offset];
                        block[start + offset] = left + right;
                        block[start + width + offset] = left - right;
                    }
                }
                width *= 2;
            }
            let scale = (block_size as f32).sqrt().recip();
            for value in block {
                *value *= scale;
            }
        }
    }
    Ok(output)
}

pub fn dense_matmul_core(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>, &'static str> {
    if [batch, out_features, in_features].contains(&0)
        || input.len() != batch * in_features
        || weight.len() != out_features * in_features
        || (!bias.is_empty() && bias.len() != out_features)
    {
        return Err("dense matrix metadata is inconsistent");
    }
    let mut output = vec![0.0; batch * out_features];
    for batch_index in 0..batch {
        for out_feature in 0..out_features {
            let input_row = &input[
                batch_index * in_features..(batch_index + 1) * in_features
            ];
            let weight_row = &weight[
                out_feature * in_features..(out_feature + 1) * in_features
            ];
            output[batch_index * out_features + out_feature] =
                bias.get(out_feature).copied().unwrap_or(0.0)
                    + dot_product(input_row, weight_row);
        }
    }
    Ok(output)
}

pub fn conv2d_same_core(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    shape: [usize; 4],
    out_channels: usize,
    kernel_size: usize,
) -> Result<Vec<f32>, &'static str> {
    let [batch, in_channels, height, width] = shape;
    if [batch, in_channels, height, width, out_channels, kernel_size].contains(&0) {
        return Err("conv dimensions must be positive");
    }
    if kernel_size % 2 == 0 {
        return Err("same convolution requires an odd kernel");
    }
    let input_values = batch
        .checked_mul(in_channels)
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(width))
        .ok_or("conv input dimensions overflow")?;
    let weight_values = out_channels
        .checked_mul(in_channels)
        .and_then(|value| value.checked_mul(kernel_size))
        .and_then(|value| value.checked_mul(kernel_size))
        .ok_or("conv weight dimensions overflow")?;
    if input.len() != input_values || weight.len() != weight_values {
        return Err("conv payload does not match dimensions");
    }
    if !bias.is_empty() && bias.len() != out_channels {
        return Err("conv bias does not match output channels");
    }
    let mut output = vec![0.0; batch * out_channels * height * width];
    let padding = kernel_size / 2;
    for batch_index in 0..batch {
        for out_channel in 0..out_channels {
            for output_y in 0..height {
                for output_x in 0..width {
                    let mut sum = bias.get(out_channel).copied().unwrap_or(0.0);
                    for in_channel in 0..in_channels {
                        for kernel_y in 0..kernel_size {
                            let input_y = output_y as isize + kernel_y as isize
                                - padding as isize;
                            if input_y < 0 || input_y >= height as isize {
                                continue;
                            }
                            for kernel_x in 0..kernel_size {
                                let input_x = output_x as isize + kernel_x as isize
                                    - padding as isize;
                                if input_x < 0 || input_x >= width as isize {
                                    continue;
                                }
                                let input_index = ((batch_index * in_channels + in_channel)
                                    * height
                                    + input_y as usize)
                                    * width
                                    + input_x as usize;
                                let weight_index = ((out_channel * in_channels + in_channel)
                                    * kernel_size
                                    + kernel_y)
                                    * kernel_size
                                    + kernel_x;
                                sum += input[input_index] * weight[weight_index];
                            }
                        }
                    }
                    let output_index =
                        ((batch_index * out_channels + out_channel) * height + output_y)
                            * width
                            + output_x;
                    output[output_index] = sum;
                }
            }
        }
    }
    Ok(output)
}

pub fn layer_norm_core(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    features: usize,
    epsilon: f32,
) -> Result<Vec<f32>, &'static str> {
    if features == 0 || input.len() % features != 0 {
        return Err("layer norm input does not contain complete feature rows");
    }
    if weight.len() != features || bias.len() != features || epsilon <= 0.0 {
        return Err("layer norm metadata is inconsistent");
    }
    let mut output = vec![0.0; input.len()];
    for (source, target) in input
        .chunks_exact(features)
        .zip(output.chunks_exact_mut(features))
    {
        let mean = source.iter().sum::<f32>() / features as f32;
        let variance = source
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / features as f32;
        let inverse_std = (variance + epsilon).sqrt().recip();
        for feature in 0..features {
            target[feature] =
                (source[feature] - mean) * inverse_std * weight[feature] + bias[feature];
        }
    }
    Ok(output)
}

pub fn gelu_core(input: &[f32]) -> Vec<f32> {
    let inverse_sqrt_two = std::f32::consts::FRAC_1_SQRT_2;
    input
        .iter()
        .map(|value| 0.5 * value * (1.0 + libm::erff(value * inverse_sqrt_two)))
        .collect()
}

pub fn softmax_last_dim_core(
    input: &[f32],
    features: usize,
) -> Result<Vec<f32>, &'static str> {
    if features == 0 || input.len() % features != 0 {
        return Err("softmax input does not contain complete rows");
    }
    let mut output = vec![0.0; input.len()];
    for (source, target) in input
        .chunks_exact(features)
        .zip(output.chunks_exact_mut(features))
    {
        let maximum = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0;
        for (target_value, source_value) in target.iter_mut().zip(source) {
            *target_value = (*source_value - maximum).exp();
            denominator += *target_value;
        }
        if !denominator.is_finite() || denominator == 0.0 {
            return Err("softmax denominator is not finite and positive");
        }
        for value in target {
            *value /= denominator;
        }
    }
    Ok(output)
}

pub fn pixel_shuffle_core(
    input: &[f32],
    shape: [usize; 4],
    upscale: usize,
) -> Result<Vec<f32>, &'static str> {
    let [batch, channels, height, width] = shape;
    if [batch, channels, height, width, upscale].contains(&0) {
        return Err("pixel shuffle dimensions must be positive");
    }
    let upscale_area = upscale
        .checked_mul(upscale)
        .ok_or("pixel shuffle upscale overflows")?;
    if channels % upscale_area != 0 {
        return Err("pixel shuffle channels are not divisible by upscale squared");
    }
    if input.len() != batch * channels * height * width {
        return Err("pixel shuffle payload does not match dimensions");
    }
    let output_channels = channels / upscale_area;
    let output_height = height * upscale;
    let output_width = width * upscale;
    let mut output = vec![0.0; batch * output_channels * output_height * output_width];
    for batch_index in 0..batch {
        for output_channel in 0..output_channels {
            for input_y in 0..height {
                for input_x in 0..width {
                    for offset_y in 0..upscale {
                        for offset_x in 0..upscale {
                            let input_channel = output_channel * upscale_area
                                + offset_y * upscale
                                + offset_x;
                            let input_index =
                                ((batch_index * channels + input_channel) * height + input_y)
                                    * width
                                    + input_x;
                            let output_y = input_y * upscale + offset_y;
                            let output_x = input_x * upscale + offset_x;
                            let output_index = ((batch_index * output_channels
                                + output_channel)
                                * output_height
                                + output_y)
                                * output_width
                                + output_x;
                            output[output_index] = input[input_index];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

pub fn window_partition_nhwc_core(
    input: &[f32],
    shape: [usize; 4],
    window: usize,
    shift: usize,
) -> Result<Vec<f32>, &'static str> {
    let [batch, height, width, channels] = shape;
    if [batch, height, width, channels, window].contains(&0)
        || height % window != 0
        || width % window != 0
        || shift >= window
        || input.len() != batch * height * width * channels
    {
        return Err("window partition metadata is inconsistent");
    }
    let windows_per_row = width / window;
    let windows_per_column = height / window;
    let mut output = vec![0.0; input.len()];
    for batch_index in 0..batch {
        for window_y in 0..windows_per_column {
            for window_x in 0..windows_per_row {
                let window_index = (batch_index * windows_per_column + window_y)
                    * windows_per_row
                    + window_x;
                for local_y in 0..window {
                    for local_x in 0..window {
                        let shifted_y = (window_y * window + local_y + shift) % height;
                        let shifted_x = (window_x * window + local_x + shift) % width;
                        for channel in 0..channels {
                            let input_index =
                                ((batch_index * height + shifted_y) * width + shifted_x)
                                    * channels
                                    + channel;
                            let output_index =
                                ((window_index * window + local_y) * window + local_x)
                                    * channels
                                    + channel;
                            output[output_index] = input[input_index];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

pub fn window_reverse_nhwc_core(
    windows: &[f32],
    shape: [usize; 4],
    window: usize,
    shift: usize,
) -> Result<Vec<f32>, &'static str> {
    let [batch, height, width, channels] = shape;
    if windows.len() != batch * height * width * channels
        || height % window != 0
        || width % window != 0
        || window == 0
        || shift >= window
    {
        return Err("window reverse metadata is inconsistent");
    }
    let windows_per_row = width / window;
    let windows_per_column = height / window;
    let mut output = vec![0.0; windows.len()];
    for batch_index in 0..batch {
        for window_y in 0..windows_per_column {
            for window_x in 0..windows_per_row {
                let window_index = (batch_index * windows_per_column + window_y)
                    * windows_per_row
                    + window_x;
                for local_y in 0..window {
                    for local_x in 0..window {
                        let shifted_y = (window_y * window + local_y + shift) % height;
                        let shifted_x = (window_x * window + local_x + shift) % width;
                        for channel in 0..channels {
                            let input_index =
                                ((window_index * window + local_y) * window + local_x)
                                    * channels
                                    + channel;
                            let output_index =
                                ((batch_index * height + shifted_y) * width + shifted_x)
                                    * channels
                                    + channel;
                            output[output_index] = windows[input_index];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{
        block_hadamard_core, conv2d_same_core, dense_matmul_core, gelu_core,
        layer_norm_core, pixel_shuffle_core,
        softmax_last_dim_core, window_partition_nhwc_core, window_reverse_nhwc_core,
    };

    #[test]
    fn normalized_hadamard_is_its_own_inverse() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let transformed = block_hadamard_core(&input, 4, 4).unwrap();
        let restored = block_hadamard_core(&transformed, 4, 4).unwrap();
        for (actual, expected) in restored.iter().zip(input) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn dense_matmul_uses_out_in_weight_layout() {
        let output = dense_matmul_core(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 0.0, 0.5, -0.5],
            &[0.25, -0.25],
            2,
            2,
            2,
        )
        .unwrap();
        assert_eq!(output, [1.25, -0.75, 3.25, -0.75]);
    }

    #[test]
    fn identity_convolution_preserves_nchw_values() {
        let input: Vec<f32> = (1..=9).map(|value| value as f32).collect();
        let output =
            conv2d_same_core(&input, &[1.0], &[0.0], [1, 1, 3, 3], 1, 1).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn layer_norm_matches_known_two_feature_rows() {
        let output =
            layer_norm_core(&[1.0, 3.0, 2.0, 6.0], &[1.0, 2.0], &[0.5, -0.5], 2, 1e-5)
                .unwrap();
        assert!((output[0] + 0.499_995).abs() < 1e-5);
        assert!((output[1] - 1.499_99).abs() < 1e-5);
        assert!((output[2] + 0.499_999).abs() < 1e-5);
        assert!((output[3] - 1.499_997).abs() < 1e-5);
    }

    #[test]
    fn exact_gelu_has_expected_reference_values() {
        let output = gelu_core(&[-1.0, 0.0, 1.0]);
        assert!((output[0] + 0.158_655_26).abs() < 1e-7);
        assert_eq!(output[1], 0.0);
        assert!((output[2] - 0.841_344_7).abs() < 1e-7);
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let output = softmax_last_dim_core(&[1000.0, 1001.0, 1002.0], 3).unwrap();
        assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((output[2] - 0.665_240_94).abs() < 1e-6);
    }

    #[test]
    fn pixel_shuffle_uses_pytorch_channel_order() {
        let output =
            pixel_shuffle_core(&[1.0, 2.0, 3.0, 4.0], [1, 4, 1, 1], 2).unwrap();
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn shifted_window_partition_round_trips() {
        let input: Vec<f32> = (0..16).map(|value| value as f32).collect();
        let windows = window_partition_nhwc_core(&input, [1, 4, 4, 1], 2, 1).unwrap();
        let output =
            window_reverse_nhwc_core(&windows, [1, 4, 4, 1], 2, 1).unwrap();
        assert_eq!(output, input);
    }
}
