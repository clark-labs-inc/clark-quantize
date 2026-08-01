//! End-to-end scalar Swin2SR x2 reference over a Clark browser artifact.

use crate::{
    block_hadamard_core, conv2d_same_core, cosine_attention_core, dense_matmul_core,
    gelu_core, layer_norm_core, pixel_shuffle_core, relative_bias_from_mlp,
    relative_coords_table, shifted_window_attention_mask, window_partition_nhwc_core,
    window_reverse_nhwc_core, ternary_matmul_affine_core, BrowserArtifact,
};

const CHANNELS: usize = 60;
const HEADS: usize = 6;
const WINDOW: usize = 8;
const LAYER_NORM_EPSILON: f32 = 1e-5;
const MEAN: [f32; 3] = [0.4488, 0.4371, 0.4040];

pub fn swin2sr_x2_core(
    artifact: &BrowserArtifact<'_>,
    input: &[f32],
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    if height < WINDOW
        || width < WINDOW
        || height % WINDOW != 0
        || width % WINDOW != 0
        || input.len() != 3 * height * width
    {
        return Err("Swin2SR input must be one RGB image with dimensions divisible by 8".into());
    }
    let mut normalized = input.to_vec();
    for channel in 0..3 {
        for value in &mut normalized[channel * height * width..(channel + 1) * height * width] {
            *value -= MEAN[channel];
        }
    }
    let embeddings = convolution(
        artifact,
        &normalized,
        [1, 3, height, width],
        "swin2sr.first_convolution",
        CHANNELS,
        3,
    )?;
    let projected = convolution(
        artifact,
        &embeddings,
        [1, CHANNELS, height, width],
        "swin2sr.embeddings.patch_embeddings.projection",
        CHANNELS,
        1,
    )?;
    let mut hidden = nchw_to_sequence(&projected, CHANNELS, height, width);
    hidden = layer_norm(
        artifact,
        &hidden,
        "swin2sr.embeddings.patch_embeddings.layernorm",
        CHANNELS,
    )?;
    let shifted_mask = shifted_window_attention_mask(height, width, WINDOW, WINDOW / 2)
        .map_err(String::from)?;
    for stage in 0..4 {
        let stage_residual = hidden.clone();
        for layer in 0..6 {
            hidden = transformer_layer(
                artifact,
                &hidden,
                height,
                width,
                stage,
                layer,
                &shifted_mask,
            )?;
        }
        let stage_prefix = format!("swin2sr.encoder.stages.{stage}");
        let unembedded = sequence_to_nchw(&hidden, CHANNELS, height, width);
        let convolved = convolution(
            artifact,
            &unembedded,
            [1, CHANNELS, height, width],
            &format!("{stage_prefix}.conv"),
            CHANNELS,
            3,
        )?;
        let projected = convolution(
            artifact,
            &convolved,
            [1, CHANNELS, height, width],
            &format!("{stage_prefix}.patch_embed.projection"),
            CHANNELS,
            1,
        )?;
        hidden = nchw_to_sequence(&projected, CHANNELS, height, width);
        add_in_place(&mut hidden, &stage_residual)?;
    }
    hidden = layer_norm(artifact, &hidden, "swin2sr.layernorm", CHANNELS)?;
    let body = sequence_to_nchw(&hidden, CHANNELS, height, width);
    let mut body = convolution(
        artifact,
        &body,
        [1, CHANNELS, height, width],
        "swin2sr.conv_after_body",
        CHANNELS,
        3,
    )?;
    add_in_place(&mut body, &embeddings)?;
    let upsample = convolution(
        artifact,
        &body,
        [1, CHANNELS, height, width],
        "upsample.conv",
        12,
        3,
    )?;
    let mut reconstruction =
        pixel_shuffle_core(&upsample, [1, 12, height, width], 2).map_err(String::from)?;
    let output_height = height * 2;
    let output_width = width * 2;
    for channel in 0..3 {
        for value in &mut reconstruction
            [channel * output_height * output_width..(channel + 1) * output_height * output_width]
        {
            *value += MEAN[channel];
        }
    }
    Ok(reconstruction)
}

fn transformer_layer(
    artifact: &BrowserArtifact<'_>,
    hidden: &[f32],
    height: usize,
    width: usize,
    stage: usize,
    layer: usize,
    shifted_mask: &[f32],
) -> Result<Vec<f32>, String> {
    let prefix = format!("swin2sr.encoder.stages.{stage}.layers.{layer}");
    let shift = if layer % 2 == 0 { 0 } else { WINDOW / 2 };
    let windows = window_partition_nhwc_core(
        hidden,
        [1, height, width, CHANNELS],
        WINDOW,
        shift,
    )
    .map_err(String::from)?;
    let window_batch = height * width;
    let query = linear(
        artifact,
        &windows,
        window_batch,
        &format!("{prefix}.attention.self.query"),
        CHANNELS,
        CHANNELS,
        true,
    )?;
    let key = linear(
        artifact,
        &windows,
        window_batch,
        &format!("{prefix}.attention.self.key"),
        CHANNELS,
        CHANNELS,
        false,
    )?;
    let value = linear(
        artifact,
        &windows,
        window_batch,
        &format!("{prefix}.attention.self.value"),
        CHANNELS,
        CHANNELS,
        true,
    )?;
    let relative_bias = relative_bias(artifact, &prefix)?;
    let raw_logit_scale = tensor(
        artifact,
        &format!("{prefix}.attention.self.logit_scale"),
        &[HEADS, 1, 1],
    )?;
    let logit_scale: Vec<f32> = raw_logit_scale
        .iter()
        .map(|value| value.min(100.0_f32.ln()).exp())
        .collect();
    let attention = cosine_attention_core(
        &query,
        &key,
        &value,
        [height * width / (WINDOW * WINDOW), WINDOW * WINDOW, CHANNELS],
        HEADS,
        &logit_scale,
        &relative_bias,
        if shift == 0 { None } else { Some(shifted_mask) },
    )
    .map_err(String::from)?;
    let attention = linear(
        artifact,
        &attention,
        window_batch,
        &format!("{prefix}.attention.output.dense"),
        CHANNELS,
        CHANNELS,
        true,
    )?;
    let attention = window_reverse_nhwc_core(
        &attention,
        [1, height, width, CHANNELS],
        WINDOW,
        shift,
    )
    .map_err(String::from)?;
    let normalized = layer_norm(
        artifact,
        &attention,
        &format!("{prefix}.layernorm_before"),
        CHANNELS,
    )?;
    let mut residual = hidden.to_vec();
    add_in_place(&mut residual, &normalized)?;
    let intermediate = linear(
        artifact,
        &residual,
        height * width,
        &format!("{prefix}.intermediate.dense"),
        CHANNELS * 2,
        CHANNELS,
        true,
    )?;
    let intermediate = gelu_core(&intermediate);
    let output = linear(
        artifact,
        &intermediate,
        height * width,
        &format!("{prefix}.output.dense"),
        CHANNELS,
        CHANNELS * 2,
        true,
    )?;
    let output = layer_norm(
        artifact,
        &output,
        &format!("{prefix}.layernorm_after"),
        CHANNELS,
    )?;
    add_in_place(&mut residual, &output)?;
    Ok(residual)
}

fn relative_bias(
    artifact: &BrowserArtifact<'_>,
    layer_prefix: &str,
) -> Result<Vec<f32>, String> {
    let coords = relative_coords_table(WINDOW).map_err(String::from)?;
    let prefix = format!("{layer_prefix}.attention.self.continuous_position_bias_mlp");
    let hidden = linear(
        artifact,
        &coords,
        (2 * WINDOW - 1) * (2 * WINDOW - 1),
        &format!("{prefix}.0"),
        512,
        2,
        true,
    )?;
    let hidden: Vec<f32> = hidden.into_iter().map(|value| value.max(0.0)).collect();
    let output = linear(
        artifact,
        &hidden,
        (2 * WINDOW - 1) * (2 * WINDOW - 1),
        &format!("{prefix}.2"),
        HEADS,
        512,
        false,
    )?;
    relative_bias_from_mlp(&output, WINDOW, HEADS).map_err(String::from)
}

fn convolution(
    artifact: &BrowserArtifact<'_>,
    input: &[f32],
    shape: [usize; 4],
    prefix: &str,
    out_channels: usize,
    kernel: usize,
) -> Result<Vec<f32>, String> {
    let weight = tensor(
        artifact,
        &format!("{prefix}.weight"),
        &[out_channels, shape[1], kernel, kernel],
    )?;
    let bias = tensor(artifact, &format!("{prefix}.bias"), &[out_channels])?;
    conv2d_same_core(input, &weight, &bias, shape, out_channels, kernel)
        .map_err(String::from)
}

fn linear(
    artifact: &BrowserArtifact<'_>,
    input: &[f32],
    batch: usize,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    has_bias: bool,
) -> Result<Vec<f32>, String> {
    let weight_name = format!("{prefix}.weight");
    let bias = if has_bias {
        tensor(artifact, &format!("{prefix}.bias"), &[out_features])?
    } else {
        Vec::new()
    };
    let is_packed = artifact.is_packed_ternary(&weight_name)?;
    let transformed;
    let transform_block = artifact.hadamard_block_size(&weight_name);
    let effective_input = if is_packed && transform_block.is_some() {
        transformed = block_hadamard_core(
            input,
            in_features,
            transform_block.expect("checked above"),
        )
        .map_err(String::from)?;
        transformed.as_slice()
    } else {
        input
    };
    if is_packed {
        let weight = artifact.packed_ternary(&weight_name)?;
        if weight.shape != [out_features, in_features] {
            return Err(format!("tensor shape mismatch for {weight_name}"));
        }
        ternary_matmul_affine_core(
            effective_input,
            &weight.packed,
            &weight.scales,
            &weight.offsets,
            &bias,
            batch,
            out_features,
            in_features,
            weight.group_size,
        )
        .map_err(String::from)
    } else {
        let weight = tensor(
            artifact,
            &weight_name,
            &[out_features, in_features],
        )?;
        dense_matmul_core(
            effective_input,
            &weight,
            &bias,
            batch,
            out_features,
            in_features,
        )
        .map_err(String::from)
    }
}

fn layer_norm(
    artifact: &BrowserArtifact<'_>,
    input: &[f32],
    prefix: &str,
    features: usize,
) -> Result<Vec<f32>, String> {
    let weight = tensor(artifact, &format!("{prefix}.weight"), &[features])?;
    let bias = tensor(artifact, &format!("{prefix}.bias"), &[features])?;
    layer_norm_core(input, &weight, &bias, features, LAYER_NORM_EPSILON)
        .map_err(String::from)
}

fn tensor(
    artifact: &BrowserArtifact<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>, String> {
    let tensor = artifact.tensor_f32(name)?;
    if tensor.shape != expected_shape {
        return Err(format!(
            "tensor shape mismatch for {name}: {:?} != {expected_shape:?}",
            tensor.shape
        ));
    }
    Ok(tensor.values)
}

fn nchw_to_sequence(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for channel in 0..channels {
        for y in 0..height {
            for x in 0..width {
                output[(y * width + x) * channels + channel] =
                    input[(channel * height + y) * width + x];
            }
        }
    }
    output
}

fn sequence_to_nchw(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                output[(channel * height + y) * width + x] =
                    input[(y * width + x) * channels + channel];
            }
        }
    }
    output
}

fn add_in_place(target: &mut [f32], source: &[f32]) -> Result<(), String> {
    if target.len() != source.len() {
        return Err("residual tensors have different lengths".into());
    }
    for (target_value, source_value) in target.iter_mut().zip(source) {
        *target_value += *source_value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{nchw_to_sequence, sequence_to_nchw};

    #[test]
    fn nchw_sequence_conversion_round_trips() {
        let input: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let sequence = nchw_to_sequence(&input, 3, 2, 4);
        assert_eq!(sequence_to_nchw(&sequence, 3, 2, 4), input);
    }
}
