//! Swin2SR cosine-window attention for the portable graph runner.

use crate::math::dot_product;

pub fn cosine_attention_core(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    shape: [usize; 3],
    heads: usize,
    logit_scale: &[f32],
    relative_bias: &[f32],
    mask: Option<&[f32]>,
) -> Result<Vec<f32>, &'static str> {
    let [windows, tokens, channels] = shape;
    if [windows, tokens, channels, heads].contains(&0) || channels % heads != 0 {
        return Err("attention dimensions are inconsistent");
    }
    let value_count = windows
        .checked_mul(tokens)
        .and_then(|count| count.checked_mul(channels))
        .ok_or("attention dimensions overflow")?;
    if query.len() != value_count || key.len() != value_count || value.len() != value_count {
        return Err("attention projection payload does not match dimensions");
    }
    if logit_scale.len() != heads
        || relative_bias.len() != heads * tokens * tokens
        || logit_scale.iter().any(|scale| !scale.is_finite())
    {
        return Err("attention bias metadata is inconsistent");
    }
    let mask_windows = if let Some(mask_values) = mask {
        let window_values = tokens * tokens;
        if mask_values.is_empty() || mask_values.len() % window_values != 0 {
            return Err("attention mask dimensions are inconsistent");
        }
        mask_values.len() / window_values
    } else {
        0
    };

    let head_dim = channels / heads;
    let mut normalized_query = query.to_vec();
    let mut normalized_key = key.to_vec();
    for window in 0..windows {
        for token in 0..tokens {
            for head in 0..heads {
                let start = (window * tokens + token) * channels + head * head_dim;
                normalize_vector(&mut normalized_query[start..start + head_dim]);
                normalize_vector(&mut normalized_key[start..start + head_dim]);
            }
        }
    }

    let mut output = vec![0.0; value_count];
    let mut scores = vec![0.0; tokens];
    for window in 0..windows {
        for head in 0..heads {
            let scale = logit_scale[head].min(100.0);
            for query_token in 0..tokens {
                let query_start =
                    (window * tokens + query_token) * channels + head * head_dim;
                for key_token in 0..tokens {
                    let key_start =
                        (window * tokens + key_token) * channels + head * head_dim;
                    let cosine = dot_product(
                        &normalized_query[query_start..query_start + head_dim],
                        &normalized_key[key_start..key_start + head_dim],
                    );
                    let bias_index = (head * tokens + query_token) * tokens + key_token;
                    let mask_value = mask.map_or(0.0, |mask_values| {
                        let mask_window = window % mask_windows;
                        mask_values[(mask_window * tokens + query_token) * tokens + key_token]
                    });
                    // Transformers 4.57.6 adds the shifted-window mask twice.
                    scores[key_token] =
                        cosine * scale + relative_bias[bias_index] + 2.0 * mask_value;
                }
                stable_softmax(&mut scores)?;
                let output_start =
                    (window * tokens + query_token) * channels + head * head_dim;
                let output_head = &mut output[output_start..output_start + head_dim];
                for value_token in 0..tokens {
                    let value_start = (window * tokens + value_token) * channels
                        + head * head_dim;
                    let value_head = &value[value_start..value_start + head_dim];
                    let score = scores[value_token];
                    for (target, source) in output_head.iter_mut().zip(value_head) {
                        *target += score * source;
                    }
                }
            }
        }
    }
    Ok(output)
}

fn normalize_vector(values: &mut [f32]) {
    let norm = dot_product(values, values).sqrt();
    let denominator = norm.max(1e-12);
    for value in values {
        *value /= denominator;
    }
}

fn stable_softmax(values: &mut [f32]) -> Result<(), &'static str> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        denominator += *value;
    }
    if !denominator.is_finite() || denominator == 0.0 {
        return Err("attention softmax denominator is invalid");
    }
    for value in values {
        *value /= denominator;
    }
    Ok(())
}

pub fn shifted_window_attention_mask(
    height: usize,
    width: usize,
    window: usize,
    shift: usize,
) -> Result<Vec<f32>, &'static str> {
    if [height, width, window, shift].contains(&0)
        || height % window != 0
        || width % window != 0
        || shift >= window
    {
        return Err("shifted-window mask dimensions are inconsistent");
    }
    let mut labels = vec![0_u8; height * width];
    for y in 0..height {
        let y_region = if y < height - window {
            0
        } else if y < height - shift {
            1
        } else {
            2
        };
        for x in 0..width {
            let x_region = if x < width - window {
                0
            } else if x < width - shift {
                1
            } else {
                2
            };
            labels[y * width + x] = y_region * 3 + x_region;
        }
    }
    let windows_per_row = width / window;
    let windows_per_column = height / window;
    let tokens = window * window;
    let mut window_labels = vec![0_u8; height * width];
    for window_y in 0..windows_per_column {
        for window_x in 0..windows_per_row {
            let window_index = window_y * windows_per_row + window_x;
            for local_y in 0..window {
                for local_x in 0..window {
                    let source = (window_y * window + local_y) * width
                        + window_x * window
                        + local_x;
                    window_labels[window_index * tokens + local_y * window + local_x] =
                        labels[source];
                }
            }
        }
    }
    let mut mask = vec![0.0; windows_per_column * windows_per_row * tokens * tokens];
    for window_index in 0..windows_per_column * windows_per_row {
        for query_token in 0..tokens {
            for key_token in 0..tokens {
                if window_labels[window_index * tokens + query_token]
                    != window_labels[window_index * tokens + key_token]
                {
                    mask[(window_index * tokens + query_token) * tokens + key_token] =
                        -100.0;
                }
            }
        }
    }
    Ok(mask)
}

pub fn relative_coords_table(window: usize) -> Result<Vec<f32>, &'static str> {
    if window < 2 {
        return Err("relative coordinates require a window of at least two");
    }
    let side = 2 * window - 1;
    let denominator = (window - 1) as f32;
    let log_denominator = 8.0_f32.log2();
    let mut table = Vec::with_capacity(side * side * 2);
    for y in -(window as isize - 1)..=window as isize - 1 {
        for x in -(window as isize - 1)..=window as isize - 1 {
            for coordinate in [y as f32, x as f32] {
                let normalized = coordinate / denominator * 8.0;
                table.push(normalized.signum() * (normalized.abs() + 1.0).log2()
                    / log_denominator);
            }
        }
    }
    Ok(table)
}

pub fn relative_position_index(window: usize) -> Result<Vec<usize>, &'static str> {
    if window == 0 {
        return Err("relative position index requires a positive window");
    }
    let tokens = window * window;
    let side = 2 * window - 1;
    let mut indices = vec![0; tokens * tokens];
    for query_y in 0..window {
        for query_x in 0..window {
            let query = query_y * window + query_x;
            for key_y in 0..window {
                for key_x in 0..window {
                    let key = key_y * window + key_x;
                    let relative_y =
                        query_y as isize - key_y as isize + window as isize - 1;
                    let relative_x =
                        query_x as isize - key_x as isize + window as isize - 1;
                    indices[query * tokens + key] =
                        relative_y as usize * side + relative_x as usize;
                }
            }
        }
    }
    Ok(indices)
}

pub fn relative_bias_from_mlp(
    mlp_output: &[f32],
    window: usize,
    heads: usize,
) -> Result<Vec<f32>, &'static str> {
    let side = 2 * window - 1;
    if window == 0 || heads == 0 || mlp_output.len() != side * side * heads {
        return Err("relative bias MLP output dimensions are inconsistent");
    }
    let tokens = window * window;
    let indices = relative_position_index(window)?;
    let mut bias = vec![0.0; heads * tokens * tokens];
    for head in 0..heads {
        for query in 0..tokens {
            for key in 0..tokens {
                let table_index = indices[query * tokens + key];
                let value = mlp_output[table_index * heads + head];
                bias[(head * tokens + query) * tokens + key] =
                    16.0 / (1.0 + (-value).exp());
            }
        }
    }
    Ok(bias)
}

#[cfg(test)]
mod tests {
    use super::{
        cosine_attention_core, relative_bias_from_mlp, relative_coords_table,
        relative_position_index, shifted_window_attention_mask,
    };

    #[test]
    fn one_head_cosine_attention_matches_reference() {
        let query = [1.0, 0.0, 0.0, 1.0];
        let key = query;
        let value = [1.0, 2.0, 3.0, 4.0];
        let output = cosine_attention_core(
            &query,
            &key,
            &value,
            [1, 2, 2],
            1,
            &[1.0],
            &[0.0; 4],
            None,
        )
        .unwrap();

        assert!((output[0] - 1.537_882_8).abs() < 1e-6);
        assert!((output[1] - 2.537_882_8).abs() < 1e-6);
        assert!((output[2] - 2.462_117_2).abs() < 1e-6);
        assert!((output[3] - 3.462_117_2).abs() < 1e-6);
    }

    #[test]
    fn shifted_mask_blocks_cross_region_tokens() {
        let mask = shifted_window_attention_mask(8, 8, 4, 2).unwrap();
        assert_eq!(mask.len(), 4 * 16 * 16);
        assert!(mask.iter().any(|value| *value == -100.0));
        assert!(mask.iter().any(|value| *value == 0.0));
    }

    #[test]
    fn relative_position_geometry_matches_swin_indexing() {
        let coordinates = relative_coords_table(2).unwrap();
        let indices = relative_position_index(2).unwrap();

        assert_eq!(coordinates.len(), 3 * 3 * 2);
        assert_eq!(&indices[0..4], &[4, 3, 1, 0]);
        assert_eq!(&indices[12..16], &[8, 7, 5, 4]);
    }

    #[test]
    fn relative_bias_gathers_head_last_mlp_output() {
        let mlp_output: Vec<f32> = (0..18).map(|value| value as f32 / 10.0).collect();
        let bias = relative_bias_from_mlp(&mlp_output, 2, 2).unwrap();

        assert_eq!(bias.len(), 2 * 4 * 4);
        let expected = 16.0 / (1.0 + (-0.8_f32).exp());
        assert!((bias[0] - expected).abs() < 1e-6);
    }
}
