use wasm_bindgen::prelude::*;

mod artifact;
mod attention;
mod external_data;
mod math;
mod model;
mod ops;
mod scale_codec;

pub use artifact::{
    BrowserArtifact,
    PackedSymmetricIntTensor,
    PackedTernaryTensor,
    RawTensor,
};
pub use attention::{
    cosine_attention_core, relative_bias_from_mlp, relative_coords_table,
    relative_position_index, shifted_window_attention_mask,
};
pub use external_data::decode_onnx_external_data_core;
pub use model::swin2sr_x2_core;
pub use ops::{
    block_hadamard_core, conv2d_same_core, dense_matmul_core, gelu_core,
    layer_norm_core, pixel_shuffle_core, softmax_last_dim_core,
    window_partition_nhwc_core, window_reverse_nhwc_core,
};

const TRITS_PER_WORD: usize = 20;

pub fn ternary_matmul_core(
    input: &[f32],
    packed: &[u32],
    scales: &[f32],
    bias: &[f32],
    batch_size: usize,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Result<Vec<f32>, &'static str> {
    ternary_matmul_affine_core(
        input,
        packed,
        scales,
        &[],
        bias,
        batch_size,
        out_features,
        in_features,
        group_size,
    )
}

pub fn ternary_matmul_affine_core(
    input: &[f32],
    packed: &[u32],
    scales: &[f32],
    offsets: &[f32],
    bias: &[f32],
    batch_size: usize,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Result<Vec<f32>, &'static str> {
    let input_count = batch_size
        .checked_mul(in_features)
        .ok_or("input dimensions overflow")?;
    if input.len() != input_count {
        return Err("input length does not match batch_size * in_features");
    }
    let weight_count = out_features
        .checked_mul(in_features)
        .ok_or("weight dimensions overflow")?;
    if group_size == 0 {
        return Err("group_size must be positive");
    }
    let expected_words = weight_count.div_ceil(TRITS_PER_WORD);
    let expected_scales = weight_count.div_ceil(group_size);
    if packed.len() != expected_words
        || scales.len() != expected_scales
        || (!offsets.is_empty() && offsets.len() != expected_scales)
    {
        return Err("packed weight metadata is inconsistent");
    }
    if !bias.is_empty() && bias.len() != out_features {
        return Err("bias length does not match out_features");
    }

    let output_count = batch_size
        .checked_mul(out_features)
        .ok_or("output dimensions overflow")?;
    let mut output = vec![0.0; output_count];
    let mut decoded_row = vec![0.0; in_features];
    for row in 0..out_features {
        let initial = if bias.is_empty() { 0.0 } else { bias[row] };
        let row_offset = row * in_features;
        let mut packed_index = row_offset / TRITS_PER_WORD;
        let packed_offset = row_offset % TRITS_PER_WORD;
        let mut word = packed[packed_index];
        for _ in 0..packed_offset {
            word /= 3;
        }
        for column in 0..in_features {
            let weight_index = row_offset + column;
            if column > 0 && weight_index % TRITS_PER_WORD == 0 {
                packed_index += 1;
                word = packed[packed_index];
            }
            let code = (word % 3) as f32 - 1.0;
            word /= 3;
            let group = weight_index / group_size;
            let offset = offsets.get(group).copied().unwrap_or(0.0);
            decoded_row[column] = code * scales[group] + offset;
        }
        for batch in 0..batch_size {
            let input_row = &input[batch * in_features..(batch + 1) * in_features];
            output[batch * out_features + row] =
                initial + math::dot_product(input_row, &decoded_row);
        }
    }
    Ok(output)
}

#[wasm_bindgen]
pub fn ternary_matmul(
    input: &[f32],
    packed: &[u32],
    scales: &[f32],
    bias: &[f32],
    batch_size: usize,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Result<Vec<f32>, JsValue> {
    ternary_matmul_core(
        input,
        packed,
        scales,
        bias,
        batch_size,
        out_features,
        in_features,
        group_size,
    )
    .map_err(JsValue::from_str)
}

#[wasm_bindgen]
pub fn ternary_linear(
    input: &[f32],
    packed: &[u32],
    scales: &[f32],
    bias: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Result<Vec<f32>, JsValue> {
    ternary_matmul_core(
        input,
        packed,
        scales,
        bias,
        1,
        out_features,
        in_features,
        group_size,
    )
    .map_err(JsValue::from_str)
}

#[wasm_bindgen]
pub fn swin2sr_x2(
    manifest_json: &[u8],
    blob: &[u8],
    input: &[f32],
    height: usize,
    width: usize,
) -> Result<Vec<f32>, JsValue> {
    let artifact =
        BrowserArtifact::from_bytes(manifest_json, blob).map_err(|error| JsValue::from_str(&error))?;
    swin2sr_x2_core(&artifact, input, height, width).map_err(|error| JsValue::from_str(&error))
}

#[wasm_bindgen]
pub fn decode_onnx_external_data(
    manifest_json: &[u8],
    blob: &[u8],
    layout_json: &[u8],
) -> Result<Vec<u8>, JsValue> {
    decode_onnx_external_data_core(manifest_json, blob, layout_json)
        .map_err(|error| JsValue::from_str(&error))
}

#[cfg(test)]
mod tests {
    use super::{ternary_matmul_affine_core, ternary_matmul_core};

    fn pack(trits: &[u32]) -> Vec<u32> {
        trits
            .chunks(20)
            .map(|chunk| {
                let mut packed = 0;
                let mut place = 1;
                for trit in chunk {
                    packed += trit * place;
                    place *= 3;
                }
                packed
            })
            .collect()
    }

    #[test]
    fn decodes_packed_trits_and_applies_group_scales() {
        let trits = [0_u32, 1, 2, 2, 0, 1];
        let mut packed = 0;
        let mut place = 1;
        for trit in trits {
            packed += trit * place;
            place *= 3;
        }
        let output = ternary_matmul_core(
            &[1.0, 2.0, 3.0],
            &[packed],
            &[2.0, 0.5],
            &[0.25, -0.5],
            1,
            2,
            3,
            3,
        )
        .expect("valid packed matrix");

        assert!((output[0] - 4.25).abs() < 1e-6);
        assert!((output[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn applies_one_affine_offset_per_weight_group() {
        let trits = [0_u32, 1, 2, 2, 0, 1];
        let packed = pack(&trits);
        let output = ternary_matmul_affine_core(
            &[1.0, 2.0, 3.0],
            &packed,
            &[2.0, 0.5],
            &[0.25, -0.5],
            &[],
            1,
            2,
            3,
            3,
        )
        .expect("valid affine ternary matrix");

        assert!((output[0] - 5.5).abs() < 1e-6);
        assert!((output[1] + 3.5).abs() < 1e-6);
    }

    #[test]
    fn streams_across_unaligned_word_row_and_scale_boundaries() {
        let trits: Vec<u32> = (0..26).map(|index| (index % 3) as u32).collect();
        let packed = pack(&trits);
        let scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let input: Vec<f32> = (1..=13).map(|value| value as f32 / 10.0).collect();
        let output = ternary_matmul_core(
            &input,
            &packed,
            &scales,
            &[],
            1,
            2,
            13,
            5,
        )
            .expect("valid unaligned packed matrix");
        let expected: Vec<f32> = (0..2)
            .map(|row| {
                (0..13)
                    .map(|column| {
                        let index = row * 13 + column;
                        input[column] * (trits[index] as f32 - 1.0) * scales[index / 5]
                    })
                    .sum()
            })
            .collect();

        assert!((output[0] - expected[0]).abs() < 1e-6);
        assert!((output[1] - expected[1]).abs() < 1e-6);
    }

    #[test]
    fn rejects_zero_group_size_without_panicking() {
        let error = ternary_matmul_core(
            &[1.0],
            &[1],
            &[1.0],
            &[],
            1,
            1,
            1,
            0,
        );
        assert_eq!(error, Err("group_size must be positive"));
    }

    #[test]
    fn reuses_packed_weights_across_a_batch() {
        let trits: Vec<u32> = (0..42).map(|index| ((index * 2 + 1) % 3) as u32).collect();
        let packed = pack(&trits);
        let scales: Vec<f32> = (1..=9).map(|value| value as f32 / 4.0).collect();
        let input: Vec<f32> = (0..21)
            .flat_map(|column| {
                [
                    column as f32 / 10.0,
                    1.0 - column as f32 / 20.0,
                    (column as f32).sin(),
                ]
            })
            .collect();
        let mut batch_major = vec![0.0; 3 * 21];
        for column in 0..21 {
            for batch in 0..3 {
                batch_major[batch * 21 + column] = input[column * 3 + batch];
            }
        }
        let output = ternary_matmul_core(
            &batch_major,
            &packed,
            &scales,
            &[0.25, -0.5],
            3,
            2,
            21,
            5,
        )
        .expect("valid batched packed matrix");

        for batch in 0..3 {
            for row in 0..2 {
                let expected = (0..21).fold(
                    [0.25, -0.5][row],
                    |sum, column| {
                        let index = row * 21 + column;
                        sum + batch_major[batch * 21 + column]
                            * (trits[index] as f32 - 1.0)
                            * scales[index / 5]
                    },
                );
                assert!((output[batch * 2 + row] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn rejects_incorrect_batch_input_length() {
        let error = ternary_matmul_core(
            &[1.0, 2.0, 3.0],
            &[1],
            &[1.0],
            &[],
            2,
            1,
            2,
            2,
        );
        assert_eq!(
            error,
            Err("input length does not match batch_size * in_features")
        );
    }
}
