use std::collections::BTreeMap;

use serde::Deserialize;

use crate::scale_codec::{decode_fp16, decode_scale_codebook};

const FORMAT_V1: &str = "clark-air-browser-v1";
const FORMAT_V2: &str = "clark-air-browser-v2";
const FORMAT_V3: &str = "clark-air-browser-v3";
const FORMAT_V4: &str = "clark-air-browser-v4";
const TRITS_PER_WORD: usize = 20;

#[derive(Deserialize)]
struct Manifest {
    format: String,
    mode: String,
    group_size: usize,
    input_transform: Option<InputTransform>,
    tensors: BTreeMap<String, TensorRecord>,
}

#[derive(Deserialize)]
struct InputTransform {
    kind: String,
    block_size: Option<usize>,
}

#[derive(Deserialize)]
struct TensorRecord {
    kind: String,
    shape: Vec<usize>,
    pad: Option<usize>,
    group_size: Option<usize>,
    codes: Option<ByteRange>,
    scales: Option<ByteRange>,
    offsets: Option<ByteRange>,
    scale_codec: Option<String>,
    scale_count: Option<usize>,
    scale_index_bits: Option<usize>,
    scale_codebook: Option<ByteRange>,
    scale_indices: Option<ByteRange>,
    bit_width: Option<usize>,
    zero_point: Option<i32>,
    granularity: Option<String>,
    data: Option<ByteRange>,
    input_transform: Option<InputTransform>,
}

#[derive(Deserialize)]
struct ByteRange {
    dtype: String,
    offset: usize,
    bytes: usize,
}

pub struct PackedTernaryTensor {
    pub shape: Vec<usize>,
    pub pad: usize,
    pub group_size: usize,
    pub packed: Vec<u32>,
    pub scales: Vec<f32>,
    pub offsets: Vec<f32>,
}

impl PackedTernaryTensor {
    pub fn decode_f32(&self) -> Result<Vec<f32>, String> {
        let values = checked_product(&self.shape)?;
        let mut output = Vec::with_capacity(values);
        for index in 0..values {
            let word = self.packed[index / TRITS_PER_WORD];
            let divisor = 3_u32.pow((index % TRITS_PER_WORD) as u32);
            let code = ((word / divisor) % 3) as f32 - 1.0;
            let group = index / self.group_size;
            let offset = self.offsets.get(group).copied().unwrap_or(0.0);
            output.push(code * self.scales[group] + offset);
        }
        Ok(output)
    }
}

pub struct RawTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

pub struct PackedSymmetricIntTensor {
    pub shape: Vec<usize>,
    pub bit_width: usize,
    pub zero_point: i32,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
}

impl PackedSymmetricIntTensor {
    pub fn decode_f32(&self) -> Result<Vec<f32>, String> {
        let values = checked_product(&self.shape)?;
        let outputs = self.shape[0];
        if self.scales.len() != outputs {
            return Err("symmetric integer scale count is inconsistent".to_string());
        }
        let values_per_output = values / outputs;
        let mut output = Vec::with_capacity(values);
        for index in 0..values {
            let bit_offset = index * self.bit_width;
            let mut unsigned = 0_u16;
            for bit in 0..self.bit_width {
                let source = bit_offset + bit;
                let value = (self.packed[source / 8] >> (source % 8)) & 1;
                unsigned |= u16::from(value) << bit;
            }
            let code = i32::from(unsigned) - self.zero_point;
            output.push(code as f32 * self.scales[index / values_per_output]);
        }
        Ok(output)
    }
}

pub struct BrowserArtifact<'a> {
    manifest: Manifest,
    blob: &'a [u8],
}

impl<'a> BrowserArtifact<'a> {
    pub fn from_bytes(manifest_json: &[u8], blob: &'a [u8]) -> Result<Self, String> {
        let manifest: Manifest = serde_json::from_slice(manifest_json)
            .map_err(|error| format!("invalid artifact manifest: {error}"))?;
        if manifest.format != FORMAT_V1
            && manifest.format != FORMAT_V2
            && manifest.format != FORMAT_V3
            && manifest.format != FORMAT_V4
        {
            return Err(format!("unsupported artifact format: {}", manifest.format));
        }
        if manifest.mode != "ternary" && manifest.mode != "mixed_ternary_int7" {
            return Err(format!("unsupported artifact mode: {}", manifest.mode));
        }
        if manifest.mode == "mixed_ternary_int7" && manifest.format != FORMAT_V4 {
            return Err("mixed ternary/int7 mode requires browser-v4".to_string());
        }
        if manifest.group_size == 0 {
            return Err("artifact group_size must be positive".to_string());
        }
        if let Some(transform) = &manifest.input_transform {
            validate_input_transform(transform)?;
        }
        for (name, tensor) in &manifest.tensors {
            if tensor.kind == "ternary" {
                let group_size = match (manifest.format.as_str(), tensor.group_size) {
                    (FORMAT_V2 | FORMAT_V3, None) => {
                        return Err(format!("tensor has no group_size: {name}"));
                    }
                    (_, Some(value)) => value,
                    (_, None) => manifest.group_size,
                };
                if group_size == 0 {
                    return Err(format!("tensor group_size must be positive: {name}"));
                }
            }
            if let Some(transform) = &tensor.input_transform {
                validate_input_transform(transform)
                    .map_err(|error| format!("{name}: {error}"))?;
            }
        }
        Ok(Self { manifest, blob })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.manifest.tensors.keys().map(String::as_str)
    }

    pub fn is_packed_ternary(&self, name: &str) -> Result<bool, String> {
        self.manifest
            .tensors
            .get(name)
            .map(|record| record.kind == "ternary")
            .ok_or_else(|| format!("tensor is absent from artifact: {name}"))
    }

    pub fn hadamard_block_size(&self, name: &str) -> Option<usize> {
        let transform = self
            .manifest
            .tensors
            .get(name)
            .and_then(|tensor| tensor.input_transform.as_ref())
            .or(self.manifest.input_transform.as_ref());
        transform.and_then(|transform| {
            (transform.kind == "block_hadamard")
                .then(|| transform.block_size)
                .flatten()
        })
    }

    pub fn packed_ternary(&self, name: &str) -> Result<PackedTernaryTensor, String> {
        let record = self
            .manifest
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor is absent from artifact: {name}"))?;
        if record.kind != "ternary" {
            return Err(format!("tensor is not ternary: {name}"));
        }
        let values = checked_product(&record.shape)?;
        let expected_pad =
            (TRITS_PER_WORD - values % TRITS_PER_WORD) % TRITS_PER_WORD;
        let pad = record
            .pad
            .ok_or_else(|| format!("tensor has no trit padding: {name}"))?;
        if pad != expected_pad {
            return Err(format!("tensor trit padding is inconsistent: {name}"));
        }
        let codes = record
            .codes
            .as_ref()
            .ok_or_else(|| format!("tensor has no packed codes: {name}"))?;
        let group_size = record.group_size.unwrap_or(self.manifest.group_size);
        if codes.dtype != "uint32" {
            return Err(format!("tensor code dtype is not uint32: {name}"));
        }
        let expected_code_bytes = values
            .div_ceil(TRITS_PER_WORD)
            .checked_mul(4)
            .ok_or_else(|| format!("tensor code size overflows: {name}"))?;
        if codes.bytes != expected_code_bytes {
            return Err(format!("tensor code payload size is inconsistent: {name}"));
        }
        let code_bytes = checked_slice(self.blob, codes, name)?;
        let packed = code_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect();
        let scale_count = values.div_ceil(group_size);
        let scales = if let Some(range) = &record.scales {
            if range.dtype != "float16" || range.bytes != scale_count * 2 {
                return Err(format!("tensor FP16 scale payload is inconsistent: {name}"));
            }
            decode_fp16(checked_slice(self.blob, range, name)?)?
        } else {
            if self.manifest.format != FORMAT_V3
                || record.scale_codec.as_deref() != Some("codebook_fp16")
                || record.scale_count != Some(scale_count)
            {
                return Err(format!("tensor scale-codebook contract is inconsistent: {name}"));
            }
            let bits = record
                .scale_index_bits
                .ok_or_else(|| format!("tensor has no scale index width: {name}"))?;
            let codebook = record
                .scale_codebook
                .as_ref()
                .ok_or_else(|| format!("tensor has no scale codebook: {name}"))?;
            let indices = record
                .scale_indices
                .as_ref()
                .ok_or_else(|| format!("tensor has no scale indices: {name}"))?;
            if codebook.dtype != "float16" || indices.dtype != "packed_uint" {
                return Err(format!("tensor scale-codebook dtype is inconsistent: {name}"));
            }
            decode_scale_codebook(
                checked_slice(self.blob, codebook, name)?,
                checked_slice(self.blob, indices, name)?,
                scale_count,
                bits,
            )?
        };
        let offsets = if let Some(range) = &record.offsets {
            if range.dtype != "float16" || range.bytes != scale_count * 2 {
                return Err(format!("tensor FP16 offset payload is inconsistent: {name}"));
            }
            decode_fp16(checked_slice(self.blob, range, name)?)?
        } else {
            Vec::new()
        };
        Ok(PackedTernaryTensor {
            shape: record.shape.clone(),
            pad,
            group_size,
            packed,
            scales,
            offsets,
        })
    }

    pub fn raw_f32(&self, name: &str) -> Result<RawTensor, String> {
        let record = self
            .manifest
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor is absent from artifact: {name}"))?;
        if record.kind != "raw" {
            return Err(format!("tensor is not raw: {name}"));
        }
        let values = checked_product(&record.shape)?;
        let data = record
            .data
            .as_ref()
            .ok_or_else(|| format!("tensor has no raw payload: {name}"))?;
        if data.dtype != "float16" {
            return Err(format!("raw tensor dtype is not float16: {name}"));
        }
        let expected_bytes = values
            .checked_mul(2)
            .ok_or_else(|| format!("tensor raw size overflows: {name}"))?;
        if data.bytes != expected_bytes {
            return Err(format!("tensor raw payload size is inconsistent: {name}"));
        }
        let bytes = checked_slice(self.blob, data, name)?;
        let values = decode_fp16(bytes)?;
        Ok(RawTensor {
            shape: record.shape.clone(),
            values,
        })
    }

    pub fn packed_symmetric_int(
        &self,
        name: &str,
    ) -> Result<PackedSymmetricIntTensor, String> {
        let record = self
            .manifest
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor is absent from artifact: {name}"))?;
        if record.kind != "symmetric_int" {
            return Err(format!("tensor is not symmetric integer: {name}"));
        }
        if self.manifest.format != FORMAT_V4 {
            return Err(format!("symmetric integer tensor requires browser-v4: {name}"));
        }
        let values = checked_product(&record.shape)?;
        let bit_width = record
            .bit_width
            .ok_or_else(|| format!("tensor has no integer bit width: {name}"))?;
        if bit_width < 2 || bit_width > 8 {
            return Err(format!("tensor integer bit width is unsupported: {name}"));
        }
        if record.granularity.as_deref() != Some("per_output") {
            return Err(format!("tensor integer granularity is unsupported: {name}"));
        }
        let zero_point = record
            .zero_point
            .ok_or_else(|| format!("tensor has no integer zero point: {name}"))?;
        let expected_zero = (1_i32 << (bit_width - 1)) - 1;
        if zero_point != expected_zero {
            return Err(format!("tensor integer zero point is inconsistent: {name}"));
        }
        let codes = record
            .codes
            .as_ref()
            .ok_or_else(|| format!("tensor has no packed integer codes: {name}"))?;
        let expected_code_bytes = (values * bit_width).div_ceil(8);
        if codes.dtype != "packed_uint" || codes.bytes != expected_code_bytes {
            return Err(format!("tensor packed integer payload is inconsistent: {name}"));
        }
        let scales = record
            .scales
            .as_ref()
            .ok_or_else(|| format!("tensor has no integer scales: {name}"))?;
        let scale_count = record.shape[0];
        if scales.dtype != "float16" || scales.bytes != scale_count * 2 {
            return Err(format!("tensor integer scale payload is inconsistent: {name}"));
        }
        Ok(PackedSymmetricIntTensor {
            shape: record.shape.clone(),
            bit_width,
            zero_point,
            packed: checked_slice(self.blob, codes, name)?.to_vec(),
            scales: decode_fp16(checked_slice(self.blob, scales, name)?)?,
        })
    }

    pub fn tensor_f32(&self, name: &str) -> Result<RawTensor, String> {
        let record = self
            .manifest
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor is absent from artifact: {name}"))?;
        match record.kind.as_str() {
            "raw" => self.raw_f32(name),
            "ternary" => {
                let tensor = self.packed_ternary(name)?;
                Ok(RawTensor {
                    shape: tensor.shape.clone(),
                    values: tensor.decode_f32()?,
                })
            }
            "symmetric_int" => {
                let tensor = self.packed_symmetric_int(name)?;
                Ok(RawTensor {
                    shape: tensor.shape.clone(),
                    values: tensor.decode_f32()?,
                })
            }
            kind => Err(format!("unsupported tensor kind for f32 decode: {kind}")),
        }
    }
}

fn validate_input_transform(transform: &InputTransform) -> Result<(), String> {
    match transform.kind.as_str() {
        "none" => Ok(()),
        "block_hadamard" => {
            let block = transform
                .block_size
                .ok_or_else(|| "Hadamard transform has no block size".to_string())?;
            if block == 0 || !block.is_power_of_two() {
                return Err("Hadamard block size must be a positive power of two".into());
            }
            Ok(())
        }
        kind => Err(format!("unsupported artifact input transform: {kind}")),
    }
}

fn checked_product(shape: &[usize]) -> Result<usize, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err("tensor shape must contain positive dimensions".to_string());
    }
    shape.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| "tensor shape overflows".to_string())
    })
}

fn checked_slice<'a>(
    blob: &'a [u8],
    range: &ByteRange,
    name: &str,
) -> Result<&'a [u8], String> {
    let end = range
        .offset
        .checked_add(range.bytes)
        .ok_or_else(|| format!("tensor byte range overflows: {name}"))?;
    blob.get(range.offset..end)
        .ok_or_else(|| format!("tensor byte range exceeds artifact: {name}"))
}
