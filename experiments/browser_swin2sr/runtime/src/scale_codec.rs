pub(crate) fn decode_fp16(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 2 != 0 {
        return Err("FP16 payload has an odd byte length".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            f16_to_f32(u16::from_le_bytes(
                chunk.try_into().expect("two-byte chunk"),
            ))
        })
        .collect())
}

pub(crate) fn decode_scale_codebook(
    center_bytes: &[u8],
    index_bytes: &[u8],
    count: usize,
    bit_width: usize,
) -> Result<Vec<f32>, String> {
    if !(1..=8).contains(&bit_width) {
        return Err("scale codebook bit width must be between one and eight".into());
    }
    let centers = decode_fp16(center_bytes)?;
    if centers.is_empty() || centers.len() > (1_usize << bit_width) {
        return Err("scale codebook level count is inconsistent".into());
    }
    let expected_index_bytes = count
        .checked_mul(bit_width)
        .ok_or("scale index size overflows")?
        .div_ceil(8);
    if index_bytes.len() != expected_index_bytes {
        return Err("scale codebook index size is inconsistent".into());
    }
    (0..count)
        .map(|index| {
            let bit_offset = index * bit_width;
            let mut code = 0_usize;
            for bit in 0..bit_width {
                let encoded = (index_bytes[(bit_offset + bit) / 8]
                    >> ((bit_offset + bit) % 8))
                    & 1;
                code |= (encoded as usize) << bit;
            }
            centers
                .get(code)
                .copied()
                .ok_or_else(|| "scale codebook index exceeds level count".into())
        })
        .collect()
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = 31 - fraction.leading_zeros();
            let shift = 10 - leading;
            let normalized = (fraction << shift) & 0x03ff;
            let adjusted_exponent = 113_u32 - shift;
            sign | (adjusted_exponent << 23) | (normalized << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

#[cfg(test)]
mod tests {
    use super::{decode_scale_codebook, f16_to_f32};

    #[test]
    fn decodes_fp16_normal_subnormal_and_special_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x0001), 2_f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn decodes_lsb_first_unaligned_scale_indices() {
        let centers = [0x3800_u16, 0x3c00, 0x4000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let decoded = decode_scale_codebook(&centers, &[0b10_01_00], 3, 2).unwrap();
        assert_eq!(decoded, [0.5, 1.0, 2.0]);
    }
}
