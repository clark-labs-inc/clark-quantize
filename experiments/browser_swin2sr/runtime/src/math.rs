//! Small vectorizable kernels shared by packed and dense operators.

#[inline]
pub fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm_dot_product(left, right);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        scalar_dot_product(left, right)
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn scalar_dot_product(left: &[f32], right: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; 4];
    let paired = left.chunks_exact(4).zip(right.chunks_exact(4));
    for (lhs, rhs) in paired {
        lanes[0] += lhs[0] * rhs[0];
        lanes[1] += lhs[1] * rhs[1];
        lanes[2] += lhs[2] * rhs[2];
        lanes[3] += lhs[3] * rhs[3];
    }
    let remainder_start = left.len() / 4 * 4;
    let remainder = left[remainder_start..]
        .iter()
        .zip(&right[remainder_start..])
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>();
    lanes.into_iter().sum::<f32>() + remainder
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn wasm_dot_product(left: &[f32], right: &[f32]) -> f32 {
    use core::arch::wasm32::{
        f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128_load,
    };

    let vector_count = left.len() / 4;
    let mut accumulator = f32x4_splat(0.0);
    for vector in 0..vector_count {
        let offset = vector * 4;
        let lhs = unsafe { v128_load(left.as_ptr().add(offset) as *const _) };
        let rhs = unsafe { v128_load(right.as_ptr().add(offset) as *const _) };
        accumulator = f32x4_add(accumulator, f32x4_mul(lhs, rhs));
    }
    let mut sum = f32x4_extract_lane::<0>(accumulator)
        + f32x4_extract_lane::<1>(accumulator)
        + f32x4_extract_lane::<2>(accumulator)
        + f32x4_extract_lane::<3>(accumulator);
    for index in vector_count * 4..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::dot_product;

    #[test]
    fn dot_product_handles_vector_and_remainder_lanes() {
        let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right = [0.5, -1.0, 2.0, 0.25, -0.5, 3.0];
        assert!((dot_product(&left, &right) - 21.0).abs() < 1e-6);
    }
}
