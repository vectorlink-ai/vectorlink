use std::simd::{f32x16, f32x32, f32x64, f32x8, num::SimdFloat};
use std::simd::{LaneCount, Simd, StdFloat, SupportedLaneCount};

use unroll::unroll_for_loops;

pub const PRIMES: [usize; 43] = [
    1, 2, 59063, 79193, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
];

// both unroll and (especially) inlining drastically improve speed
// when calling comparisons in a loop.
// tests indicate,
// - unroll: 10% speedup
// - inline: 1000% speedup
macro_rules! dot_product_n_16 {
    ($name:ident, $n:literal) => {
        #[unroll_for_loops]
        #[inline(always)]
        pub fn $name(left: &[f32], right: &[f32]) -> f32 {
            let mut sum = <f32x16>::splat(0.);
            for x in 0..$n {
                let l = <f32x16>::from_slice(&left[x * 16..(x + 1) * 16]);
                let r = <f32x16>::from_slice(&right[x * 16..(x + 1) * 16]);
                sum = l.mul_add(r, sum);
            }
            sum.reduce_sum()
        }
    };
}
macro_rules! dot_product_n_8 {
    ($name:ident, $n:literal) => {
        #[unroll_for_loops]
        #[inline(always)]
        pub fn $name(left: &[f32], right: &[f32]) -> f32 {
            let mut sum = <f32x8>::splat(0.);
            for x in 0..$n {
                let l = <f32x8>::from_slice(&left[x * 8..(x + 1) * 8]);
                let r = <f32x8>::from_slice(&right[x * 8..(x + 1) * 8]);
                sum = l.mul_add(r, sum);
            }
            sum.reduce_sum()
        }
    };
}
macro_rules! dot_product_n_32 {
    ($name:ident, $n:literal) => {
        #[unroll_for_loops]
        #[inline(always)]
        pub fn $name(left: &[f32], right: &[f32]) -> f32 {
            let mut sum = <f32x32>::splat(0.);
            for x in 0..$n {
                let l = <f32x32>::from_slice(&left[x * 32..(x + 1) * 32]);
                let r = <f32x32>::from_slice(&right[x * 32..(x + 1) * 32]);
                sum = l.mul_add(r, sum);
            }
            sum.reduce_sum()
        }
    };
}
macro_rules! dot_product_n_64 {
    ($name:ident, $n:literal) => {
        #[unroll_for_loops]
        #[inline(always)]
        pub fn $name(left: &[f32], right: &[f32]) -> f32 {
            let mut sum = <f32x64>::splat(0.);
            for x in 0..$n {
                let l = <f32x64>::from_slice(&left[x * 64..(x + 1) * 64]);
                let r = <f32x64>::from_slice(&right[x * 64..(x + 1) * 64]);
                sum = l.mul_add(r, sum);
            }
            sum.reduce_sum()
        }
    };
}

dot_product_n_8!(dot_product_1536_8, 192);
dot_product_n_8!(dot_product_1024_8, 128);
dot_product_n_16!(dot_product_1536_16, 96);
dot_product_n_16!(dot_product_1024_16, 64);
dot_product_n_32!(dot_product_1536_32, 48);
dot_product_n_32!(dot_product_1024_32, 32);
dot_product_n_64!(dot_product_1536_64, 24);
dot_product_n_64!(dot_product_1024_64, 16);

macro_rules! dot_product_small_64 {
    ($name:ident, $n:literal) => {
        #[unroll_for_loops]
        #[inline(always)]
        pub fn $name(left: &[f32], right: &[f32], results: &mut [f32]) {
            let l = <f32x64>::from_slice(&left[0..64]);
            let r = <f32x64>::from_slice(&right[0..64]);
            let sum = l * r;

            for i in 0..(64 / $n) {
                let offset = i * $n;
                let partial: Simd<f32, $n> = Simd::from_slice(&sum.as_array()[offset..offset + $n]);
                results[i] = partial.reduce_sum();
            }
        }
    };
}

dot_product_small_64!(multi_dot_product_2, 8);

macro_rules! euclidean_small_64 {
    ($name:ident, $n:literal) => {
        #[unroll_for_loops]
        #[inline(always)]
        pub fn $name(left: &[f32], right: &[f32], results: &mut [f32]) {
            let l = <f32x64>::from_slice(&left[0..64]);
            let mut r_arr = [0.0_f32; 64];
            for i in 0..(64 / $n) {
                let offset = i * $n;
                r_arr[offset..offset + $n].copy_from_slice(right);
            }
            let r = <f32x64>::from_array(r_arr);
            let dif = (l - r);
            let squared_dif = dif * dif;

            for i in 0..(64 / $n) {
                let offset = i * $n;
                let partial: Simd<f32, $n> =
                    Simd::from_slice(&squared_dif.as_array()[offset..offset + $n]);
                results[i] = partial.reduce_sum().sqrt();
            }
        }
    };
}

euclidean_small_64!(multi_euclidean_4x16, 16);
euclidean_small_64!(multi_euclidean_8x8, 8);
euclidean_small_64!(multi_euclidean_16x4, 4);

pub fn partial_dot_product<const N: usize>(left: &[f32; N], right: &[f32; N]) -> f32
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut left_simd: Simd<f32, N> = Simd::from_slice(left);
    let right_simd: Simd<f32, N> = Simd::from_slice(right);
    left_simd *= right_simd;

    left_simd.reduce_sum()
}

pub fn partial_dot_product_norm<const N: usize>(vec: &[f32; N]) -> f32
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut vec_simd: Simd<f32, N> = Simd::from_slice(vec);
    vec_simd *= vec_simd;

    vec_simd.reduce_sum()
}

macro_rules! normalize_aligned_n_64 {
    ($name:ident, $n:literal) => {
        #[unroll_for_loops]
        #[inline(always)]
        pub fn $name(vec: &mut [f32]) {
            let (prefix, simd, suffix) = vec.as_simd_mut::<64>();
            debug_assert_eq!(prefix.len(), 0);
            debug_assert_eq!(suffix.len(), 0);
            debug_assert_eq!(simd.len(), 64 * $n);

            let mut accumulator = f32x64::splat(0.0);
            for i in 0..$n {
                accumulator = simd[i].mul_add(simd[i], accumulator);
            }
            let size = accumulator.reduce_sum();
            let size_simd = f32x64::splat(size);
            for i in 0..$n {
                simd[i] /= size_simd;
            }
        }
    };
}

normalize_aligned_n_64!(normalize_aligned_1024, 16);
normalize_aligned_n_64!(normalize_aligned_1536, 24);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_8_with_other_is_not_0() {
        let left: Vec<_> = (0..64).map(|i| i as f32).collect();
        let right: Vec<_> = (0..8).map(|i| i as f32).collect();
        let mut results = [0.0; 8];

        multi_euclidean_8x8(&left, &right, &mut results);

        let expected: Vec<_> = (0..8)
            .map(|i| ((i as f32 * 8.0).powi(2) * 8.0).sqrt())
            .collect();
        assert_eq!(expected, results);
    }
}
