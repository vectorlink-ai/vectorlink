use std::simd::StdFloat;
use std::simd::{f32x16, f32x32, f32x64, f32x8, num::SimdFloat};

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
