use std::simd::{f32x16, num::SimdFloat};

use unroll::unroll_for_loops;

macro_rules! dot_product_n {
    ($name:ident, $n:literal) => {
        #[unroll_for_loops]
        pub fn $name(left: &[f32], right: &[f32]) -> f32 {
            let mut sum = <f32x16>::splat(0.);
            for x in 0..$n {
                let l = <f32x16>::from_slice(&left[x * 16..(x + 1) * 16]);
                let r = <f32x16>::from_slice(&right[x * 16..(x + 1) * 16]);
                sum += l * r;
            }
            sum.reduce_sum()
        }
    };
}

dot_product_n!(dot_product_1536, 96);
dot_product_n!(dot_product_1024, 64);
