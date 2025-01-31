#[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
use core::arch::x86_64::{__m128i, __m256};
use std::simd::{
    cmp::{SimdPartialEq, SimdPartialOrd},
    f32x8, f64x8, masksizex8,
    num::{SimdFloat, SimdUint},
    u16x8, usizex8, Simd, StdFloat,
};

use rayon::prelude::*;

use crate::vectors::Vectors;

// i < j, i != j
#[inline]
pub fn index_to_offset(n: usize, i: usize, j: usize) -> usize {
    let i_f64 = i as f64;
    let j_f64 = j as f64;
    let n_f64 = n as f64;
    let correction = (i_f64 + 2.0) * (i_f64 + 1.0) / 2.0;
    (i_f64 * n_f64 + j_f64 - correction) as usize
}

// offset = i*n - (i + 2) * (i + 1) / 2 + j
//
#[inline]
pub fn offset_to_index(n: usize, offset: usize) -> (usize, usize) {
    let d = (2 * n - 1).pow(2) - 8 * offset;
    let i2 = (2 * n - 1) as f64 - (d as f64).sqrt();
    let i = (i2 / 2.0) as usize;
    let triangle = (i * (n - 1)) - ((i + 1) * i) / 2;
    let j = offset + 1 - triangle;
    (i, j)
}

#[inline]
pub fn indexes_to_offsets(n: usize, i: usizex8, j: usizex8) -> usizex8 {
    let i_f64: f64x8 = i.cast();
    let j_f64 = j.cast();
    let n_f64 = f64x8::splat(n as f64);
    let correction = (i_f64 + f64x8::splat(2.0)) * (i_f64 + f64x8::splat(1.0)) / f64x8::splat(2.0);
    (i_f64 * n_f64 + j_f64 - correction).cast()
}

#[inline]
pub fn offsets_to_indexes(n: usize, offsets: usizex8) -> (usizex8, usizex8) {
    let n = usizex8::splat(n);
    let d_root = usizex8::splat(2) * n - usizex8::splat(1);
    let d = d_root * d_root - usizex8::splat(8) * offsets;
    let i2 = (usizex8::splat(2) * n - usizex8::splat(1)).cast::<f64>() - d.cast::<f64>().sqrt();
    let i: usizex8 = (i2 / f64x8::splat(2.0)).cast();
    let triangle =
        (i * (n - usizex8::splat(1))) - ((i + usizex8::splat(1)) * i) / usizex8::splat(2);
    let j = offsets + usizex8::splat(1) - triangle;
    (i, j)
}

#[inline]
pub fn triangle_lookup_length(n: usize) -> usize {
    index_to_offset(n, n - 2, n - 1) + 1
}

pub trait CentroidDistanceCalculator: Sync {
    fn num_centroids(&self) -> usize;
    fn calculate_partial_dot_product(&self, c1: u16, c2: u16) -> f16;
    fn calculate_partial_dot_product_norm(&self, c: u16) -> f16;
}

pub trait CentroidDistanceCalculatorConstructor {
    type Calculator<'a>: CentroidDistanceCalculator
    where
        Self: 'a;
    fn new(vectors: &Vectors) -> Self::Calculator<'_>;
}

pub struct MemoizedCentroidDistances {
    dot_products: Vec<f16>,
    norms: Vec<f16>,
    size: usize,
}

impl MemoizedCentroidDistances {
    pub fn new<C: CentroidDistanceCalculator>(calculator: &C) -> Self {
        let size = calculator.num_centroids();
        eprintln!("constructing memoized");
        let memoized_array_length = triangle_lookup_length(size);
        eprintln!(
            "for size {} we calculate an array length of {memoized_array_length}",
            size
        );
        let mut distances: Vec<f16> = Vec::with_capacity(memoized_array_length);
        {
            let partial_distances_uninit = distances.spare_capacity_mut();
            partial_distances_uninit
                .par_iter_mut()
                .enumerate()
                .for_each(|(c, elt)| {
                    let (i, j) = offset_to_index(size, c);
                    if i > 65535 || j > 65535 {
                        panic!("oh no {i} {j}");
                    }
                    elt.write(calculator.calculate_partial_dot_product(i as u16, j as u16));
                });
        }
        unsafe {
            distances.set_len(memoized_array_length);
        }
        let norms: Vec<_> = (0..size)
            .map(|i| calculator.calculate_partial_dot_product_norm(i as u16))
            .collect();
        Self {
            dot_products: distances,
            norms,
            size,
        }
    }

    #[inline]
    pub fn lookup_centroid_distance(&self, i: u16, j: u16) -> f16 {
        let offset = match i.cmp(&j) {
            std::cmp::Ordering::Equal => {
                // Early bail
                return self.lookup_centroid_squared_norm(i);
            }
            std::cmp::Ordering::Less => index_to_offset(self.size, i as usize, j as usize),
            std::cmp::Ordering::Greater => index_to_offset(self.size, j as usize, i as usize),
        };
        let distance: f16 = self.dot_products[offset];
        distance
    }

    #[inline]
    pub fn lookup_centroid_squared_norm(&self, i: u16) -> f16 {
        self.norms[i as usize]
    }

    #[inline]
    pub fn lookup_centroid_dot_products(&self, i: u16x8, j: u16x8) -> f32x8 {
        let equals_mask = i.simd_eq(j);
        let norms = self.lookup_centroid_partial_norms_masked(i, equals_mask.cast());

        let less_mask = i.simd_lt(j);
        // gotta flip the greaters with the lessers
        let temp = i;
        let i = less_mask.select(i, j);
        let j = less_mask.select(j, temp);

        let offsets = indexes_to_offsets(self.size, i.cast(), j.cast());
        let dot_products_slice: &[u16] =
            unsafe { std::mem::transmute(self.dot_products.as_slice()) };
        let gathered: Simd<u16, 8> = u16x8::gather_select(
            dot_products_slice,
            (!equals_mask).cast(),
            offsets.cast(),
            u16x8::splat(0),
        );
        let result = from_u16x8_to_f32x8(gathered.into());
        let partial_dot_products = f32x8::from(result);

        // we now have two simd registers with mutually exclusive lanes filled.
        // summing them should just give us a single register with all lanes filled.
        norms + partial_dot_products
    }

    #[inline]
    pub fn lookup_centroid_squared_norms(&self, i: u16x8) -> f32x8 {
        let i: usizex8 = i.cast();
        let norms_slice: &[u16] = unsafe { std::mem::transmute(self.norms.as_slice()) };
        let gathered = u16x8::gather_or_default(norms_slice, i);
        let result = from_u16x8_to_f32x8(gathered);
        f32x8::from(result)
    }

    #[inline]
    #[allow(unused)]
    fn lookup_centroid_partial_norms_masked(&self, i: u16x8, mask: masksizex8) -> f32x8 {
        let i: usizex8 = i.cast();
        let norms_slice: &[u16] = unsafe { std::mem::transmute(self.norms.as_slice()) };
        let gathered = u16x8::gather_select(norms_slice, mask, i, u16x8::splat(0));
        let result = from_u16x8_to_f32x8(gathered);
        f32x8::from(result)
    }
}

#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
fn from_u16x8_to_f32x8(src: Simd<u16, 8>) -> __m256 {
    unsafe { std::arch::x86_64::_mm256_cvtph_ps(src.into()) }
}

#[inline]
#[cfg(all(target_arch = "aarch64", target_feature  = "neon"))]
fn from_u16x8_to_f32x8(src: Simd<u16, 8>) -> Simd<f32, 8> {
    use core::arch::aarch64::{
        uint16x4_t, uint16x8_t, uint32x4_t, float32x4_t,
        vget_high_u16, vget_low_u16, vmovl_u16, vcvtq_f32_u32
    };
    use std::simd::f32x4;

    #[inline(always)]
    unsafe fn extract_blocks(src: uint16x8_t) -> (uint16x4_t, uint16x4_t) {
        (vget_high_u16(src), vget_low_u16(src))
    }

    #[inline(always)]
    unsafe fn simdify_block(block: uint16x4_t) -> Simd<f32, 4> {
        let block: uint32x4_t = vmovl_u16(block); // widen each scalar
        let block: float32x4_t = vcvtq_f32_u32(block); // floatify
        block.into()
    }

    let src: uint16x8_t = src.into();
    unsafe { // Process as `high` and `low` blocks of values, each of 4 elements
        let (h, l): (uint16x4_t, uint16x4_t) = extract_blocks(src);
        let (h, l): (f32x4, f32x4) = (simdify_block(h), simdify_block(l));
        let array: [[f32; 4]; 2] = [h.to_array(), l.to_array()];
        let array: [f32; 8] = std::mem::transmute(array);
        Simd::<f32, 8>::from_array(array)
    }
}

#[inline]
#[cfg(not(any( // default non-SIMD implementation
    all(target_arch = "x86_64", target_feature = "f16c"),
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
fn from_u16x8_to_f32x8(src: Simd<u16, 8>) -> Simd<f32, 8> {
    let src: &[u16; 8] = src.as_array();
    let mut dst = [0_f32; 8];
    for i in 0..8 {
        // Slow but safe. Cannot use `std::mem::transmute()`
        // because `src` and `dst` have different sizes.
        dst[i] = f32::from(src[i]);
    }
    Simd::<f32, 8>::from_array(dst)
}


#[cfg(test)]
mod offsettest {
    use super::*;
    use rand::prelude::*;
    #[test]
    fn test_triangle_offsets() {
        let n = 100;
        let mut expected_index = 0;
        for i in 0..n {
            for j in 0..n {
                if i < j {
                    let actual = index_to_offset(n, i, j);
                    assert_eq!(expected_index, actual);
                    expected_index += 1;
                }
            }
        }
        assert_eq!(expected_index, triangle_lookup_length(n));
    }

    #[test]
    #[ignore]
    fn roundtrip() {
        let n = 65535;
        (0..triangle_lookup_length(n))
            .into_par_iter()
            .for_each(|i| {
                let (a, b) = offset_to_index(n, i);
                if a >= n {
                    eprintln!("Yikes: a: {a}, b: {b}, n: {n}");
                }
                if b >= n {
                    eprintln!("Yikes: a: {a}, b: {b}, n: {n}");
                }
                assert!(a < n);
                assert!(b < n);

                if a == 0 && b == 0 {
                    panic!("Failure at {i}: a: {a}, b: {b}, n: {n}");
                }
                let i2 = index_to_offset(n, a, b);
                if i != i2 {
                    panic!("Failure n: {n}, a: {a}, b: {b}, i: {i}, i2: {i2}");
                }
                assert_eq!(i, i2);
            });
    }

    #[test]
    #[ignore]
    fn roundtrip_backwards() {
        let n = 65535;
        (0..n)
            .into_par_iter()
            .flat_map(|a| (0..n).into_par_iter().map(move |b| (a, b)))
            .for_each(|(a, b)| {
                if a >= b {
                    return;
                }
                let i = index_to_offset(n, a, b);
                let (a2, b2) = offset_to_index(n, i);
                if a2 != a || b2 != b {
                    panic!("omfg: a: {a}, b: {b}, a2: {a2}, b2: {b2}, i: {i}")
                }
            });
    }

    struct IndexProductDistanceCalculator;

    fn scaled_multiple(left: usize, right: usize) -> f16 {
        const SCALE: f32 = (u32::MAX as usize + 1) as f32 / 16.0;
        ((left * right) as f32 / SCALE) as f16
    }

    impl CentroidDistanceCalculator for IndexProductDistanceCalculator {
        fn num_centroids(&self) -> usize {
            (u16::MAX as usize) + 1
        }
        fn calculate_partial_dot_product(&self, left: u16, right: u16) -> f16 {
            scaled_multiple(left as usize, right as usize)
        }
        fn calculate_partial_dot_product_norm(&self, vec: u16) -> f16 {
            let vec = vec as usize;
            scaled_multiple(vec, vec)
        }
    }

    #[test]
    #[ignore]
    fn scalar_distances_are_mapped_right() {
        let distances = MemoizedCentroidDistances::new(&IndexProductDistanceCalculator);
        let mut rng = StdRng::seed_from_u64(2024);
        let mut set1: Vec<_> = (0..=u16::MAX).collect();
        set1.shuffle(&mut rng);
        set1.truncate(1000);
        set1.shrink_to_fit();
        let mut set2: Vec<_> = (0..=u16::MAX).collect();
        set2.shuffle(&mut rng);
        set2.truncate(1000);
        set2.shrink_to_fit();
        set1.into_par_iter()
            .flat_map(|a| set2.par_iter().map(move |b| (a, *b)))
            .for_each(|(a, b)| {
                if a == b {
                    return;
                }
                let distance = distances.lookup_centroid_distance(a, b) as f32;
                let calculated_distance = scaled_multiple(a as usize, b as usize) as f32;
                let error = (distance - calculated_distance).abs();
                if error > 0.1 {
                    panic!("{a},{b} gave distance {distance} and not {calculated_distance}");
                }
            });
    }

    #[test]
    #[ignore]
    fn simd_distances_are_mapped_right() {
        let distances = MemoizedCentroidDistances::new(&IndexProductDistanceCalculator);
        let mut rng = StdRng::seed_from_u64(2024);
        let mut set1: Vec<_> = (0..=u16::MAX).collect();
        assert!(set1.len() % 8 == 0);
        set1.shuffle(&mut rng);
        set1.truncate(1000);
        set1.shrink_to_fit();
        let mut set2: Vec<_> = (0..=u16::MAX).collect();
        set2.shuffle(&mut rng);
        set2.truncate(1000);
        set2.shrink_to_fit();
        set1.par_chunks(8)
            .flat_map(|a| set2.par_chunks(8).map(move |b| (a, b)))
            .for_each(|(a, b)| {
                let simd_a = u16x8::from_slice(a);
                let simd_b = u16x8::from_slice(b);
                let distances = distances.lookup_centroid_dot_products(simd_a, simd_b);

                for (distance, (a, b)) in
                    distances.to_array().into_iter().zip(a.iter().zip(b.iter()))
                {
                    if a == b {
                        continue;
                    }
                    let calculated_distance = scaled_multiple(*a as usize, *b as usize) as f32;
                    let error = (distance - calculated_distance).abs();
                    if error > 0.1 {
                        panic!("{a},{b} gave distance {distance} and not {calculated_distance}");
                    }
                }
            });
    }
}
