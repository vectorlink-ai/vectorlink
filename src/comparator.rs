use std::simd::{f32x8, num::SimdFloat, u16x8};

use unroll::unroll_for_loops;

use crate::{
    layer::VectorComparator,
    memoize::{CentroidDistanceCalculator, MemoizedCentroidDistances},
    vecmath,
    vectors::Vectors,
};

pub struct CosineDistance1024<'a> {
    vectors: &'a Vectors,
}

impl<'a> CosineDistance1024<'a> {
    pub fn new(vectors: &'a Vectors) -> Self {
        Self { vectors }
    }
}

impl<'a> VectorComparator for CosineDistance1024<'a> {
    #[inline(always)]
    fn num_vecs(&self) -> usize {
        self.vectors.num_vecs()
    }

    #[inline(always)]
    fn vector_byte_size() -> usize {
        1024 * std::mem::size_of::<f32>()
    }

    #[inline(always)]
    fn compare_vecs_stored(&self, left: &[u32], right: u32, result: &mut [f32]) {
        let left = left[0];
        result[0] = self.compare_vec_stored(left, right);
    }

    #[inline(always)]
    fn compare_vecs_stored_unstored(&self, stored: &[u32], unstored: &[u8], result: &mut [f32]) {
        let stored = stored[0];
        result[0] = self.compare_vec_stored_unstored(stored, unstored);
    }

    #[inline(always)]
    fn compare_vecs_unstored(&self, left: &[u8], right: &[u8], result: &mut [f32]) {
        result[0] = self.compare_vec_unstored(left, right);
    }

    #[inline(always)]
    fn vec_group_size() -> usize {
        1
    }

    #[inline(always)]
    fn compare_vec_stored(&self, left: u32, right: u32) -> f32 {
        if left == right {
            return 0.0;
        }
        if let Some(left) = self.vectors.get::<[f32; 1024]>(left as usize) {
            let right: &[f32; 1024] = self
                .vectors
                .get(right as usize)
                .expect("You are comparing to an out-of-band id");
            let product = vecmath::dot_product_1024_64(left, right);

            ((product - 1.0) / -2.0).clamp(0.0, 1.0)
        } else {
            f32::MAX
        }
    }

    #[inline(always)]
    fn compare_vec_stored_unstored(&self, stored: u32, unstored: &[u8]) -> f32 {
        debug_assert_eq!(unstored.len(), 4096);
        if let Some(stored) = self.vectors.get::<[f32; 1024]>(stored as usize) {
            let unstored: &[f32; 1024] = &unsafe { *(unstored.as_ptr() as *const [f32; 1024]) };
            let product = vecmath::dot_product_1024_64(stored, unstored);

            ((product - 1.0) / -2.0).clamp(0.0, 1.0)
        } else {
            f32::MAX
        }
    }

    #[inline(always)]
    fn compare_vec_unstored(&self, left: &[u8], right: &[u8]) -> f32 {
        debug_assert_eq!(left.len(), 4096);
        debug_assert_eq!(right.len(), 4096);
        let left: &[f32; 1024] = &unsafe { *(left.as_ptr() as *const [f32; 1024]) };
        let right: &[f32; 1024] = &unsafe { *(right.as_ptr() as *const [f32; 1024]) };
        let product = vecmath::dot_product_1024_64(left, right);

        ((product - 1.0) / -2.0).clamp(0.0, 1.0)
    }
}

pub struct CosineDistance1536<'a> {
    vectors: &'a Vectors,
}

impl<'a> CosineDistance1536<'a> {
    pub fn new(vectors: &'a Vectors) -> Self {
        Self { vectors }
    }
}

impl<'a> VectorComparator for CosineDistance1536<'a> {
    #[inline(always)]
    fn num_vecs(&self) -> usize {
        self.vectors.num_vecs()
    }

    #[inline(always)]
    fn vector_byte_size() -> usize {
        1536 * std::mem::size_of::<f32>()
    }

    #[inline(always)]
    fn compare_vecs_stored(&self, left: &[u32], right: u32, result: &mut [f32]) {
        let left = left[0];
        result[0] = self.compare_vec_stored(left, right);
    }

    #[inline(always)]
    fn compare_vecs_stored_unstored(&self, stored: &[u32], unstored: &[u8], result: &mut [f32]) {
        let stored = stored[0];
        result[0] = self.compare_vec_stored_unstored(stored, unstored);
    }

    #[inline(always)]
    fn compare_vecs_unstored(&self, left: &[u8], right: &[u8], result: &mut [f32]) {
        result[0] = self.compare_vec_unstored(left, right);
    }

    #[inline(always)]
    fn vec_group_size() -> usize {
        1
    }

    #[inline(always)]
    fn compare_vec_stored(&self, left: u32, right: u32) -> f32 {
        if left == right {
            return 0.0;
        }
        if let Some(left) = self.vectors.get::<[f32; 1536]>(left as usize) {
            let right: &[f32; 1536] = self
                .vectors
                .get(right as usize)
                .expect("You are comparing to an out-of-band id");
            let product = vecmath::dot_product_1536_64(left, right);

            ((product - 1.0) / -2.0).clamp(0.0, 1.0)
        } else {
            f32::MAX
        }
    }

    #[inline(always)]
    fn compare_vec_stored_unstored(&self, stored: u32, unstored: &[u8]) -> f32 {
        debug_assert_eq!(unstored.len(), 6144);
        if let Some(stored) = self.vectors.get::<[f32; 1536]>(stored as usize) {
            let unstored: &[f32; 1536] = &unsafe { *(unstored.as_ptr() as *const [f32; 1536]) };
            let product = vecmath::dot_product_1536_64(stored, unstored);

            ((product - 1.0) / -2.0).clamp(0.0, 1.0)
        } else {
            f32::MAX
        }
    }

    #[inline(always)]
    fn compare_vec_unstored(&self, left: &[u8], right: &[u8]) -> f32 {
        debug_assert_eq!(left.len(), 4096);
        debug_assert_eq!(right.len(), 4096);
        let left: &[f32; 1536] = &unsafe { *(left.as_ptr() as *const [f32; 1536]) };
        let right: &[f32; 1536] = &unsafe { *(right.as_ptr() as *const [f32; 1536]) };
        let product = vecmath::dot_product_1536_64(left, right);

        ((product - 1.0) / -2.0).clamp(0.0, 1.0)
    }
}

pub struct EuclideanDistance8x8<'a> {
    vectors: &'a Vectors,
}

impl<'a> EuclideanDistance8x8<'a> {
    pub fn new(vectors: &'a Vectors) -> Self {
        Self { vectors }
    }
}

impl<'a> VectorComparator for EuclideanDistance8x8<'a> {
    #[inline]
    fn num_vecs(&self) -> usize {
        self.vectors.num_vecs()
    }

    #[inline(always)]
    fn vector_byte_size() -> usize {
        8 * std::mem::size_of::<f32>()
    }

    #[inline]
    #[unroll_for_loops]
    fn compare_vecs_stored(&self, left: &[u32], right: u32, result: &mut [f32]) {
        let mut lefts = [f32::MAX; 64];
        let right: &[f32; 8] = self
            .vectors
            .get(right as usize)
            .expect("You are calling get with an out-of-band id");
        for i in 0..8 {
            let left_id = left[i] as usize;
            if let Some(vec) = self.vectors.get::<[f32; 8]>(left_id) {
                let offset = i * 8;
                lefts[offset..offset + 8].copy_from_slice(vec);
            }
        }

        vecmath::multi_euclidean_8x8(&lefts[..], right, result);
    }

    #[inline]
    #[unroll_for_loops]
    fn compare_vecs_stored_unstored(&self, stored: &[u32], unstored: &[u8], result: &mut [f32]) {
        debug_assert!(unstored.len() == 32);
        let mut storeds = [f32::MAX; 64];
        let unstored: &[f32] =
            unsafe { std::slice::from_raw_parts(unstored.as_ptr() as *const f32, 8) };
        for i in 0..8 {
            let x = stored[i];
            if x != !0 {
                if let Some(vec) = self.vectors.get::<[f32; 8]>(x as usize) {
                    let offset = i * 8;
                    storeds[offset..offset + 8].copy_from_slice(vec);
                }
            }
        }

        vecmath::multi_euclidean_8x8(&storeds[..], unstored, result);
    }

    #[inline]
    #[unroll_for_loops]
    fn compare_vecs_unstored(&self, left: &[u8], right: &[u8], result: &mut [f32]) {
        debug_assert!(left.len() == 32 * 8);
        debug_assert!(right.len() == 32);
        let left: &[f32] = unsafe { std::slice::from_raw_parts(left.as_ptr() as *const f32, 8) };
        let right: &[f32] = unsafe { std::slice::from_raw_parts(right.as_ptr() as *const f32, 8) };
        vecmath::multi_euclidean_8x8(left, right, result);
    }

    fn vec_group_size() -> usize {
        8
    }
}

pub struct DotProductCentroidDistanceCalculator8x8<'a> {
    vectors: &'a Vectors,
}

impl<'a> CentroidDistanceCalculator for DotProductCentroidDistanceCalculator8x8<'a> {
    fn num_centroids(&self) -> usize {
        self.vectors.num_vecs()
    }

    fn calculate_partial_dot_product(&self, c1: u16, c2: u16) -> f16 {
        let left = self.vectors.get::<[f32; 8]>(c1 as usize).unwrap();
        let right = self.vectors.get::<[f32; 8]>(c2 as usize).unwrap();
        vecmath::partial_dot_product(left, right) as f16
    }

    fn calculate_partial_dot_product_norm(&self, c: u16) -> f16 {
        let vec = self.vectors.get::<[f32; 8]>(c as usize).unwrap();
        vecmath::partial_dot_product_norm(vec) as f16
    }
}

pub struct MemoizedComparator128<'a> {
    quantized_vectors: &'a Vectors,
    memoized: &'a MemoizedCentroidDistances,
}

impl<'a> MemoizedComparator128<'a> {
    #[inline(always)]
    #[unroll_for_loops]
    fn compare_raw(&self, left: &[u16; 128], right: &[u16; 128]) -> f32 {
        let mut partial_dot_product_accumulator = f32x8::splat(0.0);
        let mut squared_norm1_accumulator = f32x8::splat(0.0);
        let mut squared_norm2_accumulator = f32x8::splat(0.0);
        for i in 0..16 {
            const SIZE: usize = std::mem::size_of::<u16x8>();
            let offset = i * SIZE;
            let simd_left = u16x8::from_slice(&left[offset..offset + SIZE]);
            let simd_right = u16x8::from_slice(&right[offset..offset + SIZE]);

            partial_dot_product_accumulator += self
                .memoized
                .lookup_centroid_dot_products(simd_left, simd_right);
            squared_norm1_accumulator += self.memoized.lookup_centroid_squared_norms(simd_left);
            squared_norm2_accumulator += self.memoized.lookup_centroid_squared_norms(simd_right);
        }

        let dot_product = partial_dot_product_accumulator.reduce_sum();
        let norm1 = squared_norm1_accumulator.reduce_sum().sqrt();
        let norm2 = squared_norm2_accumulator.reduce_sum().sqrt();

        ((dot_product / (norm1 * norm2) - 1.0) / -2.0).clamp(0.0, 1.0)
    }
}

impl<'a> VectorComparator for MemoizedComparator128<'a> {
    #[inline(always)]
    fn compare_vecs_stored(&self, left: &[u32], right: u32, result: &mut [f32]) {
        let left = left[0];
        result[0] = self.compare_vec_stored(left, right);
    }

    #[inline(always)]
    fn compare_vecs_stored_unstored(&self, stored: &[u32], unstored: &[u8], result: &mut [f32]) {
        let stored = stored[0];
        result[0] = self.compare_vec_stored_unstored(stored, unstored);
    }

    #[inline(always)]
    fn compare_vecs_unstored(&self, left: &[u8], right: &[u8], result: &mut [f32]) {
        result[0] = self.compare_vec_unstored(left, right);
    }

    #[inline(always)]
    fn vec_group_size() -> usize {
        1
    }

    #[inline(always)]
    fn num_vecs(&self) -> usize {
        self.quantized_vectors.num_vecs()
    }

    #[inline(always)]
    fn vector_byte_size() -> usize {
        128 * std::mem::size_of::<u16>()
    }

    #[inline(always)]
    fn compare_vec_stored(&self, left: u32, right: u32) -> f32 {
        if left == right {
            return 0.0;
        }
        if let Some(left) = self.quantized_vectors.get::<[u16; 128]>(left as usize) {
            let right: &[u16; 128] = self
                .quantized_vectors
                .get(right as usize)
                .expect("You are comparing to an out-of-band id");

            self.compare_raw(left, right)
        } else {
            f32::MAX
        }
    }

    #[inline(always)]
    fn compare_vec_stored_unstored(&self, stored: u32, unstored: &[u8]) -> f32 {
        debug_assert_eq!(unstored.len(), 128 * std::mem::size_of::<u16>());
        if let Some(stored) = self.quantized_vectors.get::<[u16; 128]>(stored as usize) {
            let unstored = &unsafe { *(unstored.as_ptr() as *const [u16; 128]) };
            self.compare_raw(stored, unstored)
        } else {
            f32::MAX
        }
    }

    #[inline(always)]
    fn compare_vec_unstored(&self, left: &[u8], right: &[u8]) -> f32 {
        debug_assert_eq!(left.len(), 128 * std::mem::size_of::<u16>());
        debug_assert_eq!(right.len(), 128 * std::mem::size_of::<u16>());
        let left = &unsafe { *(left.as_ptr() as *const [u16; 128]) };
        let right = &unsafe { *(right.as_ptr() as *const [u16; 128]) };
        self.compare_raw(left, right)
    }
}

pub trait VectorComparatorConstructor<'a>: VectorComparator {
    fn new_from_vecs(vecs: &'a Vectors) -> Self;
}

pub struct EuclideanDistance8x8Constructor;

impl<'a> VectorComparatorConstructor<'a> for EuclideanDistance8x8<'a> {
    fn new_from_vecs(vectors: &'a Vectors) -> Self {
        EuclideanDistance8x8 { vectors }
    }
}
