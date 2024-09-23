use unroll::unroll_for_loops;

use crate::{layer::VectorComparator, vecmath, vectors::Vectors};

pub struct CosineDistance1024<'a> {
    vectors: &'a Vectors,
}

impl<'a> VectorComparator for CosineDistance1024<'a> {
    #[inline(always)]
    fn compare_vecs_stored(&self, left: &[u32], right: u32, result: &mut [f32]) {
        let left = left[0];
        result[0] = self.compare_vec_stored(left, right);
    }

    #[inline(always)]
    fn compare_vecs_unstored(&self, stored: &[u32], unstored: &[u8], result: &mut [f32]) {
        let stored = stored[0];
        result[0] = self.compare_vec_unstored(stored, unstored);
    }

    #[inline(always)]
    fn vec_group_size() -> usize {
        1
    }

    #[inline(always)]
    fn compare_vec_stored(&self, left: u32, right: u32) -> f32 {
        let left: &[f32; 1024] = self.vectors.get(left as usize);
        let right: &[f32; 1024] = self.vectors.get(right as usize);
        let product = vecmath::dot_product_1024_64(left, right);

        ((product - 1.0) / -2.0).clamp(0.0, 1.0)
    }

    #[inline(always)]
    fn compare_vec_unstored(&self, stored: u32, unstored: &[u8]) -> f32 {
        debug_assert_eq!(unstored.len(), 4096);
        let stored: &[f32; 1024] = self.vectors.get(stored as usize);
        let unstored: &[f32; 1024] = &unsafe { *(unstored.as_ptr() as *const [f32; 1024]) };
        let product = vecmath::dot_product_1024_64(stored, unstored);

        ((product - 1.0) / -2.0).clamp(0.0, 1.0)
    }
}

pub struct EuclideanDistance8x8<'a> {
    vectors: &'a Vectors,
}

impl<'a> VectorComparator for EuclideanDistance8x8<'a> {
    #[inline]
    #[unroll_for_loops]
    fn compare_vecs_stored(&self, left: &[u32], right: u32, result: &mut [f32]) {
        let mut lefts = [0.0_f32; 64];
        let right: &[f32; 8] = self.vectors.get(right as usize);
        for i in 0..8 {
            let vec = self.vectors.get::<[f32; 8]>(left[i] as usize);
            let offset = i * 8;
            lefts[offset..offset + 8].copy_from_slice(vec);
        }

        vecmath::multi_euclidean_8x8(&lefts[..], right, result);
    }

    #[inline]
    #[unroll_for_loops]
    fn compare_vecs_unstored(&self, stored: &[u32], unstored: &[u8], result: &mut [f32]) {
        let mut storeds = [0.0_f32; 64];
        let unstored: &[f32] =
            unsafe { std::slice::from_raw_parts(unstored.as_ptr() as *const f32, 8) };
        for i in 0..8 {
            let vec = self.vectors.get::<[f32; 8]>(stored[i] as usize);
            let offset = i * 8;
            storeds[offset..offset + 8].copy_from_slice(vec);
        }

        vecmath::multi_euclidean_8x8(&storeds[..], unstored, result);
        todo!()
    }

    fn vec_group_size() -> usize {
        8
    }
}
