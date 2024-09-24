use unroll::unroll_for_loops;

use crate::{layer::VectorComparator, vecmath, vectors::Vectors};

pub struct CosineDistance1024<'a> {
    vectors: &'a Vectors,
}

impl<'a> CosineDistance1024<'a> {
    pub fn new(vectors: &'a Vectors) -> Self {
        Self { vectors }
    }
}

impl<'a> VectorComparator for CosineDistance1024<'a> {
    #[inline]
    fn num_vecs(&self) -> usize {
        self.vectors.len()
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
        self.vectors.len()
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
