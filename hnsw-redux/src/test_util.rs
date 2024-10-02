use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{util::SimdAlignedAllocation, vectors::Vectors};

pub fn random_vectors(num_vecs: usize, dimension: usize, seed: u64) -> Vectors {
    let mut rng = StdRng::seed_from_u64(seed);
    let float_len = num_vecs * dimension;
    let byte_len = float_len * std::mem::size_of::<f32>();
    let mut data = unsafe { SimdAlignedAllocation::alloc(byte_len) };
    let slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, float_len) };
    for elt in slice.iter_mut() {
        *elt = rng.gen_range(-1.0..1.0);
    }

    Vectors::new(data, dimension * std::mem::size_of::<f32>())
}

pub fn random_vectors_normalized(num_vecs: usize, dimension: usize, seed: u64) -> Vectors {
    let mut vectors = random_vectors(num_vecs, dimension, seed);

    // TODO VECTORIZE
    for i in 0..vectors.num_vecs() {
        let v: &mut [f32] = vectors.get_mut_f32_slice(i);
        let size: f32 = v.iter().map(|e| e * e).sum::<f32>().sqrt();
        v.iter_mut().for_each(|e| *e /= size);
    }

    vectors
}
