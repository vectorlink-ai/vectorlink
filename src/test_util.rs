use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::vectors::Vectors;

pub fn random_vectors(num_vecs: usize, dimension: usize, seed: u64) -> Vectors {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<f32> = (0..num_vecs * dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let data_cast = unsafe {
        Vec::from_raw_parts(
            data.as_mut_ptr() as *mut u8,
            data.len() * std::mem::size_of::<f32>(),
            data.capacity(),
        )
    };
    std::mem::forget(data);

    Vectors::new(data_cast, dimension * std::mem::size_of::<f32>())
}

pub fn random_vectors_normalized<const DIMENSION: usize>(num_vecs: usize, seed: u64) -> Vectors {
    let mut vectors = random_vectors(num_vecs, DIMENSION, seed);

    // TODO VECTORIZE
    for i in 0..vectors.num_vecs() {
        let v: &mut [f32; DIMENSION] = vectors.get_mut(i);
        let size: f32 = v.iter().map(|e| e * e).sum::<f32>().sqrt();
        v.iter_mut().for_each(|e| *e /= size);
    }

    vectors
}
