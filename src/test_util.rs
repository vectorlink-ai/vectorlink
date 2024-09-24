use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::vectors::Vectors;

pub fn random_8_vectors(num_vecs: usize, seed: u64) -> Vectors {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<f32> = (0..num_vecs * 8)
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

    Vectors::new(data_cast, 32)
}
