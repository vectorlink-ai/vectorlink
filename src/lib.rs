#![feature(f16)]
pub mod bitmap;
pub mod memoize;
pub mod ring_queue;
pub mod vectors;

/*
pub struct Hnsw {
    vectors: Vectors,
    layers: Vec<Layer>,
}

pub struct Layer {
    neighborhoods: Vec<u32>,
    single_neighborhood_size: usize,
}

struct VectorId(u32);

pub trait Comparator: Sync {
    fn compare_unstored(vector_id: VectorId, vector: &[f32]);
    fn compare_stored(vector_id_1: VectorId, vector_id_2: VectorId);
}

impl Layer {}
*/

#[cfg(test)]
mod tests {}
