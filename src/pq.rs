use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    comparator::{CosineDistance1024, EuclideanDistance8x8},
    hnsw::Hnsw,
    layer::VectorComparator,
    vectors::Vectors,
};

pub enum Index {
    Pq1024x8 {
        pq: Pq,
        centroids: Vectors,
        vectors: Vectors,
    },
}

pub struct Pq {
    centroid_hnsw: Hnsw,
    quantized_hnsw: Hnsw,
}

pub fn centroid_finder(
    vecs: &Vectors,
    centroid_count: usize,
    centroid_byte_size: usize,
    seed: u64,
) -> Vectors {
    assert_eq!(vecs.vector_byte_size() % centroid_byte_size, 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<u8> = vec![0; centroid_count * centroid_byte_size];

    (0..vecs.num_vecs())
        .choose_multiple(&mut rng, centroid_count)
        .into_iter()
        .map(|i| {
            (
                i,
                rng.gen_range(0..vecs.vector_byte_size() / centroid_byte_size),
            )
        })
        .zip(data.chunks_mut(centroid_byte_size))
        .par_bridge()
        .for_each(|((i, j), chunk): (_, &mut [u8])| {
            let old = &vecs[i];
            let new = &old[centroid_byte_size * j..centroid_byte_size * (j + 1)];
            chunk.copy_from_slice(new)
        });

    Vectors::new(data, centroid_byte_size)
}

pub trait CentroidConstructor<C: VectorComparator> {
    fn find_centroids(vecs: &Vectors) -> Vectors;
    fn index_centroids(vecs: &Vectors, c: C) -> Hnsw;
}

pub struct CentroidConstructor1024x8;

impl<C: VectorComparator> CentroidConstructor<C> for CentroidConstructor1024x8 {
    fn find_centroids(vecs: &Vectors) -> Vectors {
        let centroid_byte_size = std::mem::size_of::<u16>();
        centroid_finder(vecs, u16::MAX as usize, centroid_byte_size, 0x533D)
    }
    fn index_centroids(vecs: &Vectors) -> Hnsw {
        todo!();
    }
}

pub trait Quantizer<C: VectorComparator> {
    fn quantize(&self, vec: &[u8]) -> Vec<u16>;
    fn unquantize(&self, quantized: &[u16]) -> Vec<u8>;
    fn quantize_all(&self, vecs: &Vectors) -> Vectors;
    fn new(hnsw: Hnsw, comparator: &C) -> Self;
}

pub struct Quantizer1024x8<'a, C> {
    hnsw: &'a Hnsw,
    comparator: &'a C,
}

impl<'a, C: VectorComparator> Quantizer<C> for Quantizer1024x8<'a, C> {
    fn quantize(&self, vec: &[u8]) -> Vec<u16> {
        todo!()
    }

    fn unquantize(&self, quantized: &[u16]) -> Vec<u8> {
        todo!()
    }

    fn quantize_all(&self, vecs: &Vectors) -> Vectors {
        todo!();
    }

    fn new(hnsw: Hnsw, comparator: &C) -> Self {
        todo!();
    }
}

impl Pq {
    pub fn new(centroid_hnsw: Hnsw, quantized_hnsw: Hnsw) -> Self {
        Self {
            centroid_hnsw,
            quantized_hnsw,
        }
    }
}
