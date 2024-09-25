use std::ops::Range;

use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    comparator::{CosineDistance1024, EuclideanDistance8x8, VectorComparatorConstructor},
    hnsw::{BuildParams, Hnsw},
    layer::VectorComparator,
    memoize::{CentroidDistanceCalculator, MemoizedCentroidDistances},
    vectors::Vectors,
};

pub struct Pq {
    memoized_distances: MemoizedCentroidDistances,
    quantized_hnsw: Hnsw,
    quantizer: Quantizer,
}

pub trait VectorRangeIndexable {
    fn get_ranges(&self, byte_ranges: &[Range<usize>]) -> Vectors;
    fn vector_byte_size(&self) -> usize;
    fn num_vecs(&self) -> usize;
}

pub struct VectorRangeIndexableForVectors<'a>(&'a Vectors);

impl<'a> VectorRangeIndexable for VectorRangeIndexableForVectors<'a> {
    fn get_ranges(&self, byte_ranges: &[Range<usize>]) -> Vectors {
        assert!(!byte_ranges.is_empty());
        let vector_count = byte_ranges.len();
        let vector_byte_size = byte_ranges[0].len();
        let mut data: Vec<u8> = vec![0; vector_count * vector_byte_size];
        byte_ranges
            .into_par_iter()
            .zip(data.par_chunks_mut(vector_byte_size))
            .for_each(|(range, chunk): (_, &mut [u8])| {
                let new = &self.0.data()[range.clone()];
                chunk.copy_from_slice(new)
            });

        Vectors::new(data, vector_byte_size)
    }

    fn vector_byte_size(&self) -> usize {
        self.0.vector_byte_size()
    }

    fn num_vecs(&self) -> usize {
        self.0.num_vecs()
    }
}

pub fn centroid_finder<V: VectorRangeIndexable>(
    vecs: &V,
    centroid_count: usize,
    centroid_byte_size: usize,
    seed: u64,
) -> Vectors {
    assert_eq!(vecs.vector_byte_size() % centroid_byte_size, 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let range_count = vecs.num_vecs() * (vecs.vector_byte_size() / centroid_byte_size);
    let ranges: Vec<Range<usize>> = (0..range_count)
        .choose_multiple(&mut rng, centroid_count)
        .into_iter()
        .map(|i| {
            let centroid_start = i * centroid_byte_size;
            let centroid_end = centroid_start + centroid_byte_size;
            centroid_start..centroid_end
        })
        .collect();

    vecs.get_ranges(&ranges)
}

pub struct CentroidConstructor1024x8;

pub trait VectorStreamable {}

pub struct Quantizer {
    hnsw: Hnsw,
}

impl Quantizer {
    fn quantize<C: VectorComparator>(&self, unquantized: &[u8], comparator: &C) -> Vec<u16> {
        todo!()
    }

    fn reconstruct<C: VectorComparator>(&self, quantized: &[u16], comparator: &C) -> Vec<u8> {
        todo!()
    }

    fn quantize_all<C: VectorComparator, V: VectorStreamable>(
        &self,
        vecs: &V,
        comparator: &C,
    ) -> Vectors {
        todo!();
    }

    fn new(hnsw: Hnsw) -> Self {
        Self { hnsw }
    }
}

pub fn create_pq<
    'a,
    VRI: VectorRangeIndexable,
    V: VectorStreamable,
    CentroidComparatorConstructor: VectorComparatorConstructor,
    QuantizedComparatorConstructor: VectorComparatorConstructor,
    CDC: CentroidDistanceCalculator,
>(
    vectors: &'a VRI,
    vector_stream: &'a V,
    centroid_count: usize,
    centroid_byte_size: usize,
    centroid_build_params: &BuildParams,
    quantized_build_params: &BuildParams,
    cdc: CDC,
    seed: u64,
) -> Pq {
    let centroids = centroid_finder(vectors, centroid_count, centroid_byte_size, seed);
    let centroid_comparator = CentroidComparatorConstructor::new_from_vecs(&centroids);

    let centroid_hnsw = Hnsw::generate(centroid_build_params, &centroid_comparator);
    let quantizer = Quantizer::new(centroid_hnsw);
    let quantized_vectors = quantizer.quantize_all(vector_stream, &centroid_comparator);
    let quantized_comparator = QuantizedComparatorConstructor::new_from_vecs(&quantized_vectors);
    let quantized_hnsw = Hnsw::generate(quantized_build_params, &quantized_comparator);
    let memoized_distances = MemoizedCentroidDistances::new(&cdc);
    Pq {
        memoized_distances,
        quantized_hnsw,
        quantizer,
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        comparator::{CosineDistance1024, EuclideanDistance8x8},
        hnsw::Hnsw,
        test_util::{random_vectors, random_vectors_normalized},
    };

    use super::*;

    #[test]
    fn construct_pq_hnsw() {
        let number_of_vecs = 100_000;
        let vecs = random_vectors(number_of_vecs, 1024, 0x533D);
        let vector_indexable = VectorRangeIndexableForVectors(&vecs);
        //let vector_stream = VectorStreamForVectors(&vecs);
        //let pq = create_pq(vectors,
        todo!();
    }
}
