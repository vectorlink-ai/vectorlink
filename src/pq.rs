use std::ops::Range;

use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    comparator::VectorComparatorConstructor,
    hnsw::{BuildParams, Hnsw},
    layer::{SearchParams, VectorComparator},
    memoize::{CentroidDistanceCalculator, MemoizedCentroidDistances},
    vectors::{Vector, Vectors},
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

pub struct Quantizer {
    hnsw: Hnsw,
    sp: SearchParams,
}

impl Quantizer {
    pub fn quantize<C: VectorComparator>(&self, unquantized: &[u8], comparator: &C) -> Vec<u8> {
        debug_assert_eq!(0, unquantized.len() % C::vector_byte_size());
        let mut quantized: Vec<u16> = unquantized
            .par_chunks(C::vector_byte_size())
            .map(|chunk| {
                self.hnsw
                    .search_from_initial(Vector::Slice(chunk), &self.sp, comparator)
                    .first()
                    .0 as u16
            })
            .collect();

        let cast = unsafe {
            Vec::from_raw_parts(
                quantized.as_mut_ptr() as *mut u8,
                quantized.len() * 2,
                quantized.capacity() / 2,
            )
        };
        std::mem::forget(quantized);

        cast
    }

    pub fn reconstruct<C: VectorComparator>(&self, quantized: &[u8], vectors: &Vectors) -> Vec<u8> {
        let cast = unsafe {
            std::slice::from_raw_parts(quantized.as_ptr() as *const u16, quantized.len() * 2)
        };
        let reconstructed_byte_size = quantized.len() * vectors.vector_byte_size();
        let mut result: Vec<u8> = Vec::with_capacity(reconstructed_byte_size);
        for &c in cast {
            result.extend_from_slice(&vectors[c as usize]);
        }

        result
    }

    pub fn quantize_all<V: Iterator<Item = Vec<u8>>, C: VectorComparator>(
        &self,
        num_vecs: usize,
        vecs: V,
        comparator: &C,
    ) -> Vectors {
        let total_byte_size = num_vecs * C::vector_byte_size();
        let mut data = Vec::with_capacity(total_byte_size);
        for v in vecs {
            let quantized = self.quantize(&v, comparator);
            data.extend(quantized);
        }
        Vectors::new(data, C::vector_byte_size())
    }

    fn new(hnsw: Hnsw, sp: SearchParams) -> Self {
        Self { hnsw, sp }
    }
}

pub fn create_pq<
    'a,
    VRI: VectorRangeIndexable,
    V: Iterator<Item = Vec<u8>>,
    CentroidComparatorConstructor: for<'b> VectorComparatorConstructor<'b>,
    QuantizedComparatorConstructor: for<'b> VectorComparatorConstructor<'b>,
    CDC: CentroidDistanceCalculator,
>(
    vectors: &'a VRI,
    vector_stream: V,
    centroid_count: usize,
    centroid_byte_size: usize,
    centroid_build_params: &BuildParams,
    quantized_build_params: &BuildParams,
    quantizer_search_params: SearchParams,
    cdc: CDC,
    seed: u64,
) -> Pq {
    let centroids = centroid_finder(vectors, centroid_count, centroid_byte_size, seed);
    let centroid_comparator = CentroidComparatorConstructor::new_from_vecs(&centroids);

    let centroid_hnsw = Hnsw::generate(centroid_build_params, &centroid_comparator);
    let quantizer = Quantizer::new(centroid_hnsw, quantizer_search_params);
    let quantized_vectors =
        quantizer.quantize_all(vectors.num_vecs(), vector_stream, &centroid_comparator);
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

    use crate::test_util::random_vectors;

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
