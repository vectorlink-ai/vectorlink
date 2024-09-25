use std::ops::Range;

use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    comparator::{QuantizedVectorComparatorConstructor, VectorComparatorConstructor},
    hnsw::{BuildParams, Hnsw},
    layer::{SearchParams, VectorComparator},
    memoize::{
        CentroidDistanceCalculator, CentroidDistanceCalculatorConstructor,
        MemoizedCentroidDistances,
    },
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

pub struct VectorRangeIndexableForVectors<'a>(pub &'a Vectors);

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
    pub fn quantize<C: VectorComparator>(
        &self,
        unquantized: &[u8],
        comparator: &C,
        out: &mut [u8],
    ) {
        debug_assert_eq!(0, unquantized.len() % C::vector_byte_size());
        for (chunk, out) in unquantized
            .chunks(C::vector_byte_size())
            .zip(out.chunks_mut(std::mem::size_of::<u16>()))
        {
            let out_cast: &mut u16 = unsafe { &mut *(out.as_mut_ptr() as *mut u16) };
            *out_cast = self
                .hnsw
                .search_from_initial(Vector::Slice(chunk), &self.sp, comparator)
                .first()
                .0 as u16;
        }
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

    pub fn quantize_all<'a, V: Iterator<Item = &'a [u8]> + Send, C: VectorComparator>(
        &self,
        num_vecs: usize,
        vecs: V,
        comparator: &C,
    ) -> Vectors {
        let total_byte_size = num_vecs * C::vector_byte_size();
        let mut data = Vec::with_capacity(total_byte_size);
        vecs.zip(data.spare_capacity_mut().chunks_mut(C::vector_byte_size()))
            .enumerate()
            .par_bridge()
            .for_each(|(ix, (v, out))| {
                if ix % 10000 == 0 {
                    eprintln!("quantizing {ix}");
                }
                let out_cast: &mut [u8] = unsafe { std::mem::transmute(out) };
                self.quantize(&v, comparator, out_cast);
            });

        unsafe {
            data.set_len(total_byte_size);
        }
        Vectors::new(data, C::vector_byte_size())
    }

    pub fn new(hnsw: Hnsw, sp: SearchParams) -> Self {
        Self { hnsw, sp }
    }
}

pub fn create_pq<
    'a,
    CentroidComparatorConstructor: VectorComparatorConstructor,
    QuantizedComparatorConstructor: QuantizedVectorComparatorConstructor,
    CDC: CentroidDistanceCalculatorConstructor,
    VRI: VectorRangeIndexable,
    V: Iterator<Item = &'a [u8]> + Send,
>(
    vectors: &'a VRI,
    vector_stream: V,
    centroid_count: usize,
    centroid_byte_size: usize,
    centroid_build_params: &BuildParams,
    quantized_build_params: &BuildParams,
    quantizer_search_params: SearchParams,
    seed: u64,
) -> Pq {
    let centroids = centroid_finder(vectors, centroid_count, centroid_byte_size, seed);
    eprintln!("found centroids");
    let centroid_comparator = CentroidComparatorConstructor::new_from_vecs(&centroids);
    let centroid_distance_calculator = CDC::new(&centroids);

    let centroid_hnsw = Hnsw::generate(centroid_build_params, &centroid_comparator);
    eprintln!("generated centroid hnsw");
    let quantizer = Quantizer::new(centroid_hnsw, quantizer_search_params);
    let quantized_vectors =
        quantizer.quantize_all(vectors.num_vecs(), vector_stream, &centroid_comparator);
    eprintln!("quantized");

    let memoized_distances = MemoizedCentroidDistances::new(&centroid_distance_calculator);
    let quantized_comparator =
        QuantizedComparatorConstructor::new(&quantized_vectors, &memoized_distances);
    let quantized_hnsw = Hnsw::generate(quantized_build_params, &quantized_comparator);
    eprintln!("generated quantized hnsw");

    std::mem::drop(quantized_comparator);

    Pq {
        memoized_distances,
        quantized_hnsw,
        quantizer,
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        comparator::{
            DotProductCentroidDistanceCalculator8, EuclideanDistance8x8, MemoizedComparator128,
            NewDotProductCentroidDistanceCalculator8, NewEuclideanDistance8x8,
            NewMemoizedComparator128,
        },
        test_util::random_vectors,
    };

    use super::*;

    #[test]
    fn construct_pq_hnsw() {
        let number_of_vecs = 100_000;
        let vecs = random_vectors(number_of_vecs, 1024, 0x533D);
        let vector_indexable = VectorRangeIndexableForVectors(&vecs);
        let vector_stream = vecs.iter();
        let centroid_count = u16::MAX as usize;
        let centroid_byte_size = 8 * std::mem::size_of::<f32>();
        let centroid_build_params = BuildParams::default();
        let quantized_build_params = BuildParams::default();
        let quantizer_search_params = SearchParams::default();

        let pq = create_pq::<
            NewEuclideanDistance8x8,
            NewMemoizedComparator128,
            NewDotProductCentroidDistanceCalculator8,
            _,
            _,
        >(
            &vector_indexable,
            vector_stream,
            centroid_count,
            centroid_byte_size,
            &centroid_build_params,
            &quantized_build_params,
            quantizer_search_params,
            0x533D,
        );
    }
}
