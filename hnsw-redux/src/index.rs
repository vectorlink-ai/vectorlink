use enum_dispatch::enum_dispatch;

use crate::{
    comparator::{
        CosineDistance1024, EuclideanDistance8x8, NewMemoizedComparator128,
        QuantizedVectorComparatorConstructor,
    },
    hnsw::Hnsw,
    params::{BuildParams, SearchParams},
    pq::Pq,
    ring_queue::OrderedRingQueue,
    vectors::{Vector, Vectors},
};

pub enum DispatchError {
    FeatureDoesNotExist,
}

#[enum_dispatch]
pub trait Index {
    fn search(&self, query_vec: Vector, sp: &SearchParams) -> OrderedRingQueue;
    fn test_recall(&self, proportion: f32, sp: &SearchParams, seed: u64) -> f32;
    fn optimize(&mut self, sp: &SearchParams, seed: u64) -> f32;
    fn reconstruction_statistics(&self) -> Result<(f32, f32), DispatchError> {
        Err(DispatchError::FeatureDoesNotExist)
    }
}

pub trait NewIndex: Index {
    fn generate(vectors: Vectors, bp: &BuildParams) -> Self;
}

pub struct Pq1024x8 {
    pq: Pq,
    vectors: Vectors,
    name: String,
}
impl Pq1024x8 {
    pub fn new(name: String, pq: Pq, vectors: Vectors) -> Self {
        Self { name, pq, vectors }
    }
}
pub struct Hnsw1024 {
    hnsw: Hnsw,
    vectors: Vectors,
    name: String,
}

#[enum_dispatch(Index)]
pub enum IndexConfiguration {
    Hnsw1024(Hnsw1024),
    Pq1024x8(Pq1024x8),
}

impl Index for Pq1024x8 {
    fn search(&self, query_vec: Vector, sp: &SearchParams) -> OrderedRingQueue {
        let Pq1024x8 { pq, .. } = self;
        let quantized_comparator =
            NewMemoizedComparator128::new(pq.quantized_vectors(), pq.memoized_distances());
        match query_vec {
            Vector::Slice(slice) => {
                let mut quantized = vec![0_u8; 256];
                let centroid_comparator = EuclideanDistance8x8::new(pq.centroids());
                pq.quantizer()
                    .quantize(slice, &centroid_comparator, &mut quantized);

                pq.search_from_initial_quantized(
                    Vector::Slice(&quantized),
                    sp,
                    &quantized_comparator,
                )
            }
            Vector::Id(id) => {
                pq.search_from_initial_quantized(Vector::Id(id), sp, &quantized_comparator)
            }
        }
    }

    fn test_recall(&self, proportion: f32, sp: &SearchParams, seed: u64) -> f32 {
        let Pq1024x8 { pq, .. } = self;
        let quantized_comparator =
            NewMemoizedComparator128::new(pq.quantized_vectors(), pq.memoized_distances());
        self.pq
            .test_recall(proportion, sp, &quantized_comparator, seed)
    }

    fn optimize(&mut self, sp: &SearchParams, seed: u64) -> f32 {
        let quantized_comparator =
            NewMemoizedComparator128::new(&self.pq.quantized_vectors, &self.pq.memoized_distances);
        let quantized_hnsw = &mut self.pq.quantized_hnsw;
        quantized_hnsw.optimize(sp, &quantized_comparator, seed)
    }
}

impl Hnsw1024 {
    pub fn new(name: String, hnsw: Hnsw, vectors: Vectors) -> Self {
        Self {
            name,
            hnsw,
            vectors,
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn hnsw(&self) -> &Hnsw {
        &self.hnsw
    }

    pub fn generate(name: String, vectors: Vectors, bp: &BuildParams) -> Self {
        let comparator = CosineDistance1024::new(&vectors);
        let hnsw = Hnsw::generate(bp, &comparator);

        Hnsw1024 {
            name,
            hnsw,
            vectors,
        }
    }
}

impl Index for Hnsw1024 {
    fn search(&self, query_vec: Vector, sp: &SearchParams) -> OrderedRingQueue {
        let Hnsw1024 { hnsw, vectors, .. } = self;
        let comparator = CosineDistance1024::new(vectors);
        hnsw.search_from_initial(query_vec, sp, &comparator)
    }

    fn test_recall(&self, proportion: f32, sp: &SearchParams, seed: u64) -> f32 {
        let Hnsw1024 { hnsw, vectors, .. } = self;
        let comparator = CosineDistance1024::new(vectors);
        hnsw.test_recall(proportion, sp, &comparator, seed)
    }

    fn optimize(&mut self, sp: &SearchParams, seed: u64) -> f32 {
        let Hnsw1024 { hnsw, vectors, .. } = self;
        let comparator = CosineDistance1024::new(vectors);
        hnsw.optimize(sp, &comparator, seed)
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        comparator::{
            NewDotProductCentroidDistanceCalculator8, NewEuclideanDistance8x8,
            NewMemoizedComparator128,
        },
        params::BuildParams,
        pq::{create_pq, VectorRangeIndexableForVectors},
        test_util::random_vectors,
    };

    use super::*;

    #[test]
    #[ignore]
    fn search_pq_index() {
        let number_of_vecs = 100_000;
        let vecs = random_vectors(number_of_vecs, 1024, 0x533D);
        let vector_indexable = VectorRangeIndexableForVectors(&vecs);
        let vector_stream = vecs.iter();
        let centroid_count = u16::MAX as usize;
        let centroid_byte_size = 8 * std::mem::size_of::<f32>();
        let mut centroid_build_params = BuildParams::default();
        centroid_build_params.optimize_sp.parallel_visit_count = 12;
        centroid_build_params.optimize_sp.circulant_parameter_count = 8;
        let mut quantized_build_params = BuildParams::default();
        quantized_build_params.optimize_sp.parallel_visit_count = 12;
        quantized_build_params.optimize_sp.circulant_parameter_count = 8;
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

        let index = Pq1024x8 {
            pq,
            vectors: vecs,
            name: "dummy".to_string(),
        };
        let mut sp = SearchParams::default();
        sp.parallel_visit_count = 12;
        sp.circulant_parameter_count = 8;

        let recall = index.test_recall(0.10, &sp, 0x533D);
        eprintln!("recall: {recall}");
        assert!(recall > 0.95);
    }
}
