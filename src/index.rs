use crate::{
    comparator::{
        CosineDistance1024, EuclideanDistance8x8, NewMemoizedComparator128,
        QuantizedVectorComparatorConstructor,
    },
    hnsw::Hnsw,
    params::SearchParams,
    pq::Pq,
    ring_queue::OrderedRingQueue,
    vectors::{Vector, Vectors},
};

pub trait Searcher {
    fn search(&self, query_vec: Vector, sp: &SearchParams) -> OrderedRingQueue;
}

pub enum Index {
    Pq1024x8 { pq: Pq, vectors: Vectors },
    Hnsw1024 { hnsw: Hnsw, vectors: Vectors },
}

impl Searcher for Index {
    fn search(&self, query_vec: Vector, sp: &SearchParams) -> OrderedRingQueue {
        match self {
            Index::Pq1024x8 { pq, vectors: _ } => {
                let mut quantized = vec![0_u8; 256];
                let query_vec: Vector = match query_vec {
                    Vector::Slice(slice) => {
                        let centroid_comparator = EuclideanDistance8x8::new(pq.centroids());
                        pq.quantizer()
                            .quantize(slice, &centroid_comparator, &mut quantized);

                        Vector::Slice(&quantized)
                    }
                    Vector::Id(id) => Vector::Id(id),
                };
                let quantized_comparator =
                    NewMemoizedComparator128::new(pq.centroids(), pq.memoized_distances());
                pq.search_from_initial_quantized(query_vec, sp, &quantized_comparator)
            }
            Index::Hnsw1024 { hnsw, vectors } => {
                let comparator = CosineDistance1024::new(vectors);
                hnsw.search_from_initial(query_vec, sp, &comparator)
            }
        }
    }
}
