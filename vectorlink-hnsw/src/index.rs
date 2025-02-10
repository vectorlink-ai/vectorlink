use enum_dispatch::enum_dispatch;
use find_peaks::Peak;
use itertools::Either;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io;
use std::os::unix::prelude::FileExt;
use std::{fs::File, path::Path};

use crate::comparator::CosineDistance1536;
use crate::params::OptimizationParams;
use crate::{
    comparator::{
        CosineDistance1024, EuclideanDistance8x8, NewMemoizedComparator128,
        QuantizedVectorComparatorConstructor,
    },
    hnsw::Hnsw,
    params::{BuildParams, FindPeaksParams, SearchParams},
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
    fn num_vectors(&self) -> usize;
    fn test_recall_with_proportion(
        &self,
        proportion: f32,
        sp: &SearchParams,
        seed: u64,
    ) -> f32;
    fn optimize_and_save<P: AsRef<Path>>(
        &mut self,
        root: P,
        op: &OptimizationParams,
    ) -> Result<f32, io::Error>;
    fn optimize(&mut self, op: &OptimizationParams) -> f32;
    fn knn_into_file<P: AsRef<Path>>(
        &self,
        k: usize,
        sp: SearchParams,
        path: P,
    );
    fn reconstruction_statistics(&self) -> Result<(f32, f32), DispatchError> {
        Err(DispatchError::FeatureDoesNotExist)
    }

    fn test_recall(&self, sp: &SearchParams, seed: u64) -> f32 {
        let proportion = 1.0 / (self.num_vectors() as f32).sqrt();
        self.test_recall_with_proportion(proportion, sp, seed)
    }

    fn find_distance_transitions(
        &self,
        fpp: FindPeaksParams,
    ) -> Vec<(f32, Peak<f32>)>;
}

pub trait NewIndex: Index {
    fn generate(vectors: Vectors, bp: &BuildParams) -> Self;
}

pub struct Pq1024x8 {
    pq: Pq,
    vectors: Vectors,
    #[allow(unused)]
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

pub struct Hnsw1536 {
    hnsw: Hnsw,
    vectors: Vectors,
    name: String,
}

#[enum_dispatch(Index)]
pub enum IndexConfiguration {
    Hnsw1024(Hnsw1024),
    Hnsw1536(Hnsw1536),
    Pq1024x8(Pq1024x8),
}

impl Index for Pq1024x8 {
    fn num_vectors(&self) -> usize {
        self.vectors.num_vecs()
    }
    fn search(&self, query_vec: Vector, sp: &SearchParams) -> OrderedRingQueue {
        let Pq1024x8 { pq, .. } = self;
        let quantized_comparator = NewMemoizedComparator128::new(
            pq.quantized_vectors(),
            pq.memoized_distances(),
        );
        match query_vec {
            Vector::Slice(slice) => {
                let mut quantized = vec![0_u8; 256];
                let centroid_comparator =
                    EuclideanDistance8x8::new(pq.centroids());
                pq.quantizer().quantize(
                    slice,
                    &centroid_comparator,
                    &mut quantized,
                );

                pq.search_from_initial_quantized(
                    Vector::Slice(&quantized),
                    sp,
                    &quantized_comparator,
                )
            }
            Vector::Id(id) => pq.search_from_initial_quantized(
                Vector::Id(id),
                sp,
                &quantized_comparator,
            ),
        }
    }

    fn test_recall_with_proportion(
        &self,
        proportion: f32,
        sp: &SearchParams,
        seed: u64,
    ) -> f32 {
        let Pq1024x8 { pq, .. } = self;
        let quantized_comparator = NewMemoizedComparator128::new(
            pq.quantized_vectors(),
            pq.memoized_distances(),
        );
        self.pq
            .test_recall(proportion, sp, &quantized_comparator, seed)
    }

    fn optimize(&mut self, op: &OptimizationParams) -> f32 {
        let quantized_comparator = NewMemoizedComparator128::new(
            &self.pq.quantized_vectors,
            &self.pq.memoized_distances,
        );
        let mut recall = 0.0;
        for i in 0..self.pq.quantized_hnsw.layers().len() {
            let layer_recall = self.pq.quantized_hnsw.optimize_layer(
                i,
                op,
                &quantized_comparator,
            );
            recall = layer_recall;
            eprintln!("saving layer [{i}] with recall {recall}");
            //self.store_hnsw(&root)?;
            todo!("No serialization possible yet");
        }
        recall
    }

    fn optimize_and_save<P: AsRef<Path>>(
        &mut self,
        _root: P,
        op: &OptimizationParams,
    ) -> Result<f32, io::Error> {
        let quantized_comparator = NewMemoizedComparator128::new(
            &self.pq.quantized_vectors,
            &self.pq.memoized_distances,
        );
        let mut recall = 0.0;
        for i in 0..self.pq.quantized_hnsw.layers().len() {
            let layer_recall = self.pq.quantized_hnsw.optimize_layer(
                i,
                op,
                &quantized_comparator,
            );
            recall = layer_recall;
            eprintln!("saving layer [{i}] with recall {recall}");
            //self.store_hnsw(&root)?;
            todo!("No serialization possible yet");
        }
        Ok(recall)
    }

    fn knn_into_file<P: AsRef<Path>>(
        &self,
        k: usize,
        sp: SearchParams,
        path: P,
    ) {
        let quantized_comparator = NewMemoizedComparator128::new(
            &self.pq.quantized_vectors,
            &self.pq.memoized_distances,
        );
        let file = File::create(path).unwrap();
        let record_size = k * (size_of::<(u32, f32)>());
        self.pq
            .quantized_hnsw
            .knn(k, sp, quantized_comparator)
            .for_each(|(i, pairs)| {
                let raw_data = unsafe {
                    std::slice::from_raw_parts(
                        pairs.as_ptr() as *const u8,
                        record_size,
                    )
                };
                file.write_all_at(raw_data, (i as usize * record_size) as u64)
                    .unwrap()
            });
    }

    fn find_distance_transitions(
        &self,
        fpp: FindPeaksParams,
    ) -> Vec<(f32, Peak<f32>)> {
        let quantized_comparator = NewMemoizedComparator128::new(
            &self.pq.quantized_vectors,
            &self.pq.memoized_distances,
        );
        self.pq
            .quantized_hnsw
            .find_distance_transitions(fpp, quantized_comparator)
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
    fn num_vectors(&self) -> usize {
        self.vectors.num_vecs()
    }

    fn search(&self, query_vec: Vector, sp: &SearchParams) -> OrderedRingQueue {
        let Hnsw1024 { hnsw, vectors, .. } = self;
        let comparator = CosineDistance1024::new(vectors);
        hnsw.search_from_initial(query_vec, sp, &comparator)
    }

    fn test_recall_with_proportion(
        &self,
        proportion: f32,
        sp: &SearchParams,
        seed: u64,
    ) -> f32 {
        let Hnsw1024 { hnsw, vectors, .. } = self;
        let comparator = CosineDistance1024::new(vectors);
        hnsw.test_recall(proportion, sp, &comparator, seed)
    }

    fn optimize_and_save<P: AsRef<Path>>(
        &mut self,
        root: P,
        op: &OptimizationParams,
    ) -> Result<f32, io::Error> {
        let comparator = CosineDistance1024::new(&self.vectors);
        let mut recall = 0.0;
        for i in 0..self.hnsw.layers().len() {
            recall = self.hnsw.optimize_layer(i, op, &comparator);
            self.store_hnsw(&root)?;
        }
        Ok(recall)
    }

    fn optimize(&mut self, op: &OptimizationParams) -> f32 {
        let comparator = CosineDistance1024::new(&self.vectors);
        self.hnsw.optimize(op, &comparator)
    }

    fn knn_into_file<P: AsRef<Path>>(
        &self,
        k: usize,
        sp: SearchParams,
        path: P,
    ) {
        let comparator = CosineDistance1024::new(&self.vectors);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .unwrap();
        let record_size = k * (size_of::<(u32, f32)>());
        self.hnsw.knn(k, sp, comparator).for_each(|(i, mut pairs)| {
            pairs.resize(k, (u32::MAX, f32::MAX));
            let pairs_len = pairs.len();
            assert_eq!(pairs_len, k);
            let raw_data: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    pairs.as_ptr() as *const u8,
                    record_size,
                )
            };
            let address = (i as usize * record_size) as u64;
            file.write_all_at(raw_data, address).unwrap_or_else(|e| {
                panic!("writing to address: {address}: {e}")
            })
        });
    }

    fn find_distance_transitions(
        &self,
        fpp: FindPeaksParams,
    ) -> Vec<(f32, Peak<f32>)> {
        let comparator = CosineDistance1024::new(&self.vectors);
        self.hnsw.find_distance_transitions(fpp, comparator)
    }
}

impl Hnsw1536 {
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
        let comparator = CosineDistance1536::new(&vectors);
        let hnsw = Hnsw::generate(bp, &comparator);

        Hnsw1536 {
            name,
            hnsw,
            vectors,
        }
    }
}

impl Index for Hnsw1536 {
    fn num_vectors(&self) -> usize {
        self.vectors.num_vecs()
    }

    fn search(&self, query_vec: Vector, sp: &SearchParams) -> OrderedRingQueue {
        let Hnsw1536 { hnsw, vectors, .. } = self;
        let comparator = CosineDistance1536::new(vectors);
        hnsw.search_from_initial(query_vec, sp, &comparator)
    }

    fn test_recall_with_proportion(
        &self,
        proportion: f32,
        sp: &SearchParams,
        seed: u64,
    ) -> f32 {
        let Hnsw1536 { hnsw, vectors, .. } = self;
        let comparator = CosineDistance1536::new(vectors);
        hnsw.test_recall(proportion, sp, &comparator, seed)
    }

    fn optimize_and_save<P: AsRef<Path>>(
        &mut self,
        root: P,
        op: &OptimizationParams,
    ) -> Result<f32, io::Error> {
        let comparator = CosineDistance1536::new(&self.vectors);
        let mut recall = 0.0;
        for i in 0..self.hnsw.layers().len() {
            recall = self.hnsw.optimize_layer(i, op, &comparator);
            self.store_hnsw(&root)?;
        }
        Ok(recall)
    }

    fn optimize(&mut self, op: &OptimizationParams) -> f32 {
        let comparator = CosineDistance1536::new(&self.vectors);
        self.hnsw.optimize(op, &comparator)
    }

    fn knn_into_file<P: AsRef<Path>>(
        &self,
        k: usize,
        sp: SearchParams,
        path: P,
    ) {
        let comparator = CosineDistance1536::new(&self.vectors);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .unwrap();
        let record_size = k * (size_of::<(u32, f32)>());
        self.hnsw.knn(k, sp, comparator).for_each(|(i, mut pairs)| {
            pairs.resize(k, (u32::MAX, f32::MAX));
            let pairs_len = pairs.len();
            assert_eq!(pairs_len, k);
            let raw_data: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    pairs.as_ptr() as *const u8,
                    record_size,
                )
            };
            let address = (i as usize * record_size) as u64;
            file.write_all_at(raw_data, address).unwrap_or_else(|e| {
                panic!("writing to address: {address}: {e}")
            })
        });
    }

    fn find_distance_transitions(
        &self,
        fpp: FindPeaksParams,
    ) -> Vec<(f32, Peak<f32>)> {
        let comparator = CosineDistance1536::new(&self.vectors);
        self.hnsw.find_distance_transitions(fpp, comparator)
    }
}

impl IndexConfiguration {
    pub fn knn(
        &self,
        k: usize,
    ) -> impl ParallelIterator<Item = (u32, Vec<(u32, f32)>)> + '_ {
        match self {
            IndexConfiguration::Hnsw1024(index) => {
                let comparator = CosineDistance1024::new(&index.vectors);
                let sp = SearchParams::default();
                Either::Left(Either::Left(index.hnsw.knn(k, sp, comparator)))
            }
            IndexConfiguration::Hnsw1536(index) => {
                let comparator = CosineDistance1536::new(&index.vectors);
                let sp = SearchParams::default();
                Either::Right(index.hnsw.knn(k, sp, comparator))
            }
            IndexConfiguration::Pq1024x8(index) => {
                let quantized_comparator = NewMemoizedComparator128::new(
                    &index.pq.quantized_vectors,
                    &index.pq.memoized_distances,
                );
                let sp = SearchParams::default();
                Either::Left(Either::Right(index.pq.quantized_hnsw.knn(
                    k,
                    sp,
                    quantized_comparator,
                )))
            }
        }
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
        test_util::{random_vectors, random_vectors_normalized},
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
        let centroid_build_params = BuildParams {
            order: 24,
            neighborhood_size: 24,
            bottom_neighborhood_size: 48,
            optimization_params: OptimizationParams {
                search_params: SearchParams {
                    parallel_visit_count: 12,
                    visit_queue_len: 100,
                    search_queue_len: 30,
                    circulant_parameter_count: 8,
                },
                improvement_threshold: 0.01,
                recall_target: 1.0,
            },
        };
        let quantized_build_params = BuildParams {
            order: 24,
            neighborhood_size: 24,
            bottom_neighborhood_size: 48,
            optimization_params: OptimizationParams {
                search_params: SearchParams {
                    parallel_visit_count: 12,
                    visit_queue_len: 100,
                    search_queue_len: 30,
                    circulant_parameter_count: 8,
                },
                improvement_threshold: 0.01,
                recall_target: 1.0,
            },
        };
        let quantized_search_params = SearchParams {
            parallel_visit_count: 12,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 8,
        };

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
            quantized_search_params,
            0x533D,
        );

        let index = Pq1024x8 {
            pq,
            vectors: vecs,
            name: "dummy".to_string(),
        };
        let sp = SearchParams {
            parallel_visit_count: 12,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 8,
        };

        let recall = index.test_recall_with_proportion(0.10, &sp, 0x533D);
        eprintln!("recall: {recall}");
        assert!(recall > 0.95);
    }

    #[test]
    fn write_knn() {
        let number_of_vecs = 1_000;
        let vecs = random_vectors_normalized(number_of_vecs, 1024, 0x533D);
        let comparator = CosineDistance1024::new(&vecs);
        let bp = BuildParams {
            order: 24,
            neighborhood_size: 24,
            bottom_neighborhood_size: 48,
            optimization_params: OptimizationParams {
                search_params: SearchParams {
                    parallel_visit_count: 1,
                    visit_queue_len: 100,
                    search_queue_len: 30,
                    circulant_parameter_count: 8,
                },
                improvement_threshold: 0.01,
                recall_target: 1.0,
            },
        };
        let sp = SearchParams {
            parallel_visit_count: 12,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 8,
        };

        let hnsw = Hnsw::generate(&bp, &comparator);

        let mut index = Hnsw1024 {
            hnsw,
            vectors: vecs,
            name: "my_hnsw".to_string(),
        };
        let mut recall = 0.0;
        let mut last_recall = 0.0;
        let mut improvement = 1.0;
        while recall < 1.0 && improvement > 0.1 {
            index.optimize(&Default::default());
            recall = index.test_recall(&sp, 0x533D);
            eprintln!("recall: {recall}");
            improvement = recall - last_recall;
            last_recall = recall;
        }

        assert!(recall > 0.999);

        index.knn_into_file(20, sp, "/tmp/dump")
    }
}
