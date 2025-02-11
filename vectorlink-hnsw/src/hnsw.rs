use crate::{
    layer::{
        Layer, VectorComparator, VectorGrouper, VectorRecall, VectorSearcher,
    },
    params::{BuildParams, OptimizationParams, SearchParams},
    ring_queue::OrderedRingQueue,
    vectors::Vector,
};

use rand::prelude::*;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Hnsw {
    layers: Vec<Layer>,
}

impl Hnsw {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn into_inner(self) -> Vec<Layer> {
        self.layers
    }

    #[allow(unused)]
    fn with_temp_hnsw<R, F: Fn(&Hnsw) -> R>(
        layers: &mut Vec<Layer>,
        func: F,
    ) -> R {
        let mut temp_layers = Vec::new();
        std::mem::swap(&mut temp_layers, layers);
        let hnsw = Hnsw::new(temp_layers);
        let result = func(&hnsw);
        temp_layers = hnsw.into_inner();
        std::mem::swap(&mut temp_layers, layers);

        result
    }

    pub fn search<C: VectorComparator>(
        &self,
        query_vec: Vector,
        search_queue: &mut OrderedRingQueue,
        sp: &SearchParams,
        comparator: &C,
    ) {
        Hnsw::search_layers(
            &self.layers,
            query_vec,
            search_queue,
            sp,
            comparator,
        );
    }

    fn search_layers<C: VectorComparator, L: AsRef<Layer>>(
        layers: &[L],
        query_vec: Vector,
        search_queue: &mut OrderedRingQueue,
        sp: &SearchParams,
        comparator: &C,
    ) {
        let layer_count = layers.len();
        assert!(layer_count > 0);
        let bottom_layer_idx = layer_count - 1;
        let largest_neighborhood =
            layers[bottom_layer_idx].as_ref().single_neighborhood_size();
        let buffer_size = largest_neighborhood
            * (sp.parallel_visit_count + sp.circulant_parameter_count);
        debug_assert!(buffer_size % C::vec_group_size() == 0);
        let mut ids = Vec::with_capacity(buffer_size);
        let mut priorities = Vec::with_capacity(buffer_size);
        unsafe {
            ids.set_len(buffer_size);
            priorities.set_len(buffer_size);
        }

        let mut uninitialized_visit_queue =
            OrderedRingQueue::new(sp.search_queue_len);

        for layer in layers.iter() {
            layer.as_ref().closest_vectors(
                query_vec,
                search_queue,
                &mut uninitialized_visit_queue,
                &mut ids,
                &mut priorities,
                sp,
                comparator.clone(),
            )
        }
    }

    pub fn search_from_initial<C: VectorComparator>(
        &self,
        query_vec: Vector,
        sp: &SearchParams,
        comparator: &C,
    ) -> OrderedRingQueue {
        // find initial distance from the 0th vec, which is our fixed start node
        let initial_distance = comparator.compare_vec_vector(0, query_vec);

        let mut search_queue = OrderedRingQueue::new_with(
            sp.search_queue_len,
            &[0],
            &[initial_distance],
        );
        self.search(query_vec, &mut search_queue, sp, comparator);

        search_queue
    }

    pub fn generate<C: VectorComparator>(
        bp: &BuildParams,
        comparator: &C,
    ) -> Self {
        let num_vecs = comparator.num_vecs();
        eprintln!("num_vecs: {num_vecs}");
        assert!(num_vecs > bp.order);
        let mut layer_nodes = bp.order;
        let zero_layer =
            Layer::build_perfect(layer_nodes, bp.neighborhood_size, comparator);
        eprintln!("perfect first layer built");
        let mut layers = vec![zero_layer];

        while num_vecs > layer_nodes {
            layer_nodes *= bp.order;
            eprintln!("layer nodes: {layer_nodes}");
            let last_layer = layer_nodes >= num_vecs;
            let vec_count = if last_layer { num_vecs } else { layer_nodes };
            let single_neighborhood_size = if last_layer {
                bp.bottom_neighborhood_size
            } else {
                bp.neighborhood_size
            };
            eprintln!("neighborhood size: {single_neighborhood_size}");
            eprintln!("vec_count: {vec_count}");
            let grouper = SearchGrouper {
                comparator,
                layers: &layers,
                sp: &bp.optimization_params.search_params,
            };
            let mut new_layer = Layer::build_grouped(
                vec_count,
                single_neighborhood_size,
                &grouper,
            );

            let mut memoized_distances =
                new_layer.sort_neighborhoods(comparator);

            // we are going to push the buffer in a second, so layers.len()+1
            {
                let mut optimizer =
                    new_layer.get_optimizer(&mut memoized_distances);
                eprintln!("symmetrizing layer {}", layers.len() + 1);
                optimizer.symmetrize();
            }

            layers.push(new_layer.clone());
            let grouper = SearchGrouper {
                comparator,
                layers: &layers,
                sp: &bp.optimization_params.search_params,
            };
            eprintln!("improving layer {}", layers.len());
            let mut optimizer =
                new_layer.get_optimizer(&mut memoized_distances);
            optimizer.improve_all_neighbors(&grouper);

            *layers.last_mut().unwrap() = new_layer;
        }
        Hnsw::new(layers)
    }

    pub fn optimize<C: VectorComparator>(
        &mut self,
        op: &OptimizationParams,
        comparator: &C,
    ) -> f32 {
        let mut recall = 0.0;
        for i in 0..self.layers().len() {
            recall = self.optimize_layer(i, op, comparator);
        }
        recall
    }

    pub fn optimize_layer<C: VectorComparator>(
        &mut self,
        layer_i: usize,
        optimize_params: &OptimizationParams,
        comparator: &C,
    ) -> f32 {
        let (top, bottom) = self.layers.split_at_mut(layer_i);
        let pseudo_layer = bottom[0].clone();
        let mut pseudo_layers: Vec<&Layer> = Vec::with_capacity(top.len() + 1);
        pseudo_layers.extend(top.iter());
        pseudo_layers.push(&pseudo_layer);

        let searcher = SearchGrouper {
            comparator,
            layers: &pseudo_layers,
            sp: &optimize_params.search_params,
        };

        let mut distances = bottom[0].neighborhood_distances(comparator);
        let mut optimizer = bottom[0].get_optimizer(&mut distances);

        let mut recall = 0.0;
        let mut improvement = 1.0;
        let mut round = 0;
        let vector_count = pseudo_layer.number_of_neighborhoods();
        let proportion = 1.0 / (vector_count as f32).sqrt();
        while recall < optimize_params.recall_target
            && improvement > optimize_params.improvement_threshold
        {
            optimizer.improve_all_neighbors(&searcher);

            let new_recall = searcher
                .test_recall(proportion, 0x533D + layer_i as u64 + round);
            improvement = new_recall - recall;
            recall = new_recall;
            eprintln!("layer[{layer_i}]\n  Recall: {recall}\n  Improvement: {improvement}");
            round += 1;
        }
        recall
    }

    pub fn num_vectors(&self) -> usize {
        self.layers.last().unwrap().number_of_neighborhoods()
    }

    pub fn test_recall<C: VectorComparator>(
        &self,
        proportion: f32,
        sp: &SearchParams,
        comparator: &C,
        seed: u64,
    ) -> f32 {
        eprintln!("proportion: {proportion}");
        let mut rng = StdRng::seed_from_u64(seed);
        let ids: Vec<u32> = if proportion == 1.0 {
            (0..self.num_vectors() as u32).collect()
        } else {
            (0..self.num_vectors() as u32).choose_multiple(
                &mut rng,
                (proportion * self.num_vectors() as f32) as usize,
            )
        };
        let total = ids.len();
        eprintln!("searching for {total} vecs..");
        let found: f32 = ids
            .into_par_iter()
            .map(|i| {
                let result =
                    self.search_from_initial(Vector::Id(i), sp, comparator);
                let vi = result.first();
                if vi.0 == i {
                    1.0_f32
                } else {
                    0.0_f32
                }
            })
            .sum();
        eprintln!("found {found}.");
        found / total as f32
    }

    pub fn get_layer_mut(&mut self, layer_id: usize) -> &mut Layer {
        assert!(layer_id < self.layers.len());
        &mut self.layers[layer_id]
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn knn<'a, C: VectorComparator + 'a>(
        &'a self,
        k: usize,
        sp: SearchParams,
        comparator: C,
    ) -> impl ParallelIterator<Item = (u32, Vec<(u32, f32)>)> + 'a {
        assert!(!self.layers.is_empty());
        let bottom_layer = self.layers.last().unwrap();
        bottom_layer.knn(k, sp, comparator)
    }
}

pub struct SearchGrouper<'a, C, L: AsRef<Layer>> {
    comparator: &'a C,
    sp: &'a SearchParams,
    layers: &'a [L],
}

impl<'a, C: VectorComparator, L: AsRef<Layer> + Sync> VectorSearcher
    for SearchGrouper<'a, C, L>
{
    fn search(&self, vec: u32) -> OrderedRingQueue {
        let initial_distance = self.comparator.compare_vec_stored(0, vec);

        let mut search_queue = OrderedRingQueue::new_with(
            self.sp.search_queue_len,
            &[0],
            &[initial_distance],
        );
        Hnsw::search_layers(
            self.layers,
            Vector::Id(vec),
            &mut search_queue,
            self.sp,
            self.comparator,
        );

        search_queue
    }
}

impl<'a, C: VectorComparator, L: AsRef<Layer> + Sync> VectorRecall
    for SearchGrouper<'a, C, L>
{
    fn test_recall(&self, proportion: f32, seed: u64) -> f32 {
        eprintln!("proportion: {proportion}");
        let mut rng = StdRng::seed_from_u64(seed);
        let num_vectors: u32 = self
            .layers
            .last()
            .unwrap()
            .as_ref()
            .number_of_neighborhoods() as u32;
        let ids: Vec<u32> = if proportion == 1.0 {
            (0..num_vectors).collect()
        } else {
            (0..num_vectors).choose_multiple(
                &mut rng,
                (proportion * num_vectors as f32) as usize,
            )
        };
        let total = ids.len();
        eprintln!("searching for {total} vecs..");
        let found: f32 = ids
            .into_par_iter()
            .map(|i| {
                let result = self.search(i);
                let vi = result.first();
                if vi.0 == i {
                    1.0_f32
                } else {
                    0.0_f32
                }
            })
            .sum();
        eprintln!("found {found}.");
        found / total as f32
    }
}

impl<'a, C: VectorComparator, L: AsRef<Layer> + Sync> VectorGrouper
    for SearchGrouper<'a, C, L>
{
    fn vector_group(&self, vec: u32) -> usize {
        let sp = SearchParams {
            parallel_visit_count: 4,
            visit_queue_len: 32,
            search_queue_len: 16,
            circulant_parameter_count: 0,
        };
        let initial_distance = self.comparator.compare_vec_stored(0, vec);
        let mut search_queue = OrderedRingQueue::new_with(
            sp.search_queue_len,
            &[0],
            &[initial_distance],
        );
        Hnsw::search_layers(
            self.layers,
            Vector::Id(vec),
            &mut search_queue,
            &sp,
            self.comparator,
        );
        search_queue.pop_first().0 as usize
    }

    fn num_groups(&self) -> usize {
        let result = self.layers.last().unwrap();
        result.as_ref().number_of_neighborhoods()
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        comparator::{
            CosineDistance1024, CosineDistance1536, EuclideanDistance8x8,
        },
        hnsw::Hnsw,
        test_util::{random_vectors, random_vectors_normalized},
    };

    use super::*;

    #[test]
    fn construct_centroid_hnsw() {
        let number_of_vecs = 1000;
        let vecs = random_vectors(number_of_vecs, 8, 0x533D);
        let comparator = EuclideanDistance8x8::new(&vecs);
        let bp = BuildParams::default();
        let mut hnsw = Hnsw::generate(&bp, &comparator);
        let sp = SearchParams {
            parallel_visit_count: 12,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 8,
        };

        for i in 0..10 {
            let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
            eprintln!("{i}: {recall}");
            if recall == 1.0 {
                break;
            }
            hnsw.optimize(&Default::default(), &comparator);
        }
        let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
        assert_eq!(recall, 1.0);
    }

    #[test]
    fn construct_unquantized_1024_hnsw() {
        let number_of_vecs = 1000;
        let vecs = random_vectors_normalized(number_of_vecs, 1024, 0x533D);
        let comparator = CosineDistance1024::new(&vecs);
        let bp = BuildParams::default();
        let mut hnsw = Hnsw::generate(&bp, &comparator);
        let sp = SearchParams {
            parallel_visit_count: 12,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 8,
        };

        for i in 0..10 {
            let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
            eprintln!("{i}: {recall}");
            if recall == 1.0 {
                break;
            }
            hnsw.optimize(&Default::default(), &comparator);
        }
        let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
        assert_eq!(recall, 1.0);
    }

    #[test]
    #[ignore]
    fn construct_unquantized_1536_hnsw() {
        let number_of_vecs = 100_000;
        let vecs = random_vectors_normalized(number_of_vecs, 1536, 0x533D);
        let comparator = CosineDistance1536::new(&vecs);
        let bp = BuildParams::default();
        let mut hnsw = Hnsw::generate(&bp, &comparator);
        let sp = SearchParams {
            parallel_visit_count: 12,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 8,
        };

        for i in 0..10 {
            let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
            eprintln!("{i}: {recall}");
            if recall == 1.0 {
                break;
            }
            hnsw.optimize(&Default::default(), &comparator);
        }
        let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
        assert!(recall > 0.999);
    }
}
