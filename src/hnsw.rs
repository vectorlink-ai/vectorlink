use crate::{
    layer::{Layer, VectorComparator, VectorGrouper, VectorSearcher},
    params::{BuildParams, SearchParams},
    ring_queue::OrderedRingQueue,
    vectors::Vector,
};

use rand::prelude::*;
use rayon::prelude::*;

pub struct Hnsw {
    layers: Vec<Layer>,
}

impl Hnsw {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn search<C: VectorComparator>(
        &self,
        query_vec: Vector,
        search_queue: &mut OrderedRingQueue,
        sp: &SearchParams,
        comparator: &C,
    ) {
        Hnsw::search_layers(&self.layers, query_vec, search_queue, sp, comparator);
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
        let largest_neighborhood = layers[bottom_layer_idx].as_ref().single_neighborhood_size();
        let buffer_size =
            largest_neighborhood * (sp.parallel_visit_count + sp.circulant_parameter_count);
        debug_assert!(buffer_size % C::vec_group_size() == 0);
        let mut ids = Vec::with_capacity(buffer_size);
        let mut priorities = Vec::with_capacity(buffer_size);
        unsafe {
            ids.set_len(buffer_size);
            priorities.set_len(buffer_size);
        }

        let mut uninitialized_visit_queue = OrderedRingQueue::new(sp.search_queue_len);

        for layer in layers.iter() {
            layer.as_ref().closest_vectors(
                query_vec,
                search_queue,
                &mut uninitialized_visit_queue,
                &mut ids,
                &mut priorities,
                sp,
                comparator,
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

        let mut search_queue =
            OrderedRingQueue::new_with(sp.search_queue_len, &[0], &[initial_distance]);
        self.search(query_vec, &mut search_queue, sp, comparator);

        search_queue
    }

    pub fn generate<C: VectorComparator>(bp: &BuildParams, comparator: &C) -> Self {
        let num_vecs = comparator.num_vecs();
        assert!(num_vecs > bp.order);
        let mut layer_nodes = bp.order;
        let zero_layer = Layer::build_perfect(layer_nodes, bp.neighborhood_size, comparator);
        eprintln!("perfect first layer built");
        let mut layers = vec![zero_layer];
        while layer_nodes <= num_vecs {
            layer_nodes *= bp.order;
            eprintln!("layer nodes: {layer_nodes}");
            let last_layer = layer_nodes >= num_vecs;
            let vec_count = if last_layer { num_vecs } else { layer_nodes };
            let single_neighborhood_size = if last_layer {
                bp.bottom_neighborhood_size
            } else {
                bp.neighborhood_size
            };
            eprintln!("vec_count: {vec_count}");
            let grouper = SearchGrouper {
                comparator,
                layers: &layers,
                sp: &bp.optimize_sp,
            };
            let mut new_layer = Layer::build_grouped(vec_count, single_neighborhood_size, &grouper);
            new_layer.symmetrize(comparator);
            layers.push(new_layer.clone());
            let grouper = SearchGrouper {
                comparator,
                layers: &layers,
                sp: &bp.optimize_sp,
            };
            new_layer.improve_neighbors(comparator, &grouper);
            *layers.last_mut().unwrap() = new_layer;

            layer_nodes *= bp.order;
        }
        Hnsw::new(layers)
    }

    pub fn improve_neighbors_in_all_layers<C: VectorComparator>(
        &mut self,
        optimize_sp: &SearchParams,
        comparator: &C,
    ) {
        for i in 1..self.layers.len() {
            let (top, bottom) = self.layers.split_at_mut(i);
            let pseudo_layer = bottom[0].clone();
            let mut pseudo_layers: Vec<&Layer> = Vec::with_capacity(top.len() + 1);
            pseudo_layers.extend(top.iter());
            pseudo_layers.push(&pseudo_layer);

            let searcher = SearchGrouper {
                comparator,
                layers: &pseudo_layers,
                sp: optimize_sp,
            };

            bottom[0].improve_neighbors(comparator, &searcher);
        }
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
        let mut rng = StdRng::seed_from_u64(seed);
        let total: f32 = (0..self.num_vectors() as u32)
            .choose_multiple(&mut rng, (proportion * self.num_vectors() as f32) as usize)
            .into_par_iter()
            .map(|i| {
                let result = self.search_from_initial(Vector::Id(i), sp, comparator);
                let vi = result.first();
                if vi.0 == i {
                    1.0_f32
                } else {
                    0.0_f32
                }
            })
            .sum();
        total / self.num_vectors() as f32
    }

    pub fn get_layer_mut(&mut self, layer_id: usize) -> &mut Layer {
        assert!(layer_id < self.layers.len());
        &mut self.layers[layer_id]
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

pub struct SearchGrouper<'a, C, L: AsRef<Layer>> {
    comparator: &'a C,
    sp: &'a SearchParams,
    layers: &'a [L],
}

impl<'a, C: VectorComparator, L: AsRef<Layer> + Sync> VectorSearcher for SearchGrouper<'a, C, L> {
    fn search(&self, vec: u32) -> OrderedRingQueue {
        let initial_distance = self.comparator.compare_vec_stored(0, vec);

        let mut search_queue =
            OrderedRingQueue::new_with(self.sp.search_queue_len, &[0], &[initial_distance]);
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

impl<'a, C: VectorComparator, L: AsRef<Layer> + Sync> VectorGrouper for SearchGrouper<'a, C, L> {
    fn vector_group(&self, vec: u32) -> usize {
        let sp = SearchParams {
            parallel_visit_count: 4,
            visit_queue_len: 32,
            search_queue_len: 16,
            circulant_parameter_count: 0,
        };
        let initial_distance = self.comparator.compare_vec_stored(0, vec);
        let mut search_queue =
            OrderedRingQueue::new_with(sp.search_queue_len, &[0], &[initial_distance]);
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
        comparator::{CosineDistance1024, EuclideanDistance8x8},
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
        let sp = SearchParams::default();

        for i in 0..10 {
            let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
            eprintln!("{i}: {recall}");
            if recall == 1.0 {
                break;
            }
            hnsw.improve_neighbors_in_all_layers(&Default::default(), &comparator);
        }
        let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
        assert_eq!(recall, 1.0);
    }

    #[test]
    fn construct_unquantized_1024_hnsw() {
        let number_of_vecs = 1000;
        let vecs = random_vectors_normalized::<1024>(number_of_vecs, 0x533D);
        let comparator = CosineDistance1024::new(&vecs);
        let bp = BuildParams::default();
        let mut hnsw = Hnsw::generate(&bp, &comparator);
        let sp = SearchParams::default();

        for i in 0..10 {
            let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
            eprintln!("{i}: {recall}");
            if recall == 1.0 {
                break;
            }
            hnsw.improve_neighbors_in_all_layers(&Default::default(), &comparator);
        }
        let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
        assert_eq!(recall, 1.0);
    }

    #[test]
    #[ignore]
    fn construct_unquantized_1536_hnsw() {
        let number_of_vecs = 100_000;
        let vecs = random_vectors_normalized::<1536>(number_of_vecs, 0x533D);
        let comparator = CosineDistance1024::new(&vecs);
        let bp = BuildParams::default();
        let mut hnsw = Hnsw::generate(&bp, &comparator);
        let mut sp = SearchParams::default();
        sp.circulant_parameter_count = 8;
        sp.parallel_visit_count = 12;

        for i in 0..10 {
            let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
            eprintln!("{i}: {recall}");
            if recall == 1.0 {
                break;
            }
            hnsw.improve_neighbors_in_all_layers(&Default::default(), &comparator);
        }
        let recall = hnsw.test_recall(1.0, &sp, &comparator, 0x533D);
        assert!(recall > 0.999);
    }
}
