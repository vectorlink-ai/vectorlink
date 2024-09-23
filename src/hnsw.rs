use crate::{
    layer::{Layer, SearchParams, VectorComparator},
    ring_queue::OrderedRingQueue,
};

pub struct Hnsw {
    layers: Vec<Layer>,
}

impl Hnsw {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn search<C: VectorComparator>(
        &self,
        query_vec: &[u8],
        search_queue: &mut OrderedRingQueue,
        sp: &SearchParams,
        comparator: &C,
    ) {
        let layer_count = self.layers.len();
        assert!(layer_count > 0);
        let bottom_layer_idx = layer_count - 1;
        let largest_neighborhood = self.layers[bottom_layer_idx].single_neighborhood_size();
        let buffer_size = largest_neighborhood * sp.parallel_visit_count;
        let mut ids = Vec::with_capacity(buffer_size);
        let mut priorities = Vec::with_capacity(buffer_size);
        unsafe {
            ids.set_len(buffer_size);
            priorities.set_len(buffer_size);
        }

        let mut uninitialized_visit_queue = OrderedRingQueue::new(sp.search_queue_len);

        for layer in self.layers.iter() {
            layer.closest_vectors(
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
        query_vec: &[u8],
        sp: &SearchParams,
        comparator: &C,
    ) -> OrderedRingQueue {
        // find initial distance from the 0th vec, which is our fixed start node
        let initial_distance = comparator.compare_vec_stored_unstored(0, query_vec);
        let mut search_queue =
            OrderedRingQueue::new_with(sp.search_queue_len, &[0], &[initial_distance]);
        self.search(query_vec, &mut search_queue, sp, comparator);

        search_queue
    }
}
