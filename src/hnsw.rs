use crate::{
    layer::{Layer, SearchParams, VectorComparator},
    ring_queue::OrderedRingQueue,
};

pub struct Hnsw {
    layers: Vec<Layer>,
}

impl Hnsw {
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

        // TODO figure out a good count for this one
        let mut uninitialized_visit_queue = OrderedRingQueue::new(search_queue.len() * 3);

        for layer in self.layers.iter() {
            layer.closest_vectors(
                query_vec,
                search_queue,
                &mut uninitialized_visit_queue,
                &mut ids,
                &mut priorities,
                &sp,
                comparator,
            )
        }
    }
}
