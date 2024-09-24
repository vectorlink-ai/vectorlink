use crate::{
    layer::{Layer, SearchParams, VectorComparator, VectorGrouper},
    ring_queue::OrderedRingQueue,
    vectors::Vector,
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
        query_vec: Vector,
        search_queue: &mut OrderedRingQueue,
        sp: &SearchParams,
        comparator: &C,
    ) {
        Hnsw::search_layers(&self.layers, query_vec, search_queue, sp, comparator);
    }

    fn search_layers<C: VectorComparator>(
        layers: &[Layer],
        query_vec: Vector,
        search_queue: &mut OrderedRingQueue,
        sp: &SearchParams,
        comparator: &C,
    ) {
        let layer_count = layers.len();
        assert!(layer_count > 0);
        let bottom_layer_idx = layer_count - 1;
        let largest_neighborhood = layers[bottom_layer_idx].single_neighborhood_size();
        let buffer_size = largest_neighborhood * sp.parallel_visit_count;
        let mut ids = Vec::with_capacity(buffer_size);
        let mut priorities = Vec::with_capacity(buffer_size);
        unsafe {
            ids.set_len(buffer_size);
            priorities.set_len(buffer_size);
        }

        let mut uninitialized_visit_queue = OrderedRingQueue::new(sp.search_queue_len);

        for layer in layers.iter() {
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

    pub fn generate<C: VectorComparator>(
        order: usize,
        single_neighborhood_size: usize,
        num_vecs: usize,
        comparator: &C,
    ) -> Self {
        let mut layer_count = order;
        let zero_layer = Layer::build_perfect(layer_count, single_neighborhood_size, comparator);
        eprintln!("zero layer built");
        let layers = vec![zero_layer];
        let mut grouper = SearchGrouper { comparator, layers };
        while layer_count <= num_vecs {
            layer_count *= order;
            eprintln!("layer_count: {layer_count}");
            let last_layer = layer_count >= num_vecs;
            let vec_count = if last_layer { num_vecs } else { layer_count };
            let single_neighborhood_size = if last_layer {
                single_neighborhood_size * 2
            } else {
                single_neighborhood_size
            };
            eprintln!("vec_count: {vec_count}");
            let mut new_layer = Layer::build_grouped(vec_count, single_neighborhood_size, &grouper);
            new_layer.symmetrize(comparator);

            grouper.push(new_layer);

            layer_count *= order;
        }
        Hnsw::new(grouper.layers())
    }
}

pub struct SearchGrouper<'a, C> {
    comparator: &'a C,
    layers: Vec<Layer>,
}

impl<'a, C> SearchGrouper<'a, C> {
    pub fn push(&mut self, l: Layer) {
        self.layers.push(l);
    }

    pub fn layers(self) -> Vec<Layer> {
        self.layers
    }
}

impl<'a, C: VectorComparator> VectorGrouper for SearchGrouper<'a, C> {
    fn vector_group(&self, vec: u32) -> usize {
        let sp = SearchParams {
            parallel_visit_count: 4,
            visit_queue_len: 32,
            search_queue_len: 16,
        };
        eprintln!("vec: {vec}");
        let initial_distance = self.comparator.compare_vec_stored(0, vec);
        eprintln!("initial distance: {initial_distance}");
        let mut search_queue =
            OrderedRingQueue::new_with(sp.search_queue_len, &[0], &[initial_distance]);
        eprintln!("searching layers");
        Hnsw::search_layers(
            &self.layers,
            Vector::Id(vec),
            &mut search_queue,
            &sp,
            self.comparator,
        );
        eprintln!("{search_queue:?}");
        search_queue.pop_first().0 as usize
    }

    fn num_groups(&self) -> usize {
        let result = self.layers.last().unwrap();
        result.number_of_neighborhoods()
    }
}
