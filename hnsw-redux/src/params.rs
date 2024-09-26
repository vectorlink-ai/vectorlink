#[derive(Clone, Copy, Debug)]
pub struct BuildParams {
    pub order: usize,
    pub neighborhood_size: usize,
    pub bottom_neighborhood_size: usize,
    pub optimize_sp: SearchParams,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            order: 24,
            neighborhood_size: 24,
            bottom_neighborhood_size: 48,
            optimize_sp: SearchParams {
                parallel_visit_count: 1,
                visit_queue_len: 100,
                search_queue_len: 30,
                circulant_parameter_count: 0,
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    pub parallel_visit_count: usize,
    pub visit_queue_len: usize,
    pub search_queue_len: usize,
    pub circulant_parameter_count: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            parallel_visit_count: 12,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 8,
        }
    }
}
