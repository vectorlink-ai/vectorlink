use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct BuildParams {
    pub order: usize,
    pub neighborhood_size: usize,
    pub bottom_neighborhood_size: usize,
    pub optimization_params: OptimizationParams,
}

impl Default for BuildParams {
    fn default() -> Self {
        let build_params = Self {
            order: 24,
            neighborhood_size: 24,
            bottom_neighborhood_size: 48,
            optimization_params: OptimizationParams {
                search_params: SearchParams {
                    parallel_visit_count: 12,
                    visit_queue_len: 300,
                    search_queue_len: 100,
                    circulant_parameter_count: 0,
                },
                improvement_threshold: 0.01,
                recall_target: 1.0,
            },
        };
        build_params
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
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

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct OptimizationParams {
    pub search_params: SearchParams,
    pub improvement_threshold: f32,
    pub recall_target: f32,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            search_params: SearchParams::default(),
            improvement_threshold: 0.01,
            recall_target: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct FindPeaksParams {
    pub number_of_searches: usize,
    pub results_per_search: usize,
    pub number_of_peaks: usize,
    pub search_params: SearchParams,
}

impl Default for FindPeaksParams {
    fn default() -> Self {
        Self {
            number_of_searches: 300,
            results_per_search: 20,
            number_of_peaks: 5,
            search_params: Default::default(),
        }
    }
}
