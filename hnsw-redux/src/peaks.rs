use find_peaks::{Peak, PeakFinder};
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::{
    hnsw::Hnsw,
    layer::{OrderedFloat, VectorComparator},
    params::{FindPeaksParams, SearchParams},
};

fn derivative(source: &[f32]) -> Vec<f32> {
    source
        .iter()
        .take(source.len() - 1)
        .zip(source.iter().skip(1))
        .map(|(l, r)| *r - *l)
        .collect()
}

fn find_distance_transitions_from_slice(
    number_of_peaks: usize,
    distances: &[f32],
) -> Vec<(f32, Peak<f32>)> {
    let first_derivative = derivative(&distances);

    let finder = PeakFinder::new(&first_derivative);
    let mut peaks = finder.find_peaks();

    peaks.par_sort_unstable_by_key(|p| OrderedFloat(p.height.unwrap()));

    peaks
        .into_iter()
        .take(number_of_peaks)
        .map(|p| {
            (
                (distances[p.position.start] + distances[p.position.end]) / 2.0,
                p,
            )
        })
        .collect()
}

impl Hnsw {
    pub fn find_distance_transitions<C: VectorComparator>(
        &self,
        params: FindPeaksParams,
        comparator: C,
    ) -> Vec<(f32, Peak<f32>)> {
        let mut distances: Vec<f32> = self
            .knn(params.results_per_search, params.search_params, comparator)
            .take_any(params.number_of_searches)
            .flat_map(|(_, v)| v.into_par_iter().map(|(_, elt)| elt))
            .collect();
        distances.par_sort_unstable_by_key(|v| OrderedFloat(*v));

        find_distance_transitions_from_slice(params.number_of_peaks, &distances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn find_peaks() {
        let data = vec![
            0.1, 0.101, 0.102, 0.3, 0.301, 0.302, 0.5, 0.55, 0.6, 0.8, 0.9,
        ];
        let results = find_distance_transitions_from_slice(4, &data);
        todo!();
    }
}
