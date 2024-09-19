use rayon::prelude::*;

use crate::{
    bitmap::Bitmap,
    ring_queue::{ring_double_insert, OrderedRingQueue},
};

pub struct Layer {
    neighborhoods: Vec<u32>,
    single_neighborhood_size: usize,
}

#[derive(Clone, Copy)]
pub enum InputVector<'a> {
    Id(u32),
    Data(&'a [u8]),
}

#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    pub parallel_visit_count: usize,
}

pub trait VectorComparator: Sync {
    fn compare_vec_stored(&self, left: u32, right: u32) -> f32;
    fn compare_vec_unstored(&self, stored: u32, unstored: &[u8]) -> f32;
}

impl Layer {
    pub fn new(neighborhoods: Vec<u32>, single_neighborhood_size: usize) -> Self {
        assert_eq!(0, neighborhoods.len() % single_neighborhood_size);
        Self {
            neighborhoods,
            single_neighborhood_size,
        }
    }
    pub fn number_of_neighborhoods(&self) -> usize {
        self.neighborhoods.len() / self.single_neighborhood_size
    }
    pub fn single_neighborhood_size(&self) -> usize {
        self.single_neighborhood_size
    }
    pub fn search_from_seeds<C: VectorComparator>(
        &self,
        query_vec: &[u8],
        visit_queue: &mut OrderedRingQueue,
        search_params: &SearchParams,
        ids_slice: &mut [u32],
        priorities_slice: &mut [f32],
        comparator: &C,
    ) -> usize {
        let (seeds, n_pops) = visit_queue.pop_first_n(search_params.parallel_visit_count);
        let number_of_neighbors = n_pops * self.single_neighborhood_size;
        let seeds_iter = seeds.into_par_iter().map(|(id, _)| id);
        let ids_iter =
            ids_slice[0..number_of_neighbors].par_chunks_mut(self.single_neighborhood_size);
        let priorities_iter =
            priorities_slice[0..number_of_neighbors].par_chunks_mut(self.single_neighborhood_size);

        let zipped = seeds_iter.zip(ids_iter.zip(priorities_iter));

        zipped
            .flat_map(|(neighborhood, (ids_chunk, priorities_chunk))| {
                ids_chunk
                    .into_par_iter()
                    .zip(priorities_chunk.into_par_iter())
                    .enumerate()
                    .map(move |(neighbor, (id_out, priority_out))| {
                        (neighborhood, neighbor, id_out, priority_out)
                    })
            })
            .for_each(|(neighborhood, neighbor, id_out, priority_out)| {
                let vector_id = self.neighborhoods
                    [neighborhood as usize * self.single_neighborhood_size + neighbor];
                let priority = comparator.compare_vec_unstored(vector_id, query_vec);
                *id_out = vector_id;
                *priority_out = priority;
            });

        n_pops
    }
    pub fn closest_vectors<C: VectorComparator>(
        &self,
        query_vec: &[u8],
        search_queue: &mut OrderedRingQueue,
        uninitalized_visit_queue: &mut OrderedRingQueue,
        ids_slice: &mut [u32],
        priorities_slice: &mut [f32],
        search_params: &SearchParams,
        comparator: &C,
    ) {
        uninitalized_visit_queue.reinit_from(search_queue);
        let visit_queue = uninitalized_visit_queue;

        // bitmap does mutate, but it's internal mutability
        let seen = Bitmap::new(self.number_of_neighborhoods());
        let mut did_something = true;

        while did_something {
            let n_pops = self.search_from_seeds(
                query_vec,
                visit_queue,
                search_params,
                ids_slice,
                priorities_slice,
                comparator,
            );

            if n_pops == 0 {
                break;
            }
            let actual_result_length = n_pops * self.single_neighborhood_size;
            let ids_found = &ids_slice[0..actual_result_length];
            let priorities_found = &priorities_slice[0..actual_result_length];

            did_something = ring_double_insert(
                visit_queue,
                search_queue,
                ids_found,
                priorities_found,
                &seen,
            );
            seen.set_from_ids(ids_found);
        }
    }
}
