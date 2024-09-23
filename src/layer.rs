use std::sync::Arc;

use itertools::Itertools;
use rayon::prelude::*;

use crate::{
    bitmap::Bitmap,
    ring_queue::{ring_double_insert, OrderedRingQueue},
    vecmath::PRIMES,
};

pub struct Layer {
    neighborhoods: Vec<u32>,
    single_neighborhood_size: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    pub parallel_visit_count: usize,
    pub visit_queue_len: usize,
    pub search_queue_len: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            parallel_visit_count: 1,
            visit_queue_len: 100,
            search_queue_len: 300,
        }
    }
}

pub trait VectorComparator: Sync {
    fn compare_vec_stored(&self, left: u32, right: u32) -> f32;
    fn compare_vec_unstored(&self, stored: u32, unstored: &[u8]) -> f32;
}

pub trait VectorGrouper: Sync {
    fn vector_group(&self, vec: u32) -> usize;
    fn num_groups(&self) -> usize;
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
    #[allow(clippy::too_many_arguments)]
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

    pub fn build_random(num_vecs: usize, single_neighborhood_size: usize) -> Self {
        let size = num_vecs * single_neighborhood_size;
        let mut neighborhoods: Vec<u32> = Vec::with_capacity(size);
        neighborhoods.spare_capacity_mut()[..size]
            .par_chunks_mut(single_neighborhood_size)
            .enumerate()
            .for_each(|(idx, neighborhood)| {
                for (i, n) in neighborhood.iter_mut().enumerate() {
                    let n = unsafe { n.assume_init_mut() };
                    let new = ((idx + PRIMES[i % PRIMES.len()]) % num_vecs) as u32;
                    // We might have accidentally selected
                    // ourselves, need to shift to another prime
                    if new == idx as u32 {
                        *n = ((idx + PRIMES[(i + 1) % PRIMES.len()]) % num_vecs) as u32;
                    } else {
                        *n = new
                    }
                }
            });
        unsafe {
            neighborhoods.set_len(size);
        }

        Self {
            neighborhoods,
            single_neighborhood_size,
        }
    }

    pub fn build_grouped<G: VectorGrouper>(
        num_vecs: usize,
        single_neighborhood_size: usize,
        grouper: &G,
    ) -> Self {
        let size = num_vecs * single_neighborhood_size;
        let mut neighborhoods: Vec<u32> = Vec::with_capacity(size);
        let neighborhoods_iter =
            neighborhoods.spare_capacity_mut()[..size].par_chunks_mut(single_neighborhood_size);
        let mut grouped_vecs: Vec<_> = (0..num_vecs as u32)
            .into_par_iter()
            .zip(neighborhoods_iter)
            .map(|(v, n)| (grouper.vector_group(v), v, n))
            .collect();

        grouped_vecs.par_sort_unstable_by_key(|(g, _, _)| *g);

        let groups: Vec<_> = grouped_vecs
            .into_iter()
            .chunk_by(|(g, _, _)| *g)
            .into_iter()
            .map(|(_, g)| {
                g.map(|(_, vec, neighborhood)| (vec, neighborhood))
                    .collect::<Vec<_>>()
            })
            .collect();

        groups
            .into_par_iter()
            .flat_map(|g| {
                let (vec_ids, neighborhoods): (Vec<_>, Vec<_>) = g.into_iter().unzip();
                let vec_ids = Arc::new(vec_ids);
                neighborhoods
                    .into_par_iter()
                    .enumerate()
                    .map(move |(idx, neighborhood)| (idx, neighborhood, vec_ids.clone()))
            })
            .for_each(|(idx, neighborhood, vec_ids)| {
                for (i, n) in neighborhood.iter_mut().enumerate() {
                    let n = unsafe { n.assume_init_mut() };
                    let new = (idx + PRIMES[i % PRIMES.len()]) % num_vecs;
                    // We might have accidentally selected
                    // ourselves, need to shift to another prime
                    if new == idx {
                        *n = vec_ids[(idx + PRIMES[(i + 1) % PRIMES.len()]) % num_vecs];
                    } else {
                        *n = vec_ids[new]
                    }
                }
            });

        unsafe {
            neighborhoods.set_len(size);
        }

        Self {
            neighborhoods,
            single_neighborhood_size,
        }
    }
}
