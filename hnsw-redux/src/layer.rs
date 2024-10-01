use std::{
    ops::{Index, IndexMut},
    sync::Arc,
};

use itertools::Itertools;
use rayon::prelude::*;

use crate::{
    bitmap::Bitmap,
    memoize::{index_to_offset, triangle_lookup_length},
    optimize::LayerOptimizer,
    params::SearchParams,
    ring_queue::{ring_double_insert, OrderedRingQueue},
    util::SimdAlignedAllocation,
    vecmath::PRIMES,
    vectors::Vector,
};

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct OrderedFloat(pub f32);

impl Eq for OrderedFloat {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let res = self.partial_cmp(other);
        if res.is_none() {
            eprintln!("incomparable: {self:?} <> {other:?}");
            panic!();
        };
        res.unwrap()
    }
}

#[derive(Clone)]
pub struct Layer {
    neighborhoods: SimdAlignedAllocation<u32>,
    single_neighborhood_size: usize,
}

impl AsRef<Layer> for Layer {
    fn as_ref(&self) -> &Layer {
        self
    }
}

pub trait VectorComparator: Sync {
    fn compare_vecs_stored(&self, left: &[u32], right: u32, result: &mut [f32]);
    fn compare_vecs_stored_unstored(&self, stored: &[u32], unstored: &[u8], result: &mut [f32]);
    fn compare_vecs_unstored(&self, left: &[u8], right: &[u8], result: &mut [f32]);
    fn vec_group_size() -> usize;
    fn num_vecs(&self) -> usize;
    fn vector_byte_size() -> usize;

    fn compare_vec_stored(&self, left: u32, right: u32) -> f32 {
        if left == right {
            return 0.0;
        }
        // need to pad to group size
        let mut result = vec![0.0; Self::vec_group_size()];
        let left_list = vec![left; Self::vec_group_size()];
        self.compare_vecs_stored(&left_list, right, &mut result);

        result[0]
    }

    fn compare_vec_stored_unstored(&self, stored: u32, unstored: &[u8]) -> f32 {
        // need to pad to group size
        let mut result = vec![0.0; Self::vec_group_size()];
        let storeds = vec![stored; Self::vec_group_size()];
        self.compare_vecs_stored_unstored(&storeds, unstored, &mut result);

        result[0]
    }

    fn compare_vec_unstored(&self, left: &[u8], right: &[u8]) -> f32 {
        let mut result = vec![0.0; Self::vec_group_size()];
        // TODO we should probably have a vec_size fn for comparators too
        let mut lefts = vec![0; left.len() * Self::vec_group_size()];
        lefts[0..left.len()].copy_from_slice(left);
        self.compare_vecs_unstored(&lefts, right, &mut result);

        result[0]
    }

    fn compare_vecs_vector(&self, left: &[u32], right: Vector, result: &mut [f32]) {
        match right {
            Vector::Slice(slice) => self.compare_vecs_stored_unstored(left, slice, result),
            Vector::Id(id) => self.compare_vecs_stored(left, id, result),
        }
    }

    fn compare_vec_vector(&self, left: u32, right: Vector) -> f32 {
        match right {
            Vector::Slice(slice) => self.compare_vec_stored_unstored(left, slice),
            Vector::Id(id) => self.compare_vec_stored(left, id),
        }
    }
}

pub trait VectorGrouper: Sync {
    fn vector_group(&self, vec: u32) -> usize;
    fn num_groups(&self) -> usize;
}

pub trait VectorSearcher: Sync {
    fn search(&self, vec: u32) -> OrderedRingQueue;
}

impl Layer {
    pub fn new(neighborhoods: SimdAlignedAllocation<u32>, single_neighborhood_size: usize) -> Self {
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

    pub fn from_data(data: SimdAlignedAllocation<u8>, single_neighborhood_size: usize) -> Self {
        assert_eq!(
            0,
            data.len() % (std::mem::size_of::<u32>() * single_neighborhood_size),
            "data is not a multiple of neighborhood size"
        );

        Self::new(data.cast_to(), single_neighborhood_size)
    }

    pub fn data(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.neighborhoods.as_ptr() as *const u8,
                self.neighborhoods.len() * std::mem::size_of::<u32>(),
            )
        }
    }

    pub fn neighborhood_distances<C: VectorComparator>(
        &self,
        comparator: &C,
    ) -> SimdAlignedAllocation<f32> {
        let len = self.number_of_neighborhoods() * self.single_neighborhood_size;
        let mut distances: SimdAlignedAllocation<f32> =
            unsafe { SimdAlignedAllocation::alloc(len) };

        (0..self.number_of_neighborhoods())
            .into_par_iter()
            .map(|vector_id| &self[vector_id])
            .zip(distances.par_chunks_mut(self.single_neighborhood_size))
            .enumerate()
            .flat_map(|(i, (neighborhoods, distances))| {
                neighborhoods
                    .par_chunks(C::vec_group_size())
                    .zip(distances.par_chunks_mut(C::vec_group_size()))
                    .map(move |(n, d)| (i, n, d))
            })
            .for_each(|(i, neighborhood_chunk, distances_chunk)| {
                comparator.compare_vecs_stored(neighborhood_chunk, i as u32, distances_chunk);
            });

        distances
    }

    pub fn search_from_seeds<C: VectorComparator>(
        &self,
        query_vec: Vector,
        visit_queue: &mut OrderedRingQueue,
        search_params: &SearchParams,
        ids_slice: &mut [u32],
        priorities_slice: &mut [f32],
        comparator: &C,
    ) -> usize {
        let total_neighborhood_size =
            self.single_neighborhood_size + search_params.circulant_parameter_count;
        debug_assert!(
            total_neighborhood_size >= C::vec_group_size()
                && total_neighborhood_size % C::vec_group_size() == 0,
            "comparator takes a vector group size that doesn't cleanly fit into the neighborhood size"
        );
        let (seeds, n_pops) = visit_queue.pop_first_n(search_params.parallel_visit_count);
        let number_of_neighbors = n_pops * total_neighborhood_size;
        let seeds_iter = seeds.into_iter().map(|(id, _)| id);
        let ids_iter = ids_slice[0..number_of_neighbors].chunks_mut(total_neighborhood_size);
        let priorities_iter =
            priorities_slice[0..number_of_neighbors].chunks_mut(total_neighborhood_size);

        let zipped = seeds_iter.zip(ids_iter.zip(priorities_iter));

        zipped
            .flat_map(|(neighborhood, (ids_chunk, priorities_chunk))| {
                ids_chunk
                    .chunks_mut(C::vec_group_size())
                    .zip(priorities_chunk.chunks_mut(C::vec_group_size()))
                    .enumerate()
                    .map(move |(neighbor_group, (ids_out, priorities_out))| {
                        (neighborhood, neighbor_group, ids_out, priorities_out)
                    })
            })
            .for_each(|(neighborhood, neighbor_group, ids_out, priorities_out)| {
                if neighbor_group < self.single_neighborhood_size / C::vec_group_size() {
                    let neighbor_offset = neighborhood as usize * self.single_neighborhood_size
                        + neighbor_group * C::vec_group_size();
                    let vector_ids =
                        &self.neighborhoods[neighbor_offset..neighbor_offset + C::vec_group_size()];
                    comparator.compare_vecs_vector(vector_ids, query_vec, priorities_out);
                    ids_out.copy_from_slice(vector_ids);
                } else {
                    let prime_group =
                        neighbor_group - self.single_neighborhood_size / C::vec_group_size();
                    let primes_offset = prime_group * C::vec_group_size();
                    let mut vector_ids = vec![0; C::vec_group_size()];
                    for (idx, prime) in PRIMES[primes_offset..primes_offset + C::vec_group_size()]
                        .iter()
                        .enumerate()
                    {
                        vector_ids[idx] =
                            (neighborhood + *prime as u32) % self.number_of_neighborhoods() as u32;
                    }
                    comparator.compare_vecs_vector(&vector_ids, query_vec, priorities_out);
                    ids_out.copy_from_slice(&vector_ids);
                }
            });

        n_pops
    }

    #[allow(clippy::too_many_arguments)]
    pub fn closest_vectors<C: VectorComparator>(
        &self,
        query_vec: Vector,
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
        let mut seen = Bitmap::new(self.number_of_neighborhoods());
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
            let actual_result_length =
                n_pops * (self.single_neighborhood_size + search_params.circulant_parameter_count);
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

    pub fn copy_neighbors(&self, vector_id: u32) -> Vec<u32> {
        let mut result = vec![0_u32; self.single_neighborhood_size];

        self.copy_neighbors_into(vector_id, &mut result);

        result
    }

    pub fn copy_neighbors_into(&self, vector_id: u32, result: &mut [u32]) {
        debug_assert_eq!(result.len(), self.single_neighborhood_size());
        let offset = self.single_neighborhood_size * vector_id as usize;
        result.copy_from_slice(&self.neighborhoods[offset..offset + self.single_neighborhood_size]);
    }

    pub fn par_extract_neighbors(&self, vector_ids: &[u32], results: &mut [u32]) {
        assert_eq!(
            results.len(),
            vector_ids.len() * self.single_neighborhood_size()
        );
        vector_ids
            .par_iter()
            .zip(results.par_chunks_mut(self.single_neighborhood_size))
            .for_each(|(&vector_id, neighborhood_out)| {
                self.copy_neighbors_into(vector_id, neighborhood_out);
            });
    }

    /// pass in search_vector_ids and results with plenty of space
    /// reserved. it'll safely resize, but that takes time.
    ///
    /// this will not automatially set all search_vector_ids in the
    /// `seen` bitmap, except if any neighbors point back at them.
    /// It is the caller's responsibility to properly initialize the
    /// bitmap, if necessary.
    ///
    /// There are cases where it is not, such as if we know this
    /// already happend. Or if we actually want to check if any of its
    /// neighbors connect back.
    ///
    /// This function modifies both vector id lists. At the end,
    /// results will contain the outer convolution ring of the
    /// flood. search_vector_ids will contain the second-to last
    /// convolution ring. Every convolution will have been set in the
    /// `seen` bitmap.
    pub fn par_flood_find_neighbors(
        &self,
        distance: usize,
        search_vector_ids: &mut Vec<u32>,
        results: &mut Vec<u32>,
        seen: &mut Bitmap,
    ) {
        results.clear();
        let mut swapped = false;
        for i in 0..distance {
            // figure out how many results we're going to get, and set
            // the length of the result vector accordingly.
            let len = self.single_neighborhood_size * search_vector_ids.len();
            #[allow(clippy::uninit_vec)]
            unsafe {
                results.reserve(len - results.len());
                results.set_len(len);
            }

            // get the next convolution of neighbors out
            eprintln!("iteration {i}: search queue is {}", search_vector_ids.len());
            self.par_extract_neighbors(search_vector_ids, results);

            // register results with the 'seen' bitmap.
            // this will also modify duplicate results to u32::MAX
            eprintln!("iteration {i}: found {} connecteds", results.len());
            seen.check_set_from_ids(results);

            // sort the results, and dedup to eliminate double
            // results.
            results.par_sort_unstable();
            results.dedup();

            // if the last element is u32::MAX, we don't want it.
            if results.last() == Some(&u32::MAX) {
                results.pop();
            }

            // Turn results into the search vectors for the next convolution.
            std::mem::swap(search_vector_ids, results);
            swapped = !swapped;
        }

        // Ensure the returned results are actually in the right vector.
        // This makes sure that we can reliably get the outer convolution ring.
        if swapped {
            std::mem::swap(search_vector_ids, results);
        }
    }

    pub fn par_find_unconnected_neighbors(
        &self,
        distance: usize,
        search_vector_ids: &mut Vec<u32>,
        result_vector_ids: &mut Vec<u32>,
    ) {
        result_vector_ids.clear();
        let mut seen = Bitmap::new(self.number_of_neighborhoods());
        seen.set_from_ids(search_vector_ids);

        // find out what /is/ connected.
        self.par_flood_find_neighbors(distance, search_vector_ids, result_vector_ids, &mut seen);

        // invert that, so we know the unconnecteds
        seen.par_invert();

        // read the seens back out into the results vec.
        result_vector_ids.clear();
        result_vector_ids.par_extend(seen.par_iter_ids());
    }

    pub fn build_perfect<C: VectorComparator>(
        num_vecs: usize,
        single_neighborhood_size: usize,
        comparator: &C,
    ) -> Self {
        // safety note: While not specified as mut here, we'll be
        // modifying results unsafely below.
        // This is safe, because it is a set without get (no
        // synchronization issues), and each set will go to a unique
        // offset.
        let triangle_len = triangle_lookup_length(num_vecs);
        let distances = vec![0.0_f32; triangle_len];
        (0..num_vecs as u32)
            .into_par_iter()
            .flat_map(|i| {
                (i..num_vecs as u32)
                    .into_par_iter()
                    .chunks(C::vec_group_size())
                    .map(move |js| (i, js))
            })
            .for_each(|(i, js)| {
                let offset = index_to_offset(num_vecs, i as usize, js[0] as usize);
                let results: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        distances.as_ptr().add(offset) as *mut f32,
                        js.len(),
                    )
                };
                if js.len() == C::vec_group_size() {
                    comparator.compare_vecs_stored(&js, i, results);
                } else {
                    // the final chunk might be too short for the
                    // group size of the comparator. In this case, we
                    // need padded buffers.
                    let mut inputs = vec![0_u32; C::vec_group_size()];
                    let mut temp_results = vec![0.0_f32; C::vec_group_size()];
                    inputs[0..js.len()].copy_from_slice(&js);
                    comparator.compare_vecs_stored(&inputs, i, &mut temp_results);

                    results.copy_from_slice(&temp_results[0..js.len()]);
                }
            });

        // we got a cross product, let's use it to construct perfect neighborhoods.

        let size = num_vecs * single_neighborhood_size;
        let mut neighborhoods: SimdAlignedAllocation<u32> =
            unsafe { SimdAlignedAllocation::alloc(size) };
        neighborhoods[..size]
            .par_chunks_mut(single_neighborhood_size)
            .enumerate()
            .for_each(|(neighborhood, neighborhood_slice)| {
                // collect our best matches
                let mut distances_for_vec: Vec<_> = (0..num_vecs as u32)
                    .filter(|v| *v != neighborhood as u32)
                    .map(|v| {
                        (
                            v,
                            distances[index_to_offset(num_vecs, neighborhood, v as usize)],
                        )
                    })
                    .collect();
                distances_for_vec.sort_unstable_by_key(|(n, x)| (OrderedFloat(*x), *n));
                for i in 0..single_neighborhood_size {
                    let ptr = &mut neighborhood_slice[i];

                    if i >= distances_for_vec.len() {
                        *ptr = u32::MAX;
                    } else {
                        *ptr = distances_for_vec[i].0;
                    }
                }
            });

        Self {
            neighborhoods,
            single_neighborhood_size,
        }
    }

    pub fn build_random<C: VectorComparator>(
        num_vecs: usize,
        single_neighborhood_size: usize,
        comparator: &C,
    ) -> (Self, SimdAlignedAllocation<f32>) {
        let size = num_vecs * single_neighborhood_size;
        let mut neighborhoods: SimdAlignedAllocation<u32> =
            unsafe { SimdAlignedAllocation::alloc(size) };
        neighborhoods[..size]
            .par_chunks_mut(single_neighborhood_size)
            .enumerate()
            .for_each(|(idx, neighborhood)| {
                for (i, n) in neighborhood.iter_mut().enumerate() {
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

        let mut result = Self {
            neighborhoods,
            single_neighborhood_size,
        };

        let distances = result.sort_neighborhoods(comparator);

        (result, distances)
    }

    pub fn build_grouped<G: VectorGrouper, C: VectorComparator>(
        num_vecs: usize,
        single_neighborhood_size: usize,
        grouper: &G,
        comparator: &C,
    ) -> Self {
        let size = num_vecs * single_neighborhood_size;
        let mut neighborhoods: SimdAlignedAllocation<u32> =
            unsafe { SimdAlignedAllocation::alloc(size) };
        let neighborhoods_iter = neighborhoods[..size].par_chunks_mut(single_neighborhood_size);
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
                    let new = (idx + PRIMES[i % PRIMES.len()]) % vec_ids.len();
                    // We might have accidentally selected
                    // ourselves, need to shift to another prime
                    if new == idx {
                        *n = vec_ids[(idx + PRIMES[(i + 1) % PRIMES.len()]) % vec_ids.len()];
                    } else {
                        *n = vec_ids[new]
                    }
                }
            });

        let mut result = Self {
            neighborhoods,
            single_neighborhood_size,
        };

        //let distances = result.sort_neighborhoods(comparator);

        result
    }

    pub fn sort_neighborhoods<C: VectorComparator>(
        &mut self,
        comparator: &C,
    ) -> SimdAlignedAllocation<f32> {
        println!("sort neighborhood");
        let mut memoized_distances = self.neighborhood_distances(comparator);
        self.neighborhoods
            .par_chunks_mut(self.single_neighborhood_size)
            .zip(memoized_distances.par_chunks_mut(self.single_neighborhood_size))
            .enumerate()
            .for_each(|(vector_id, (neighborhood, distances))| {
                let mut pairs: Vec<_> = neighborhood
                    .iter()
                    .copied()
                    .zip(distances.iter().copied())
                    .collect();
                pairs.sort_unstable_by_key(|p| (OrderedFloat(p.1), p.0));
                pairs.dedup();
                let pairs_len = pairs.len();
                for (ix, (n, d)) in pairs.into_iter().enumerate() {
                    neighborhood[ix] = n;
                    distances[ix] = d;
                }
                for ix in pairs_len..self.single_neighborhood_size {
                    neighborhood[ix] = u32::MAX;
                    distances[ix] = f32::MAX;
                }
            });

        memoized_distances
    }

    pub fn get_optimizer<'a>(
        &'a mut self,
        distances: &'a mut SimdAlignedAllocation<f32>,
    ) -> LayerOptimizer<'a> {
        LayerOptimizer::new(
            &mut self.neighborhoods,
            distances,
            self.single_neighborhood_size,
        )
    }

    pub fn knn<'a, C: VectorComparator>(
        &'a self,
        k: usize,
        sp: &'a SearchParams,
        comparator: &'a C,
    ) -> impl ParallelIterator<Item = (u32, Vec<(u32, f32)>)> + 'a {
        let neighborhood_size = self.single_neighborhood_size();
        let node_count = self.number_of_neighborhoods();
        let buffer_size =
            neighborhood_size * (sp.parallel_visit_count + sp.circulant_parameter_count);
        (0..node_count as u32).into_par_iter().map(move |node_id| {
            let mut uninitialized_visit_queue = OrderedRingQueue::new(sp.search_queue_len);
            let mut search_queue =
                OrderedRingQueue::new_with(sp.search_queue_len, &[node_id], &[0.0]);
            let mut ids = Vec::with_capacity(buffer_size);
            let mut priorities = Vec::with_capacity(buffer_size);
            unsafe {
                ids.set_len(buffer_size);
                priorities.set_len(buffer_size);
            }

            self.closest_vectors(
                Vector::Id(node_id),
                &mut search_queue,
                &mut uninitialized_visit_queue,
                &mut ids,
                &mut priorities,
                sp,
                comparator,
            );
            let mut results = search_queue.get_all();
            results.truncate(k);
            (node_id, results)
        })
    }
}

impl Index<usize> for Layer {
    type Output = [u32];

    fn index(&self, index: usize) -> &Self::Output {
        let offset = index * self.single_neighborhood_size;
        &self.neighborhoods[offset..offset + self.single_neighborhood_size]
    }
}

impl IndexMut<usize> for Layer {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let offset = index * self.single_neighborhood_size;
        &mut self.neighborhoods[offset..offset + self.single_neighborhood_size]
    }
}

#[cfg(test)]
mod tests {

    use crate::{comparator::EuclideanDistance8x8, hnsw::Hnsw, test_util::random_vectors};

    use super::*;

    #[test]
    fn construct_perfect_layer() {
        let vecs = random_vectors(24, 8, 0x533D);
        let comparator = EuclideanDistance8x8::new(&vecs);
        let layer = Layer::build_perfect(24, 24, &comparator);

        let hnsw = Hnsw::new(vec![layer]);
        let result = hnsw.search_from_initial(
            Vector::Id(5),
            &SearchParams {
                parallel_visit_count: 1,
                visit_queue_len: 100,
                search_queue_len: 30,
                circulant_parameter_count: 0,
            },
            &comparator,
        );

        let left: &[f32; 8] = vecs.get(5).unwrap();
        for (result, distance) in result.iter() {
            let right: &[f32; 8] = vecs.get(result as usize).unwrap();
            let expected = left
                .iter()
                .zip(right.iter())
                .map(|(l, r)| (l - r).powi(2))
                .sum::<f32>()
                .sqrt();

            assert!((expected - distance).abs() < 0.001);
        }
    }
}
