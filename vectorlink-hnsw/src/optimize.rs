use std::sync::{Mutex, MutexGuard};

use rayon::prelude::*;

use crate::{layer::VectorSearcher, queue_view::QueueView, util::SimdAlignedAllocation};

pub struct LayerOptimizer<'a> {
    queues: Vec<Mutex<QueueView<'a>>>,
    neighborhood_size: usize,
}

impl<'a> LayerOptimizer<'a> {
    pub fn new(
        neighbors: &'a mut SimdAlignedAllocation<u32>,
        distances: &'a mut SimdAlignedAllocation<f32>,
        neighborhood_size: usize,
    ) -> Self {
        assert_eq!(0, neighborhood_size % 8);
        assert_eq!(neighbors.len(), distances.len());
        assert_eq!(0, neighbors.len() % neighborhood_size);

        let queues: Vec<_> = neighbors
            .chunks_mut(neighborhood_size)
            .zip(distances.chunks_mut(neighborhood_size))
            .map(|(n, d)| Mutex::new(QueueView::new(n, d)))
            .collect();

        Self {
            queues,
            neighborhood_size,
        }
    }

    pub fn copy_neighborhood(&self, vector_id: u32) -> (Vec<u32>, Vec<f32>) {
        let mut neighbors = Vec::with_capacity(self.neighborhood_size);
        let mut distances = Vec::with_capacity(self.neighborhood_size);

        {
            let guard = self.get(vector_id);
            neighbors.extend_from_slice(guard.neighbors);
            distances.extend_from_slice(guard.distances);
        }

        (neighbors, distances)
    }

    pub fn get<'b>(&'b self, vector_id: u32) -> MutexGuard<'b, QueueView<'a>> {
        self.queues[vector_id as usize].lock().unwrap()
    }

    pub fn symmetrize(&mut self) {
        eprintln!("symmetrize: symmetrize neighborhoods");
        // symmetrize neighborhoods
        self.queues.par_iter().enumerate().for_each(|(i, q)| {
            // local copy
            let (neighbors, distances) = q.lock().unwrap().copy_neighborhood();
            for (neighbor, distance) in neighbors
                .into_iter()
                .zip(distances)
                .filter(|(n, _)| *n != u32::MAX)
            {
                //eprintln!("inserting into: {}", neighbor.0);
                self.get(neighbor).insert((i as u32, distance));
            }
        });
    }

    pub fn improve_all_neighbors<S: VectorSearcher>(&mut self, searcher: &S) {
        self.improve_neighbors(searcher, (0..self.queues.len() as u32).into_par_iter())
    }

    pub fn improve_neighbors<S: VectorSearcher, I: ParallelIterator<Item = u32>>(
        &mut self,
        searcher: &S,
        iter: I,
    ) {
        eprintln!("improve neighbors: optimizing neighborhoods");
        // optimize neighborhoods
        iter.for_each(|query_vector_id| {
            let results = searcher.search(query_vector_id);
            if query_vector_id == 45 {
                //eprintln!("improve_neighbors found {:?}", results);
            }
            for (found_vector_id, priority) in results.iter() {
                let new_pair = (query_vector_id, priority);
                if query_vector_id == found_vector_id {
                    continue;
                }
                //eprintln!("inserting into: {}", neighbor.0);
                let mut destination_queue = self.get(found_vector_id);
                let _result = destination_queue.insert(new_pair);
            }
        });
    }
}
