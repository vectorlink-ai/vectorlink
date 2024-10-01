use std::{
    fmt::Debug,
    simd::{cmp::SimdPartialOrd, f32x8, u32x8},
};

#[derive(Debug)]
pub struct QueueView<'a> {
    pub(crate) neighbors: &'a mut [u32],
    pub(crate) distances: &'a mut [f32],
}

impl<'a> QueueView<'a> {
    pub fn new(neighbors: &'a mut [u32], distances: &'a mut [f32]) -> Self {
        assert_eq!(neighbors.len(), distances.len());
        assert_eq!(
            0,
            neighbors.as_ptr() as usize % std::mem::size_of::<u32x8>()
        );
        assert_eq!(
            0,
            distances.as_ptr() as usize % std::mem::size_of::<u32x8>()
        );
        assert_eq!(0, neighbors.len() % 8);
        Self {
            neighbors,
            distances,
        }
    }

    fn distances_simd(&mut self) -> &mut [f32x8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.distances.as_ptr() as *mut f32x8,
                self.distances.len() / 8,
            )
        }
    }

    pub fn insert(&mut self, pair: (u32, f32)) -> bool {
        let distances = self.distances_simd();
        let comparison = f32x8::splat(pair.1);
        for (simd_idx, simd) in distances.iter().enumerate() {
            let cmp = simd.simd_ge(comparison);
            if let Some(idx) = cmp.first_set() {
                // done with the simd part.
                // skip forward until we're at a good point
                let mut idx = simd_idx * 8 + idx;
                while idx < self.neighbors.len()
                    && self.distances[idx] <= pair.1
                    && self.neighbors[idx] < pair.0
                {
                    idx += 1;
                }
                if idx == self.neighbors.len() || self.neighbors[idx] == pair.0 {
                    // bail if we ran past the end or ended up at ourselves.
                    return false;
                }

                let len = self.neighbors.len();
                if idx != len - 1 {
                    // shift over everything by one position
                    self.neighbors.copy_within(idx..(len - 1), idx + 1);
                    self.distances.copy_within(idx..(len - 1), idx + 1);
                }

                // finally set ourselves
                self.neighbors[idx] = pair.0;
                self.distances[idx] = pair.1;

                return true;
            }
        }

        false
    }

    pub fn copy_neighborhood(&self) -> (Vec<u32>, Vec<f32>) {
        let mut neighbors = Vec::with_capacity(self.neighbors.len());
        let mut distances = Vec::with_capacity(self.neighbors.len());

        {
            neighbors.extend_from_slice(self.neighbors);
            distances.extend_from_slice(self.distances);
        }

        (neighbors, distances)
    }
}

#[cfg(test)]
mod tests {
    use crate::util::SimdAlignedAllocation;

    use super::*;

    #[test]
    fn insert_into_empty() {
        let mut empty_neighbors = SimdAlignedAllocation::alloc_default(24, u32::MAX);
        let mut empty_distances = SimdAlignedAllocation::alloc_default(24, f32::MAX);

        let mut queue = QueueView::new(&mut empty_neighbors, &mut empty_distances);
        queue.insert((42, 0.5));
        queue.insert((42, 0.5));
        queue.insert((41, 0.5));
        queue.insert((123, 0.3));
        queue.insert((345, 0.7));

        assert_eq!([123, 41, 42, 345, u32::MAX], empty_neighbors[0..5]);
        assert_eq!([0.3, 0.5, 0.5, 0.7, f32::MAX], empty_distances[0..5]);
    }

    #[test]
    fn insert_into_full_fails() {
        let mut neighbors = SimdAlignedAllocation::alloc_default(8, u32::MAX);
        let mut distances = SimdAlignedAllocation::alloc_default(8, f32::MAX);

        let mut queue = QueueView::new(&mut neighbors, &mut distances);
        // fill up queue first
        for i in 0..8 {
            queue.insert((i, i as f32));
        }

        // too big! should not fit
        queue.insert((10, 100.0));

        for i in 0..8 {
            assert_eq!(i as u32, neighbors[i]);
            assert_eq!(i as f32, distances[i]);
        }
    }

    #[test]
    fn insert_into_full_succeeds() {
        let mut neighbors = SimdAlignedAllocation::alloc_default(8, u32::MAX);
        let mut distances = SimdAlignedAllocation::alloc_default(8, f32::MAX);

        let mut queue = QueueView::new(&mut neighbors, &mut distances);
        // fill up queue first
        for i in 0..8 {
            queue.insert((i, i as f32));
        }

        // should go right into the middle
        queue.insert((10, 4.5));

        for i in 0..4 {
            assert_eq!(i as u32, neighbors[i]);
            assert_eq!(i as f32, distances[i]);
        }
        assert_eq!(10, neighbors[5]);
        assert_eq!(4.5, distances[5]);
        for i in 6..8 {
            assert_eq!((i - 1) as u32, neighbors[i]);
            assert_eq!((i - 1) as f32, distances[i]);
        }
    }
}
