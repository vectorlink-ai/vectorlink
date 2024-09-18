use std::sync::atomic::{self, AtomicU64};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Bitmap that lets you atomically set entries, but reads entries
/// without synchronizing.
///
/// The upshot is that you might get outdated values, but reading can
/// be done quickly, without any synchronization overhead.
pub struct Bitmap {
    data: Vec<u64>,
}

impl Bitmap {
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![0; (len + 63) / 64],
        }
    }

    pub fn check(&self, index: usize) -> bool {
        let elt = self.data[index / 64];
        elt & (1 << (index % 64)) != 0
    }

    pub fn set(&self, index: usize) {
        // let's pretend this is actually an atomic
        let elt = &self.data[index / 64];

        unsafe {
            let cast: &AtomicU64 = AtomicU64::from_ptr(elt as *const u64 as *mut u64);
            cast.fetch_or(1 << (index % 64), atomic::Ordering::Relaxed);
        }
    }

    pub fn set_from_ids(&self, ids: &[u32]) {
        ids.par_iter().for_each(|id| self.set(*id as usize));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn do_a_bitmap_once() {
        let idx = 42;
        let bitmap = Bitmap::new(100);

        assert!(!bitmap.check(idx));
        assert!(!bitmap.check(idx + 1));
        bitmap.set(idx);
        assert!(bitmap.check(idx));
        assert!(!bitmap.check(idx + 1));
    }

    #[test]
    fn do_a_bitmap_set_all() {
        let bitmap = Bitmap::new(100);

        let to_set: Vec<u32> = vec![1, 1, 2, 3, 5, 8, 13, 21];

        bitmap.set_from_ids(&to_set);

        for i in 0..100 {
            assert_eq!(to_set.contains(&i), bitmap.check(i as usize), "{i}");
        }
    }
}
