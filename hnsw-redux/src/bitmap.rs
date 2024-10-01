use std::simd::u64x64;

use rayon::prelude::*;

use crate::util::SimdAlignedAllocation;

/// Bitmap that lets you atomically set entries, but reads entries
/// without synchronizing.
///
/// The upshot is that you might get outdated values, but reading can
/// be done quickly, without any synchronization overhead.
pub struct Bitmap {
    data: SimdAlignedAllocation<u64>,
    len: usize,
}

impl Clone for Bitmap {
    fn clone(&self) -> Self {
        let u64_len = (self.len + 4095) / 64;
        let mut data = unsafe { SimdAlignedAllocation::alloc(u64_len) };
        data.copy_from_slice(&self.data[..]);
        Self {
            data,
            len: self.len,
        }
    }
}

impl Bitmap {
    pub fn new(len: usize) -> Self {
        let u64_len = (len + 4095) / 64;
        let data = unsafe { SimdAlignedAllocation::alloc(u64_len) };
        Self { data, len }
    }

    #[inline(always)]
    fn check_elt(elt: u64, index: usize) -> bool {
        elt & (1 << (index % 64)) != 0
    }

    #[inline(always)]
    pub fn check(&self, index: usize) -> bool {
        debug_assert!(index < self.len);
        let elt = self.data[index / 64];
        Self::check_elt(elt, index % 64)
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize) {
        if index == u32::MAX as usize {
            return;
        }
        debug_assert!(index < self.len);
        let elt = &mut self.data[index / 64];
        *elt |= 1 << (index % 64);

        /*
        unsafe {
            let cast: &AtomicU64 = AtomicU64::from_ptr(elt as *const u64 as *mut u64);
            cast.fetch_or(1 << (index % 64), atomic::Ordering::Relaxed);
        }
        */
    }

    pub fn set_from_ids(&mut self, ids: &[u32]) {
        ids.iter().for_each(|id| self.set(*id as usize));
    }

    /// returns the previous value, or true for out of band
    #[inline(always)]
    pub fn check_set(&mut self, index: usize) -> bool {
        if index == u32::MAX as usize {
            return true;
        }
        debug_assert!(index < self.len);
        let elt = &mut self.data[index / 64];
        let result = Self::check_elt(*elt, index % 64);
        *elt |= 1 << (index % 8);

        result
    }

    /// set_from_ids. retains ids in the argument if they were not yet
    /// set. changes them to out of band (u32::MAX) if they were
    /// already set.
    pub fn check_set_from_ids(&mut self, ids: &mut [u32]) {
        for id in ids {
            if self.check_set(*id as usize) {
                *id = u32::MAX;
            }
        }
    }

    #[inline(always)]
    pub fn invert(&mut self) {
        let simds: &mut [u64x64] = self.data.as_simd_mut();
        for simd in simds {
            *simd = !*simd;
        }
    }

    pub fn par_invert(&mut self) {
        let simds: &mut [u64x64] = self.data.as_simd_mut();

        simds.par_iter_mut().for_each(|simd| *simd = !*simd);
    }

    #[inline(always)]
    pub fn invert_into(&self, other: &mut Bitmap) {
        assert_eq!(self.len, other.len);
        let self_simds: &[u64x64] = self.data.as_simd();
        let other_simds: &mut [u64x64] = other.data.as_simd_mut();

        for (self_simd, other_simd) in self_simds.iter().zip(other_simds.iter_mut()) {
            *other_simd = !*self_simd;
        }
    }

    pub fn par_invert_into(&self, other: &mut Bitmap) {
        assert_eq!(self.len, other.len);
        let self_simds: &[u64x64] = self.data.as_simd();
        let other_simds: &mut [u64x64] = other.data.as_simd_mut();

        for (self_simd, other_simd) in self_simds.iter().zip(other_simds.iter_mut()) {
            *other_simd = !*self_simd;
        }
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        self.data
            .iter()
            .flat_map(|elt| (0..8).map(move |index| Self::check_elt(*elt, index)))
    }

    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = bool> + '_ {
        (0..self.len).into_par_iter().map(|index| self.check(index))
    }

    #[inline(always)]
    pub fn iter_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.len as u32).filter(|&i| self.check(i as usize))
    }

    pub fn par_iter_ids(&self) -> impl ParallelIterator<Item = u32> + '_ {
        (0..self.len as u32)
            .into_par_iter()
            .filter(|&i| self.check(i as usize))
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};

    use super::*;
    #[test]
    fn do_a_bitmap_once() {
        let idx = 42;
        let mut bitmap = Bitmap::new(100);

        assert!(!bitmap.check(idx));
        assert!(!bitmap.check(idx + 1));
        bitmap.set(idx);
        assert!(bitmap.check(idx));
        assert!(!bitmap.check(idx + 1));
    }

    #[test]
    fn do_a_bitmap_set_all() {
        let mut bitmap = Bitmap::new(100);

        let to_set: Vec<u32> = vec![1, 1, 2, 3, 5, 8, 13, 21];

        bitmap.set_from_ids(&to_set);

        for i in 0..100 {
            assert_eq!(to_set.contains(&i), bitmap.check(i as usize), "{i}");
        }
    }

    #[test]
    fn clone_bitmap() {
        const LEN: usize = 12345;
        let mut bitmap = Bitmap::new(LEN);
        let mut rng = StdRng::seed_from_u64(0x533D);
        let to_set: Vec<u32> = (0..LEN as u32).choose_multiple(&mut rng, 200);
        bitmap.set_from_ids(&to_set);

        let cloned = bitmap.clone();

        for i in 0..LEN {
            assert_eq!(bitmap.check(i), cloned.check(i));
        }
    }

    #[test]
    fn bitmap_serial_invert() {
        const LEN: usize = 12345000;
        let mut bitmap = Bitmap::new(LEN);
        let mut rng = StdRng::seed_from_u64(0x533D);
        let to_set: Vec<u32> = (0..LEN as u32).choose_multiple(&mut rng, 200);
        bitmap.set_from_ids(&to_set);
        let original = bitmap.clone();

        bitmap.invert();

        for i in 0..LEN {
            assert_ne!(bitmap.check(i), original.check(i));
        }
    }

    #[test]
    fn bitmap_par_invert() {
        const LEN: usize = 12345000;
        let mut bitmap = Bitmap::new(LEN);
        let mut rng = StdRng::seed_from_u64(0x533D);
        let to_set: Vec<u32> = (0..LEN as u32).choose_multiple(&mut rng, 200);
        bitmap.set_from_ids(&to_set);
        let original = bitmap.clone();

        bitmap.par_invert();

        for i in 0..LEN {
            assert_ne!(bitmap.check(i), original.check(i));
        }
    }
}
