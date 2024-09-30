use std::{
    alloc::{self, Layout},
    ops::{Deref, DerefMut},
    ptr::NonNull,
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

/// Align allocations to be rust simd friendly.
/// The biggest possible simd abstration is f64x64, which is 512
/// bytes. So this will make sure we're aligned to that, allowing simd instructions over the entire structure.
/// It also means the allocation will round up, making sure we can read an even number of 512 byte chunks.
pub struct SimdAlignedAllocation {
    data: NonNull<u8>,
    len: usize,
}

unsafe impl Send for SimdAlignedAllocation {}
unsafe impl Sync for SimdAlignedAllocation {}

impl SimdAlignedAllocation {
    const fn rounded_len(len: usize) -> usize {
        (len + 511) / 512 * 512
    }
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(Self::rounded_len(len), 512).unwrap()
    }
    pub unsafe fn alloc(len: usize) -> Self {
        let layout = Self::layout(len);
        let data = unsafe { alloc::alloc(layout) };
        if data.is_null() {
            alloc::handle_alloc_error(layout);
        }
        let data = unsafe { NonNull::new_unchecked(data) };
        Self { data, len }
    }

    pub fn alloc_zeroed(len: usize) -> Self {
        let layout = Self::layout(len);
        let data = unsafe { alloc::alloc_zeroed(layout) };
        if data.is_null() {
            alloc::handle_alloc_error(layout);
        }
        let data = unsafe { NonNull::new_unchecked(data) };
        Self { data, len }
    }

    pub fn as_simd<T, const LANES: usize>(&self) -> &[Simd<T, LANES>]
    where
        LaneCount<LANES>: SupportedLaneCount,
        T: SimdElement,
    {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const Simd<T, LANES>,
                Self::rounded_len(self.len) / (std::mem::size_of::<T>() * LANES),
            )
        }
    }

    pub fn as_simd_mut<T, const LANES: usize>(&mut self) -> &mut [Simd<T, LANES>]
    where
        LaneCount<LANES>: SupportedLaneCount,
        T: SimdElement,
    {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_ptr() as *mut Simd<T, LANES>,
                Self::rounded_len(self.len) / (std::mem::size_of::<T>() * LANES),
            )
        }
    }
}

impl Deref for SimdAlignedAllocation {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

impl DerefMut for SimdAlignedAllocation {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
}

impl Drop for SimdAlignedAllocation {
    fn drop(&mut self) {
        let layout = Self::layout(self.len);
        unsafe { alloc::dealloc(self.data.as_ptr(), layout) }
    }
}

impl Clone for SimdAlignedAllocation {
    fn clone(&self) -> Self {
        let layout = Self::layout(self.len);
        unsafe {
            let data = alloc::alloc(layout);
            if data.is_null() {
                alloc::handle_alloc_error(layout);
            }
            let data = NonNull::new(data).unwrap();
            data.copy_from_nonoverlapping(self.data, layout.size());

            Self {
                data,
                len: self.len,
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocate_simd_aligned() {
        let data = SimdAlignedAllocation::alloc_zeroed(12345);
        assert_eq!(data.as_ptr() as usize % 256, 0);
    }
}
