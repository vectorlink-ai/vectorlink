use std::{
    alloc::{self, Layout},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

use datafusion::arrow::alloc::Allocation;

/// Align allocations to be rust simd friendly.
/// The biggest possible simd abstration is f64x64, which is 512
/// bytes. So this will make sure we're aligned to that, allowing simd instructions over the entire structure.
/// It also means the allocation will round up, making sure we can read an even number of 512 byte chunks.
pub struct SimdAlignedAllocation<T: SimdElement> {
    data: NonNull<u8>,
    len: usize,
    _owned: PhantomData<T>,
}

unsafe impl<T: SimdElement> Send for SimdAlignedAllocation<T> {}
unsafe impl<T: SimdElement> Sync for SimdAlignedAllocation<T> {}

impl<T: SimdElement> SimdAlignedAllocation<T> {
    const fn rounded_len(len: usize) -> usize {
        (len * std::mem::size_of::<T>() + 511) / 512 * 512
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
        Self {
            data,
            len,
            _owned: PhantomData,
        }
    }

    pub fn alloc_zeroed(len: usize) -> Self {
        let layout = Self::layout(len);
        let data = unsafe { alloc::alloc_zeroed(layout) };
        if data.is_null() {
            alloc::handle_alloc_error(layout);
        }
        let data = unsafe { NonNull::new_unchecked(data) };
        Self {
            data,
            len,
            _owned: PhantomData,
        }
    }

    pub fn alloc_default(len: usize, default_value: T) -> Self {
        let layout = Self::layout(len);
        let data = unsafe { alloc::alloc_zeroed(layout) };
        if data.is_null() {
            alloc::handle_alloc_error(layout);
        }
        let data = unsafe { NonNull::new_unchecked(data) };
        let mut result = Self {
            data,
            len,
            _owned: PhantomData,
        };

        let default: Simd<T, 64> = Simd::splat(default_value);
        for simd in result.as_simd_mut::<64>() {
            *simd = default;
        }

        result
    }

    pub fn as_simd<const LANES: usize>(&self) -> &[Simd<T, LANES>]
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const Simd<T, LANES>,
                Self::rounded_len(self.len) / std::mem::size_of::<Simd<T, LANES>>(),
            )
        }
    }

    pub fn as_simd_mut<const LANES: usize>(&mut self) -> &mut [Simd<T, LANES>]
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_ptr() as *mut Simd<T, LANES>,
                Self::rounded_len(self.len) / std::mem::size_of::<Simd<T, LANES>>(),
            )
        }
    }
}

impl SimdAlignedAllocation<u8> {
    pub fn cast_to<T>(self) -> SimdAlignedAllocation<T>
    where
        T: SimdElement,
    {
        assert_eq!(0, self.len % std::mem::size_of::<T>());
        let result = SimdAlignedAllocation {
            data: self.data,
            len: self.len / std::mem::size_of::<T>(),
            _owned: PhantomData,
        };

        std::mem::forget(self);

        result
    }
}

impl<T: SimdElement> Deref for SimdAlignedAllocation<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }
}

impl<T: SimdElement> DerefMut for SimdAlignedAllocation<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr() as *mut T, self.len) }
    }
}

impl<T: SimdElement> Drop for SimdAlignedAllocation<T> {
    fn drop(&mut self) {
        let layout = Self::layout(self.len);
        unsafe { alloc::dealloc(self.data.as_ptr(), layout) }
    }
}

impl<T: SimdElement> Clone for SimdAlignedAllocation<T> {
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
                _owned: PhantomData,
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocate_simd_aligned() {
        let data: SimdAlignedAllocation<u8> = SimdAlignedAllocation::alloc_zeroed(12345);
        assert_eq!(data.as_ptr() as usize % 256, 0);
    }
}
