use std::alloc;

pub fn aligned_256_vec<T>(capacity: usize) -> Vec<T> {
    let layout = alloc::Layout::from_size_align(capacity * std::mem::size_of::<T>(), 256).unwrap();
    unsafe {
        let ptr = alloc::alloc(layout);
        debug_assert_eq!(ptr as usize % 256, 0);
        Vec::from_raw_parts(ptr as *mut T, 0, capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocate_aligned_vec_buf() {
        let vec: Vec<[f32; 1024]> = aligned_256_vec(100);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 100);
        assert_eq!(vec.as_ptr() as usize % 256, 0);
    }
}
