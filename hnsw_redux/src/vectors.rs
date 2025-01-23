use std::{
    fs::File,
    io::{self, Read},
    ops::{Index, IndexMut},
    os::unix::fs::MetadataExt,
    path::Path,
};

use rayon::prelude::*;

use crate::{
    util::SimdAlignedAllocation,
    vecmath::{normalize_aligned_1024, normalize_aligned_1536},
};

#[derive(Debug, Clone, Copy)]
pub enum Vector<'a> {
    Slice(&'a [u8]),
    Id(u32),
}

#[derive(Clone)]
pub struct Vectors {
    data: SimdAlignedAllocation<u8>,
    vector_byte_size: usize,
}

impl Vectors {
    pub fn empty(vector_byte_size: usize) -> Self {
        let data: SimdAlignedAllocation<u8> = unsafe { SimdAlignedAllocation::alloc(0) };
        Self {
            data,
            vector_byte_size,
        }
    }

    pub fn new(data: SimdAlignedAllocation<u8>, vector_byte_size: usize) -> Self {
        assert_eq!(0, data.len() % vector_byte_size);
        Self {
            data,
            vector_byte_size,
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P, vector_byte_size: usize) -> io::Result<Self> {
        eprintln!("Loading vectors from file {:?}", path.as_ref());
        let path = path.as_ref();
        let file = File::open(path)?;
        let file_size = file.metadata()?.size() as usize;
        assert_eq!(
            file_size % vector_byte_size,
            0,
            "vector file does not contain an exact amount of vectors"
        );
        let num_vecs = file_size / vector_byte_size;

        Self::from_reader(file, num_vecs, vector_byte_size)
    }

    pub fn from_reader<R: Read>(
        mut reader: R,
        num_vecs: usize,
        vector_byte_size: usize,
    ) -> io::Result<Self> {
        let len = num_vecs * vector_byte_size;
        let mut data = unsafe { SimdAlignedAllocation::alloc(len) };
        reader.read_exact(&mut data[..])?;
        Ok(Self::new(data, vector_byte_size))
    }

    pub fn num_vecs(&self) -> usize {
        self.data.len() / self.vector_byte_size
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn vector_byte_size(&self) -> usize {
        self.vector_byte_size
    }

    pub fn get<T>(&self, index: usize) -> Option<&T> {
        if index == u32::MAX as usize {
            None
        } else {
            debug_assert_eq!(std::mem::size_of::<T>(), self.vector_byte_size);
            let offset = self.vector_byte_size * index;
            debug_assert!(offset + self.vector_byte_size <= self.data.len());
            unsafe { Some(&*(self.data.as_ptr().add(offset) as *const T)) }
        }
    }

    pub fn get_mut<T>(&mut self, index: usize) -> &mut T {
        if index == u32::MAX as usize {
            panic!("wrong index {index}");
        } else {
            debug_assert_eq!(std::mem::size_of::<T>(), self.vector_byte_size);
            let offset = self.vector_byte_size * index;
            debug_assert!(offset + self.vector_byte_size <= self.data.len());
            unsafe { &mut *(self.data.as_ptr().add(offset) as *mut T) }
        }
    }

    pub fn get_mut_f32_slice(&mut self, index: usize) -> &mut [f32] {
        if index == u32::MAX as usize {
            panic!("wrong index {index}");
        } else {
            let offset = self.vector_byte_size * index;
            debug_assert!(offset + self.vector_byte_size <= self.data.len());
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.data.as_ptr().add(offset) as *mut f32,
                    self.vector_byte_size / std::mem::size_of::<f32>(),
                )
            }
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u8]> + '_ {
        (0..self.num_vecs()).map(|i| &self[i])
    }

    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &[u8]> + '_ {
        (0..self.num_vecs()).into_par_iter().map(|i| &self[i])
    }

    pub fn normalize(&mut self) {
        assert!(
            self.vector_byte_size >= 256 && self.vector_byte_size % 256 == 0,
            "cannot normalize vectors that aren't 256bit aligned (for avx)"
        );
        self.data
            .par_chunks_mut(self.vector_byte_size)
            .for_each(|vector| {
                let vector = unsafe {
                    std::slice::from_raw_parts_mut(
                        vector.as_mut_ptr() as *mut f32,
                        self.vector_byte_size / std::mem::size_of::<f32>(),
                    )
                };
                // TODO it's not just about vector byte size. we really should
                // keep around some metadata about what sort of vectors we
                // have.
                // TODO this could be faster if there wasn't this dispatch
                match self.vector_byte_size {
                    4096 => normalize_aligned_1024(vector),
                    6114 => normalize_aligned_1536(vector),
                    n => panic!("don't know how to normalize vectors of size {n}"),
                }
            });
    }
}

impl Index<usize> for Vectors {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        let offset = self.vector_byte_size * index;
        &self.data[offset..offset + self.vector_byte_size]
    }
}

impl IndexMut<usize> for Vectors {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let offset = self.vector_byte_size * index;
        &mut self.data[offset..offset + self.vector_byte_size]
    }
}
