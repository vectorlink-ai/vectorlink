use std::{
    fs::File,
    io::{self, Read},
    ops::Index,
    os::unix::fs::MetadataExt,
    path::Path,
};

#[derive(Debug, Clone, Copy)]
pub enum Vector<'a> {
    Slice(&'a [u8]),
    Id(u32),
}

#[derive(Debug, Clone)]
pub struct Vectors {
    data: Vec<u8>,
    vector_byte_size: usize,
}

impl Vectors {
    pub fn new(data: Vec<u8>, vector_byte_size: usize) -> Self {
        assert_eq!(0, data.len() % vector_byte_size);
        Self {
            data,
            vector_byte_size,
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P, vector_byte_size: usize) -> io::Result<Self> {
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
        let mut data = Vec::with_capacity(len);
        let slice: &mut [u8] = unsafe { std::mem::transmute(data.spare_capacity_mut()) };
        reader.read_exact(slice)?;
        unsafe {
            data.set_len(len);
        }

        Ok(Self {
            data: data,
            vector_byte_size,
        })
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

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u8]> + '_ {
        (0..self.num_vecs()).map(|i| &self[i])
    }
}

impl Index<usize> for Vectors {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        let offset = self.vector_byte_size * index;
        &self.data[offset..offset + self.vector_byte_size]
    }
}
