use std::ops::Index;

#[derive(Debug)]
pub enum Vector<'a> {
    Slice(&'a [u8]),
    Id(u32),
}

#[derive(Debug)]
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

    pub fn len(&self) -> usize {
        self.data.len() / self.vector_byte_size
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn vector_byte_size(&self) -> usize {
        self.vector_byte_size
    }

    pub fn get<T>(&self, index: usize) -> &T {
        if index == u32::MAX as usize {
            self.get(0)
        } else {
            debug_assert_eq!(std::mem::size_of::<T>(), self.vector_byte_size);
            let offset = self.vector_byte_size * index;
            debug_assert!(offset + self.vector_byte_size <= self.data.len());
            unsafe { &*(self.data.as_ptr().add(offset) as *const T) }
        }
    }
}

impl Index<usize> for Vectors {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        let offset = self.vector_byte_size * index;
        &self.data[offset..offset + self.vector_byte_size]
    }
}
