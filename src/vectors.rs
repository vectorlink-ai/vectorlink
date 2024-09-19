use std::ops::Index;

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

    pub fn vector_byte_size(&self) -> usize {
        self.vector_byte_size
    }
}

impl Index<usize> for Vectors {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        let offset = self.vector_byte_size * index;
        &self.data[offset..offset + self.vector_byte_size]
    }
}
