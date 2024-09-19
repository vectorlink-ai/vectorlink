use std::ops::Index;

pub struct Vectors<T = f32> {
    data: Vec<T>,
    vector_size: usize,
}

impl<T> Vectors<T> {
    pub fn new(data: Vec<T>, vector_size: usize) -> Self {
        assert_eq!(0, data.len() % vector_size);
        Self { data, vector_size }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.vector_size
    }

    pub fn vector_size(&self) -> usize {
        self.vector_size
    }
}

impl<T> Index<usize> for Vectors<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let offset = self.vector_size * index;
        &self.data[offset..offset + self.vector_size]
    }
}

pub type QuantizedVectors = Vectors<u16>;
