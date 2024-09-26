use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::vectors::Vectors;

#[derive(Serialize, Deserialize)]
pub struct VectorsMetadata {
    vector_byte_size: usize,
}

impl Vectors {
    fn vec_path(directory: &Path, identity: &str) -> PathBuf {
        directory.join(format!("{identity}.vecs"))
    }
    fn meta_path(directory: &Path, identity: &str) -> PathBuf {
        directory.join(format!("{identity}.metadata.json"))
    }
    fn metadata(&self) -> VectorsMetadata {
        VectorsMetadata {
            vector_byte_size: self.vector_byte_size(),
        }
    }
    pub fn store<P: AsRef<Path>>(&self, directory: P, identity: &str) -> io::Result<()> {
        let directory = directory.as_ref();
        let mut vec_file = File::create(Self::vec_path(directory, identity))?;
        vec_file.write_all(self.data())?;
        let metadata_file = File::create(Self::meta_path(directory, identity))?;
        serde_json::to_writer(metadata_file, &self.metadata())?;

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(&self, directory: P, identity: &str) -> io::Result<Self> {
        let directory = directory.as_ref();
        let metadata_path = Self::meta_path(directory, identity);
        let meta: VectorsMetadata = serde_json::from_reader(File::open(metadata_path)?)?;
        let data = fs::read(Self::vec_path(directory, identity))?;
        Ok(Self::new(data, meta.vector_byte_size))
    }
}
