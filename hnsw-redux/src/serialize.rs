use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, MetadataExt},
    },
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::DataType, error::DataFusionError, execution::SendableRecordBatchStream,
};
use futures::TryStreamExt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    hnsw::Hnsw,
    index::{Hnsw1024, Hnsw1536, IndexConfiguration},
    layer::Layer,
    util::SimdAlignedAllocation,
    vectors::Vectors,
};

#[derive(Serialize, Deserialize)]
pub struct VectorsMetadata {
    pub vector_byte_size: usize,
}

#[async_trait]
pub trait VectorsLoader {
    fn vector_byte_size(&self) -> usize;
    fn number_of_vectors(&self) -> usize;
    async fn load(&self) -> SendableRecordBatchStream;
}

impl Vectors {
    pub async fn from_loader(loader: &dyn VectorsLoader) -> Result<Self, DataFusionError> {
        let vector_byte_size = loader.vector_byte_size();
        let meta = VectorsMetadata { vector_byte_size };
        let total_file_size = vector_byte_size * loader.number_of_vectors();
        let mut data = unsafe { SimdAlignedAllocation::<u8>::alloc(total_file_size) };

        let mut stream = loader.load().await;
        let mut ingested = 0;
        while let Some(batch) = stream.try_next().await? {
            let num_rows = batch.num_rows();
            let batch_size = num_rows * vector_byte_size;
            let start_offset = ingested * vector_byte_size;
            let end_offset = start_offset + num_rows * vector_byte_size;
            let slice = &mut data[start_offset..end_offset];
            let col = batch.column(0).to_data();
            let data: &[u8] = col.buffer(batch_size);
            slice.copy_from_slice(data);
            ingested += num_rows;
        }
        Ok(Self::new(data, meta.vector_byte_size))
    }
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

    pub fn load<P: AsRef<Path>>(directory: P, identity: &str) -> io::Result<Self> {
        eprintln!("loading from {:?} as {identity}", directory.as_ref());
        let directory = directory.as_ref();
        let metadata_path = Self::meta_path(directory, identity);

        let meta: VectorsMetadata = serde_json::from_reader(File::open(metadata_path)?)?;

        // we don't want this read to be buffered, because vector
        // files are huge, and buffering is expensive.
        let vector_file = OpenOptions::new()
            .read(true)
            .open(Self::vec_path(directory, identity))?;
        let raw_fd = vector_file.as_raw_fd();
        unsafe {
            assert_eq!(
                libc::posix_fadvise(raw_fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL),
                0,
                "fadvice (sequential) failed"
            );
            assert_eq!(
                libc::posix_fadvise(raw_fd, 0, 0, libc::POSIX_FADV_DONTNEED),
                0,
                "fadvice (dontneed) failed"
            );
        }
        let mut data =
            unsafe { SimdAlignedAllocation::alloc(vector_file.metadata()?.size() as usize) };
        eprintln!("allocated");
        const PART_SIZE: usize = 1 << 30;
        data.par_chunks_mut(PART_SIZE)
            .enumerate()
            .for_each(|(ix, part)| {
                // TODO handle error properly
                vector_file
                    .read_exact_at(part, (ix * PART_SIZE) as u64)
                    .unwrap()
            });
        eprintln!("vector data loaded...");
        Ok(Self::new(data, meta.vector_byte_size))
    }
}

#[derive(Serialize, Deserialize)]
pub struct LayerMetadata {
    single_neighborhood_size: usize,
}

#[async_trait]
pub trait LayerLoader {
    fn number_of_neighborhoods(&self) -> usize;
    async fn load(&self) -> SendableRecordBatchStream;
}

impl Layer {
    pub async fn from_loader(loader: &dyn LayerLoader) -> Result<Self, DataFusionError> {
        let number_of_neighborhoods = loader.number_of_neighborhoods();
        let mut stream = loader.load().await;
        let schema = stream.schema();
        let (col_index, field) = schema.column_with_name("neighbors").unwrap();
        if let DataType::FixedSizeList(inner_field, single_neighborhood_size) = field.data_type() {
            let single_neighborhood_size = *single_neighborhood_size as usize;
            let mut neighborhoods = unsafe {
                SimdAlignedAllocation::<u32>::alloc(
                    number_of_neighborhoods * single_neighborhood_size,
                )
            };
            assert_eq!(&DataType::Int32, inner_field.data_type());
            let mut offset = 0;
            while let Some(batch) = stream.try_next().await? {
                let batch_size = batch.num_rows();
                let start_offset = offset * single_neighborhood_size;
                let end_offset = start_offset + batch_size * single_neighborhood_size;
                let slice = &mut neighborhoods[start_offset..end_offset];
                let col = batch.column(col_index).to_data();
                let data: &[u32] = col.buffer(batch_size);
                slice.copy_from_slice(data);
                offset += batch_size;
            }

            Ok(Self::new(neighborhoods, single_neighborhood_size))
        } else {
            panic!("argh");
        }
    }
}

impl Layer {
    fn metadata(&self) -> LayerMetadata {
        LayerMetadata {
            single_neighborhood_size: self.single_neighborhood_size(),
        }
    }

    fn neighbors_path(directory: &Path, layer_index: usize) -> PathBuf {
        directory.join(format!("layer.{layer_index}.neighbors"))
    }
    fn meta_path(directory: &Path, layer_index: usize) -> PathBuf {
        directory.join(format!("layer.{layer_index}.metadata.json"))
    }
    pub fn store<P: AsRef<Path>>(&self, directory: P, layer_index: usize) -> io::Result<()> {
        let directory = directory.as_ref();
        let data = self.data();
        fs::write(Self::neighbors_path(directory, layer_index), data)?;

        let metadata_file = File::create(Self::meta_path(directory, layer_index))?;
        serde_json::to_writer(metadata_file, &self.metadata())?;

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(directory: P, layer_index: usize) -> io::Result<Layer> {
        let directory = directory.as_ref();
        let metadata: LayerMetadata =
            serde_json::from_reader(File::open(Self::meta_path(directory, layer_index))?)?;
        let mut data_file = File::open(Self::neighbors_path(directory, layer_index))?;
        let data_size = data_file.metadata()?.size();
        let mut data: SimdAlignedAllocation<u8> =
            unsafe { SimdAlignedAllocation::alloc(data_size as usize) };

        data_file.read_exact(&mut data[..])?;

        Ok(Layer::from_data(
            data.cast_to(),
            metadata.single_neighborhood_size,
        ))
    }
}

#[derive(Serialize, Deserialize)]
pub struct HnswMetadata {
    layer_count: usize,
}

#[async_trait]
pub trait HnswLoader {
    fn layer_count(&self) -> usize;
    async fn get_layer_loader(&self, index: usize)
        -> Result<Box<dyn LayerLoader>, DataFusionError>;
}

impl Hnsw {
    pub async fn from_loader(loader: &dyn HnswLoader) -> Result<Self, DataFusionError> {
        let layer_count = loader.layer_count();
        let mut layers = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            let layer_loader = loader.get_layer_loader(i).await?;
            let layer = Layer::from_loader(&*layer_loader).await?;
            layers.push(layer);
        }

        Ok(Self::new(layers))
    }
}

impl Hnsw {
    fn metadata(&self) -> HnswMetadata {
        HnswMetadata {
            layer_count: self.layer_count(),
        }
    }
    fn meta_path(directory: &Path) -> PathBuf {
        directory.join("hnsw.json")
    }

    pub fn store<P: AsRef<Path>>(&self, directory: P) -> io::Result<()> {
        eprintln!("storing inner hnsw");
        let directory = directory.as_ref();
        for (i, layer) in self.layers().iter().enumerate() {
            layer.store(directory, i)?;
        }

        serde_json::to_writer(File::create(Self::meta_path(directory))?, &self.metadata())?;

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(directory: P) -> io::Result<Self> {
        let directory = directory.as_ref();
        let metadata: HnswMetadata =
            serde_json::from_reader(File::open(Self::meta_path(directory))?)?;
        let mut layers = Vec::with_capacity(metadata.layer_count);
        for i in 0..metadata.layer_count {
            layers.push(Layer::load(directory, i)?);
        }

        Ok(Hnsw::new(layers))
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
pub enum HnswType {
    Hnsw1536,
    Hnsw1024,
    Quantized128x8,
}

#[derive(Serialize, Deserialize)]
pub struct HnswConfigurationMetadata {
    name: String,
    hnsw_type: HnswType,
}

impl Hnsw1024 {
    fn metadata(&self) -> HnswConfigurationMetadata {
        HnswConfigurationMetadata {
            name: self.name().to_owned(),
            hnsw_type: HnswType::Hnsw1024,
        }
    }

    pub fn load<P1: AsRef<Path>, P2: AsRef<Path>>(
        name: &str,
        hnsw_root_directory: P1,
        vector_directory: P2,
    ) -> io::Result<Self> {
        let hnsw_root_directory = hnsw_root_directory.as_ref();
        let hnsw_path = IndexConfiguration::hnsw_path(name, hnsw_root_directory);
        let meta_path = IndexConfiguration::meta_path(name, hnsw_root_directory);
        let metadata: HnswConfigurationMetadata = serde_json::from_reader(File::open(meta_path)?)?;
        assert_eq!(metadata.hnsw_type, HnswType::Hnsw1024);

        let hnsw = Hnsw::load(hnsw_path)?;
        let vector_name = &metadata.name;
        let vectors = Vectors::load(vector_directory, vector_name)?;

        Ok(Self::new(name.to_string(), hnsw, vectors))
    }

    pub fn store_hnsw<P: AsRef<Path>>(&self, hnsw_directory: P) -> io::Result<()> {
        eprintln!("storing hnsw");
        let hnsw_root_directory = hnsw_directory.as_ref();
        let metadata = self.metadata();
        let hnsw_path = IndexConfiguration::hnsw_path(&metadata.name, hnsw_root_directory);
        let metadata_path = IndexConfiguration::meta_path(&metadata.name, hnsw_root_directory);
        fs::create_dir_all(&hnsw_path)?;
        serde_json::to_writer(File::create(metadata_path)?, &metadata)?;
        self.hnsw().store(hnsw_path)?;

        Ok(())
    }
}

impl Hnsw1536 {
    fn metadata(&self) -> HnswConfigurationMetadata {
        HnswConfigurationMetadata {
            name: self.name().to_owned(),
            hnsw_type: HnswType::Hnsw1536,
        }
    }

    pub fn load<P1: AsRef<Path>, P2: AsRef<Path>>(
        name: &str,
        hnsw_root_directory: P1,
        vector_directory: P2,
    ) -> io::Result<Self> {
        let hnsw_root_directory = hnsw_root_directory.as_ref();
        let hnsw_path = IndexConfiguration::hnsw_path(name, hnsw_root_directory);
        let meta_path = IndexConfiguration::meta_path(name, hnsw_root_directory);
        let metadata: HnswConfigurationMetadata = serde_json::from_reader(File::open(meta_path)?)?;
        assert_eq!(metadata.hnsw_type, HnswType::Hnsw1536);

        let hnsw = Hnsw::load(hnsw_path)?;
        let vector_name = &metadata.name;
        let vectors = Vectors::load(vector_directory, vector_name)?;

        Ok(Self::new(name.to_string(), hnsw, vectors))
    }

    pub fn store_hnsw<P: AsRef<Path>>(&self, hnsw_directory: P) -> io::Result<()> {
        eprintln!("storing hnsw");
        let hnsw_root_directory = hnsw_directory.as_ref();
        let metadata = self.metadata();
        let hnsw_path = IndexConfiguration::hnsw_path(&metadata.name, hnsw_root_directory);
        let metadata_path = IndexConfiguration::meta_path(&metadata.name, hnsw_root_directory);
        fs::create_dir_all(&hnsw_path)?;
        serde_json::to_writer(File::create(metadata_path)?, &metadata)?;
        self.hnsw().store(hnsw_path)?;

        Ok(())
    }
}

impl IndexConfiguration {
    fn hnsw_path(name: &str, hnsw_root_directory: &Path) -> PathBuf {
        hnsw_root_directory.join(name)
    }

    fn meta_path(name: &str, hnsw_root_directory: &Path) -> PathBuf {
        Self::hnsw_path(name, hnsw_root_directory).join("configuration.json")
    }

    pub fn load<P1: AsRef<Path>, P2: AsRef<Path>>(
        name: &str,
        hnsw_root_directory: P1,
        vector_directory: P2,
    ) -> io::Result<Self> {
        let hnsw_root_directory = hnsw_root_directory.as_ref();
        let meta_path = IndexConfiguration::meta_path(name, hnsw_root_directory);
        let metadata: HnswConfigurationMetadata = serde_json::from_reader(File::open(meta_path)?)?;

        match metadata.hnsw_type {
            HnswType::Hnsw1024 => {
                Ok(Hnsw1024::load(name, hnsw_root_directory, vector_directory)?.into())
            }
            HnswType::Hnsw1536 => {
                Ok(Hnsw1536::load(name, hnsw_root_directory, vector_directory)?.into())
            }
            HnswType::Quantized128x8 => todo!(),
        }
    }
    pub fn store_hnsw<P: AsRef<Path>>(&self, hnsw_directory: P) -> io::Result<()> {
        eprintln!("storing index configuration");
        match self {
            IndexConfiguration::Hnsw1536(hnsw) => hnsw.store_hnsw(hnsw_directory),
            IndexConfiguration::Hnsw1024(hnsw) => hnsw.store_hnsw(hnsw_directory),
            IndexConfiguration::Pq1024x8(_pq) => todo!(),
        }
    }
}
