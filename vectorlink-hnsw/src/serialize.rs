use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{FileExt, MetadataExt},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::ffi_stream::ArrowArrayStreamReader;
use datafusion::arrow::{
    array::{
        Array, FixedSizeListArray, RecordBatch, RecordBatchReader, UInt32Array,
    },
    buffer::{Buffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema},
    error::ArrowError,
};
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

    pub fn store<P: AsRef<Path>>(
        &self,
        directory: P,
        identity: &str,
    ) -> io::Result<()> {
        let directory = directory.as_ref();
        let mut vec_file = File::create(Self::vec_path(directory, identity))?;
        vec_file.write_all(self.data())?;
        let metadata_file = File::create(Self::meta_path(directory, identity))?;
        serde_json::to_writer(metadata_file, &self.metadata())?;
        Ok(())
    }

    pub fn from_arrow(
        reader: ArrowArrayStreamReader,
        number_of_records: usize,
    ) -> Result<Self, SerializeError> {
        let schema = reader.schema();
        let col_type = schema.field(0).data_type();
        let mut offset = 0;
        if let DataType::FixedSizeList(_, single_vector_size) = col_type {
            let single_vector_size: usize =
                (*single_vector_size).try_into().unwrap();
            let mut destination = unsafe {
                SimdAlignedAllocation::<u8>::alloc(
                    number_of_records
                        * single_vector_size
                        * std::mem::size_of::<f32>(),
                )
            };
            for batch in reader {
                let batch = batch?;
                let count = batch.num_rows();
                let float_count = count * single_vector_size;
                let byte_count = float_count * std::mem::size_of::<f32>();
                let col = batch.column(0);
                let data = col.to_data();
                let child_data = data.child_data();

                let data_slice: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        child_data[0].buffers().last().unwrap().as_ptr(),
                        byte_count,
                    )
                };
                let destination_slice =
                    &mut destination[offset..offset + byte_count];
                destination_slice.copy_from_slice(data_slice);

                offset += float_count;
            }

            return Ok(Vectors::new(
                destination,
                single_vector_size * std::mem::size_of::<f32>(),
            ));
        } else {
            Err(SerializeError::ExpectedFixedSizeList)
        }
    }

    pub fn load<P: AsRef<Path>>(
        directory: P,
        identity: &str,
    ) -> io::Result<Self> {
        eprintln!("loading from {:?} as {identity}", directory.as_ref());
        let directory = directory.as_ref();
        let metadata_path = Self::meta_path(directory, identity);

        let meta: VectorsMetadata =
            serde_json::from_reader(File::open(metadata_path)?)?;

        // we don't want this read to be buffered, because vector
        // files are huge, and buffering is expensive.
        let vector_file = OpenOptions::new()
            .read(true)
            .open(Self::vec_path(directory, identity))?;
        #[cfg(target_os = "linux")]
        unsafe {
            use std::os::fd::AsRawFd;
            let raw_fd = vector_file.as_raw_fd();
            // The `libc::posix_fadvise()` fn doesn't exist on e.g. MacOS.
            // Therefore, only use this optimization when it's available.
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
        let mut data = unsafe {
            SimdAlignedAllocation::alloc(vector_file.metadata()?.size() as usize)
        };
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

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerMetadata {
    single_neighborhood_size: usize,
}

impl LayerMetadata {
    pub fn single_neighborhood_size(&self) -> usize {
        self.single_neighborhood_size
    }
}

impl Layer {
    pub fn from_arrow(
        reader: ArrowArrayStreamReader,
        number_of_records: usize,
    ) -> Result<Self, SerializeError> {
        let schema = reader.schema();
        let col_type = schema.field(0).data_type();
        let mut offset = 0;
        if let DataType::FixedSizeList(_, single_neighborhood_size) = col_type {
            let single_neighborhood_size: usize =
                (*single_neighborhood_size).try_into().unwrap();
            let mut neighborhoods = unsafe {
                SimdAlignedAllocation::<u32>::alloc(
                    number_of_records * single_neighborhood_size,
                )
            };

            for batch in reader {
                let batch = batch?;
                let count = batch.num_rows();
                let u32_count = count * single_neighborhood_size;
                let col = batch.column(0);
                let data = col.to_data();
                let child_data = data.child_data();

                let data_slice: &[u32] = unsafe {
                    let last_buffer =
                        child_data[0].buffers().last().unwrap().as_ptr()
                            as *const u32;
                    std::slice::from_raw_parts(last_buffer, u32_count)
                };
                let destination_slice =
                    &mut neighborhoods[offset..offset + u32_count];
                destination_slice.copy_from_slice(data_slice);

                offset += u32_count;
            }

            Ok(Layer::new(neighborhoods, single_neighborhood_size))
        } else {
            Err(SerializeError::ExpectedFixedSizeList)
        }
    }

    pub fn arrow_schema(&self) -> Arc<Schema> {
        // TODO this should cache
        let neighbor_field =
            Arc::new(Field::new("item".to_string(), DataType::UInt32, false));

        let neighborhood_field = Arc::new(Field::new(
            "neighborhood".to_string(),
            DataType::FixedSizeList(
                neighbor_field.clone(),
                self.single_neighborhood_size() as i32,
            ),
            false,
        ));
        let index_field =
            Arc::new(Field::new("index", DataType::UInt32, false));

        Arc::new(Schema::new([index_field, neighborhood_field]))
    }

    pub fn neighborhood_batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Result<RecordBatch, ArrowError>> + '_ {
        (0..self.number_of_neighborhoods())
            .step_by(batch_size)
            .map(move |i| Ok(self.neighborhood_batch(i, batch_size)))
    }

    pub fn neighborhood_batch(
        &self,
        index: usize,
        batch_size: usize,
    ) -> RecordBatch {
        let schema = self.arrow_schema();
        let start_offset = index * self.single_neighborhood_size();
        let end_offset = std::cmp::min(
            start_offset + batch_size * self.single_neighborhood_size(),
            self.neighborhoods().len(),
        );
        let len = end_offset - start_offset;
        let neighborhood_buf = Buffer::from_slice_ref(
            &self.neighborhoods()[start_offset..end_offset],
        );
        let scalar_buf = ScalarBuffer::new(neighborhood_buf, 0, len);
        let array = Arc::new(UInt32Array::new(scalar_buf, None));
        let neighbor_field = match schema
            .field_with_name("index")
            .expect("no index field in schema")
            .data_type()
        {
            DataType::FixedSizeList(field, _) => field.clone(),
            _ => panic!("field not of expected type"),
        };
        let neighborhoods_array: Arc<dyn Array> =
            Arc::new(FixedSizeListArray::new(
                neighbor_field,
                self.single_neighborhood_size() as i32,
                array,
                None,
            ));
        let indexes_array: Arc<dyn Array> = Arc::new(
            UInt32Array::from_iter_values(index as u32..(index + len) as u32),
        );

        RecordBatch::try_new(schema, vec![indexes_array, neighborhoods_array])
            .expect("failed to produce record batch")
    }

    pub fn metadata(&self) -> LayerMetadata {
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

    pub fn store<P: AsRef<Path>>(
        &self,
        directory: P,
        layer_index: usize,
    ) -> io::Result<()> {
        let directory = directory.as_ref();
        let data = self.raw_data();
        fs::write(Self::neighbors_path(directory, layer_index), data)?;

        let metadata_file =
            File::create(Self::meta_path(directory, layer_index))?;
        serde_json::to_writer(metadata_file, &self.metadata())?;

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(
        directory: P,
        layer_index: usize,
    ) -> io::Result<Layer> {
        let directory = directory.as_ref();
        let metadata: LayerMetadata = serde_json::from_reader(File::open(
            Self::meta_path(directory, layer_index),
        )?)?;
        let mut data_file =
            File::open(Self::neighbors_path(directory, layer_index))?;
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

#[derive(Debug, Serialize, Deserialize)]
pub struct HnswMetadata {
    layer_count: usize,
}

impl HnswMetadata {
    pub fn layer_count(&self) -> usize {
        self.layer_count
    }
}

impl Hnsw {
    pub fn metadata(&self) -> HnswMetadata {
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
        serde_json::to_writer(
            File::create(Self::meta_path(directory))?,
            &self.metadata(),
        )?;
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
        let hnsw_path =
            IndexConfiguration::hnsw_path(name, hnsw_root_directory);
        let meta_path =
            IndexConfiguration::meta_path(name, hnsw_root_directory);
        let metadata: HnswConfigurationMetadata =
            serde_json::from_reader(File::open(meta_path)?)?;
        assert_eq!(metadata.hnsw_type, HnswType::Hnsw1024);

        let hnsw = Hnsw::load(hnsw_path)?;
        let vector_name = &metadata.name;
        let vectors = Vectors::load(vector_directory, vector_name)?;

        Ok(Self::new(name.to_string(), hnsw, vectors))
    }

    pub fn store_hnsw<P: AsRef<Path>>(
        &self,
        hnsw_directory: P,
    ) -> io::Result<()> {
        eprintln!("storing hnsw");
        let hnsw_root_directory = hnsw_directory.as_ref();
        let metadata = self.metadata();
        let hnsw_path =
            IndexConfiguration::hnsw_path(&metadata.name, hnsw_root_directory);
        let metadata_path =
            IndexConfiguration::meta_path(&metadata.name, hnsw_root_directory);
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
        let hnsw_path =
            IndexConfiguration::hnsw_path(name, hnsw_root_directory);
        let meta_path =
            IndexConfiguration::meta_path(name, hnsw_root_directory);
        let metadata: HnswConfigurationMetadata =
            serde_json::from_reader(File::open(meta_path)?)?;
        assert_eq!(metadata.hnsw_type, HnswType::Hnsw1536);

        let hnsw = Hnsw::load(hnsw_path)?;
        let vector_name = &metadata.name;
        let vectors = Vectors::load(vector_directory, vector_name)?;

        Ok(Self::new(name.to_string(), hnsw, vectors))
    }

    pub fn store_hnsw<P: AsRef<Path>>(
        &self,
        hnsw_directory: P,
    ) -> io::Result<()> {
        eprintln!("storing hnsw");
        let hnsw_root_directory = hnsw_directory.as_ref();
        let metadata = self.metadata();
        let hnsw_path =
            IndexConfiguration::hnsw_path(&metadata.name, hnsw_root_directory);
        let metadata_path =
            IndexConfiguration::meta_path(&metadata.name, hnsw_root_directory);
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
        let meta_path =
            IndexConfiguration::meta_path(name, hnsw_root_directory);
        let metadata: HnswConfigurationMetadata =
            serde_json::from_reader(File::open(meta_path)?)?;

        match metadata.hnsw_type {
            HnswType::Hnsw1024 => {
                Ok(Hnsw1024::load(name, hnsw_root_directory, vector_directory)?
                    .into())
            }
            HnswType::Hnsw1536 => {
                Ok(Hnsw1536::load(name, hnsw_root_directory, vector_directory)?
                    .into())
            }
            HnswType::Quantized128x8 => todo!(),
        }
    }
    pub fn store_hnsw<P: AsRef<Path>>(
        &self,
        hnsw_directory: P,
    ) -> io::Result<()> {
        eprintln!("storing index configuration");
        match self {
            IndexConfiguration::Hnsw1536(hnsw) => {
                hnsw.store_hnsw(hnsw_directory)
            }
            IndexConfiguration::Hnsw1024(hnsw) => {
                hnsw.store_hnsw(hnsw_directory)
            }
            IndexConfiguration::Pq1024x8(_pq) => todo!(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SerializeError {
    #[error("Arrow error: {0}")]
    ArrowError(#[from] ArrowError),

    #[error(" Expected a FixedSizeList in Arrow stream")]
    ExpectedFixedSizeList,
}
