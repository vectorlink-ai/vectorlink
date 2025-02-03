#![allow(unused)] // TODO: remove

use ::datafusion::{
    arrow::datatypes::Schema, error::DataFusionError,
    execution::SendableRecordBatchStream,
};
use ::hnsw_redux::{
    hnsw, index, layer, serialize,
    util::{self, SimdAlignedAllocation},
    vectors,
};
use ::pyo3::{
    exceptions::{PyException, PyIndexError, PyTypeError},
    prelude::*,
    pyclass::PyClass,
    types::{IntoPyDict, PyCapsule, PyDict},
};
use async_trait::async_trait;
use datafusion::{
    arrow::{
        alloc,
        array::{RecordBatch, RecordBatchReader},
        datatypes::DataType,
        ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream},
    },
    execution::RecordBatchStream,
    physical_plan::stream::RecordBatchStreamAdapter,
};
use pyo3::types::PyType;
use std::{
    pin::Pin,
    sync::{Arc, Mutex},
};
use tokio_stream::StreamExt;

// This function defines a Python module. Its name MUST match the the `lib.name`
// settings in `Cargo.toml`, else Python will not be able to import the module.
#[pymodule]
fn hnsw_redux(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vectors>()?;
    m.add_class::<SimdAlignedAllocationU8>()?;
    m.add_class::<LayerMetadata>()?;
    m.add_class::<Layer>()?;
    m.add_class::<HnswMetadata>()?;
    m.add_class::<Hnsw>()?;
    m.add_class::<HnswType>()?;
    m.add_class::<Hnsw1024>()?;
    m.add_class::<Hnsw1536>()?;
    m.add_class::<IndexConfiguration>()?;
    m.add_class::<VectorsLoader>()?;
    m.add_class::<LayerLoader>()?;
    m.add_class::<HnswLoader>()?;
    Ok(())
}

#[pyclass(module = "hnsw_redux")]
pub struct Vectors(vectors::Vectors);

#[pymethods]
impl Vectors {
    #[new]
    pub fn new(data: SimdAlignedAllocationU8, vector_byte_size: usize) -> Self {
        Self(vectors::Vectors::new(data.0, vector_byte_size))
    }

    #[staticmethod]
    pub fn empty(vector_byte_size: usize) -> Self {
        Self(vectors::Vectors::empty(vector_byte_size))
    }

    #[staticmethod]
    pub fn from_file(
        filepath: &str,
        vector_byte_size: usize,
    ) -> PyResult<Self> {
        Ok(Self(vectors::Vectors::from_file(
            filepath,
            vector_byte_size,
        )?))
    }

    #[staticmethod]
    pub fn from_arrow(
        arrow_stream: Bound<'_, PyCapsule>,
        number_of_records: usize,
    ) -> PyResult<Self> {
        let arrow_stream = arrow_stream.as_ptr() as *mut FFI_ArrowArrayStream;
        // TODO we need an error type
        let reader =
            unsafe { ArrowArrayStreamReader::from_raw(arrow_stream) }.unwrap();
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
                let batch = batch.unwrap();
                let count = batch.num_rows();
                let float_count = count * single_vector_size;
                let col = batch.column(0);
                // TODO this might incur a copy. that's stupid.
                let data = col.to_data();

                let data_slice: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        data.buffers().last().unwrap().as_ptr(),
                        float_count * std::mem::size_of::<f32>(),
                    )
                };
                let destination_slice =
                    &mut destination[offset..offset + float_count];
                destination_slice.copy_from_slice(data_slice);

                offset += float_count;
            }

            return Ok(Vectors::new(
                SimdAlignedAllocationU8(destination),
                single_vector_size * std::mem::size_of::<f32>(),
            ));
        }

        panic!("bad datatype");
    }

    /*
    #[staticmethod]
    pub async fn from_loader(loader: VectorsLoader) -> PyResult<Self> {
        vectors::Vectors::from_loader(
            &mut *loader.0.lock().expect("poisoned lock"),
        )
        .await
        .map(Self)
        .map_err(|datafusion_error| {
            let msg = format!("DataFusion error: {datafusion_error}");
            PyErr::new::<PyTypeError, _>(msg)
        })
    }
    */

    pub fn num_vecs(&self) -> usize {
        self.0.num_vecs()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn vector_byte_size(&self) -> usize {
        self.0.vector_byte_size()
    }

    pub fn data(&self) -> &[u8] {
        self.0.data()
    }

    pub fn normalize(&mut self) {
        self.0.normalize()
    }

    pub fn __getitem__(&self, index: usize) -> PyResult<&[u8]> {
        if index < self.num_vecs() {
            Ok(&self.0[index])
        } else {
            let msg = format!("Index {index} is out of bounds");
            Err(PyIndexError::new_err(msg))
        }
    }

    // #[staticmethod]
    // pub async fn from_loader(l: &dyn VectorsLoader) -> PyResult<Self> {
    //     // TODO: Provide a way to
    //     //       1. obtain a `serialize::VectorsLoader` instance, or
    //     //       2. load from a DataFusion DataFrame
    //     vectors::Vectors::from_loader(l).await
    //         .map(Self)
    //         .map_err(|datafusion_error| {
    //             let msg = format!("DataFusion error: {datafusion_error}");
    //             PyErr::new::<PyTypeError, _>(msg)
    //         })
    // }

    pub fn store(&self, dirpath: &str, identity: &str) -> PyResult<()> {
        self.0.store(dirpath, identity)?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(dirpath: &str, identity: &str) -> PyResult<Self> {
        Ok(Self(vectors::Vectors::load(dirpath, identity)?))
    }
}

#[pyclass(module = "hnsw_redux")]
#[derive(Clone)]
pub struct VectorsLoader(Arc<dyn serialize::VectorsLoader>);

impl VectorsLoader {
    async fn load(&self) -> SendableRecordBatchStream {
        self.0.load().await
    }
}

impl From<Arc<dyn serialize::VectorsLoader>> for VectorsLoader {
    fn from(inner: Arc<dyn serialize::VectorsLoader>) -> Self {
        Self(inner)
    }
}

unsafe impl Send for VectorsLoader {}
unsafe impl Sync for VectorsLoader {}

pub struct ArrowVectorsLoader {
    stream: Pin<Box<dyn RecordBatchStream + Send + Sync>>,
    vector_byte_size: usize,
    number_of_vectors: usize,
}

/*
#[async_trait]
impl serialize::VectorsLoader for ArrowVectorsLoader {
    fn vector_byte_size(&self) -> usize {
        self.vector_byte_size
    }

    fn number_of_vectors(&self) -> usize {
        self.number_of_vectors
    }

    async fn load(&mut self) -> SendableRecordBatchStream {
        loop {
            let next = self.stream.next().await;
            /*
            match next {
                Some(Ok(next)) => todo!(),
                Some(Err(e)) => todo!(),
                None => todo!(),
            }
            */
        }
        todo!();
    }
}

#[pymethods]
impl VectorsLoader {
    #[classmethod]
    fn from_arrow_c_stream(
        cls: &Bound<'_, PyType>,
        capsule: Bound<'_, PyCapsule>,
        vector_byte_size: usize,
        number_of_vectors: usize,
    ) -> Self {
        let stream = unsafe { capsule.pointer() as *mut FFI_ArrowArrayStream };
        eprintln!("constructed stream");
        let reader =
            unsafe { ArrowArrayStreamReader::from_raw(stream).unwrap() };
        let schema = reader.schema();

        let stream = tokio_stream::iter(reader);
        let wrapped_stream = RecordBatchStreamAdapter::new(
            schema,
            stream.map(|r| r.map_err(Into::into)),
        );

        let loader = ArrowVectorsLoader {
            stream: Box::pin(wrapped_stream),
            vector_byte_size,
            number_of_vectors,
        };

        Self(Arc::new(loader))
    }
}
*/

#[pyclass(module = "hnsw_redux")]
#[derive(Clone)]
pub struct LayerLoader(Arc<dyn serialize::LayerLoader>);

impl LayerLoader {
    async fn load(&self) -> SendableRecordBatchStream {
        self.0.load().await
    }
}

impl From<Arc<dyn serialize::LayerLoader>> for LayerLoader {
    fn from(inner: Arc<dyn serialize::LayerLoader>) -> Self {
        Self(inner)
    }
}

unsafe impl Send for LayerLoader {}
unsafe impl Sync for LayerLoader {}

#[pyclass(module = "hnsw_redux")]
#[derive(Clone)]
pub struct HnswLoader(Arc<dyn serialize::HnswLoader>);

#[pymethods]
impl HnswLoader {
    fn layer_count(&self) -> usize {
        self.0.layer_count()
    }

    async fn get_layer_loader(
        &self,
        index: usize,
    ) -> Result<LayerLoader, DataFusionError> {
        let loader = self.0.get_layer_loader(index).await?;
        Ok(LayerLoader(loader))
    }
}

impl From<Arc<dyn serialize::HnswLoader>> for HnswLoader {
    fn from(inner: Arc<dyn serialize::HnswLoader>) -> Self {
        Self(inner)
    }
}

unsafe impl Send for HnswLoader {}
unsafe impl Sync for HnswLoader {}

// PyO3 / Maturin require that no generics are used at FFI boundaries.
// This means that for types like `hnsw_redux::util::SimdAlignedAllocation<E>`,
// the SIMD element type parameter `E` needs to be filled out rather than be
// left up to the user.  This macro does that by providing that `$element` while
// also wrapping the generic type into a new `SimdAlignedAllocation*` type for
// each `$element`.
// For example, for `$element = u8`, it generates the `SimdAlignedAllocationU8`
// type, which wraps `hnsw_redux::util::SimdAlignedAllocation<u8>`.
macro_rules! wrap_SimdAlignedAllocation_type_for_element {
    ($($element:ty),*) => {
        paste::paste! {
            $(
                #[pyclass(module = "hnsw_redux")]
                #[derive(Clone)]
                pub struct [<SimdAlignedAllocation $element:upper>](
                    util::SimdAlignedAllocation<$element>
                );

                #[pymethods]
                impl [<SimdAlignedAllocation $element:upper>] {
                    #[staticmethod]
                    pub unsafe fn alloc(len: usize) -> Self {
                        // TODO: this method is still unsafe
                        Self(util::SimdAlignedAllocation::alloc(len))
                    }

                    #[staticmethod]
                    pub fn alloc_zeroed(len: usize) -> Self {
                        Self(util::SimdAlignedAllocation::alloc_zeroed(len))
                    }

                    #[staticmethod]
                    pub fn alloc_default(
                        len: usize,
                        default_value: $element
                    ) -> Self {
                        Self(util::SimdAlignedAllocation::alloc_default(
                            len,
                            default_value
                        ))
                    }
                }
            )*
        }
    }
}

// These wrapper types for `SimdAlignedAllocation<_>`
// are needed for `self::Vectors::new()`
wrap_SimdAlignedAllocation_type_for_element![u8];

#[pyclass(module = "hnsw_redux")]
pub struct LayerMetadata(serialize::LayerMetadata);

#[pyclass(module = "hnsw_redux")]
pub struct Layer(::hnsw_redux::layer::Layer);

#[pymethods]
impl Layer {
    #[staticmethod]
    pub async fn from_loader(
        loader: LayerLoader,
    ) -> PyResult<Self /*, DataFusionError*/> {
        layer::Layer::from_loader(&*loader.0)
            .await
            .map(Self)
            .map_err(|datafusion_error| {
                let msg = format!("DataFusion error: {datafusion_error}");
                PyErr::new::<PyTypeError, _>(msg)
            })
    }

    // TODO: defer, not sure we need it
    // pub fn arrow_schema(&self) -> Arc<Schema> {
    //     self.0.arrow_schema()
    // }

    pub fn metadata(&self) -> LayerMetadata {
        LayerMetadata(self.0.metadata())
    }

    pub fn store(&self, dirpath: &str, layer_index: usize) -> PyResult<()> {
        Ok(self.0.store(dirpath, layer_index)?)
    }

    #[staticmethod]
    pub fn load(dirpath: &str, layer_index: usize) -> PyResult<Layer> {
        Ok(Self(::hnsw_redux::layer::Layer::load(
            dirpath,
            layer_index,
        )?))
    }
}

#[pyclass(module = "hnsw_redux")]
pub struct HnswMetadata(serialize::HnswMetadata);

#[pymethods]
impl HnswMetadata {
    pub fn layer_count(&self) -> usize {
        self.0.layer_count()
    }
}

#[pyclass(module = "hnsw_redux")]
pub struct Hnsw(hnsw::Hnsw);

#[pymethods]
impl Hnsw {
    #[staticmethod]
    pub async fn from_loader(loader: HnswLoader) -> PyResult<Self> {
        hnsw::Hnsw::from_loader(&*loader.0).await.map(Self).map_err(
            |datafusion_error| {
                let msg = format!("DataFusion error: {datafusion_error}");
                PyErr::new::<PyTypeError, _>(msg)
            },
        )
    }

    pub fn metadata(&self) -> HnswMetadata {
        HnswMetadata(self.0.metadata())
    }

    pub fn store(&self, dirpath: &str) -> PyResult<()> {
        Ok(self.0.store(dirpath)?)
    }

    #[staticmethod]
    pub fn load(dirpath: &str) -> PyResult<Self> {
        Ok(Self(hnsw::Hnsw::load(dirpath)?))
    }
}

#[pyclass(module = "hnsw_redux")]
pub struct HnswType(serialize::HnswType);

macro_rules! wrap_index_type {
    ($($type:ident),* $(,)?) => {
        $(
            #[pyclass(module = "hnsw_redux")]
            pub struct $type(index::$type);

            #[pymethods]
            impl $type {
                #[staticmethod]
                pub fn load(
                    name: &str,
                    hnsw_root_dirpath: &str,
                    vector_dirpath:&str,
                ) -> PyResult<Self> {
                    Ok(Self(index::$type::load(
                        name,
                        hnsw_root_dirpath,
                        vector_dirpath
                    )?))
                }

                pub fn store_hnsw(&self, hnsw_dirpath: &str) -> PyResult<()> {
                    Ok(self.0.store_hnsw(hnsw_dirpath)?)
                }
            }
        )*
    }
}

wrap_index_type![Hnsw1024, Hnsw1536, IndexConfiguration];

// TODO HNSW:
// - access beams
// - access distances
// - search([vector]) -> [queue]
