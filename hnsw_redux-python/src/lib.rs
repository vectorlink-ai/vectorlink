#![allow(unused)] // TODO: remove

use ::datafusion::arrow::datatypes::Schema;
use ::hnsw_redux::{
    hnsw,
    index,
    serialize,
    util,
    vectors,
};
use ::pyo3::{
    exceptions::{PyException, PyIndexError, PyTypeError},
    prelude::*,
    pyclass::PyClass,
    types::{IntoPyDict, PyDict},
};
use std::sync::Arc;


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
    pub fn from_file(filepath: &str, vector_byte_size: usize) -> PyResult<Self> {
        Ok(Self(vectors::Vectors::from_file(filepath, vector_byte_size)?))
    }

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
            Err(PyIndexError::new_err("error message"))
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

    // pub async fn from_loader(loader: &dyn LayerLoader) -> Result<Self, DataFusionError> {
    //     // TODO: Provide a way to
    //     //       1. obtain a `serialize::LayerLoader` instance, or
    //     //       2. load from a DataFusion DataFrame
    //     todo!() // TODO
    // }

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
        Ok(Self(::hnsw_redux::layer::Layer::load(dirpath, layer_index)?))
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

    // pub async fn from_loader(
    //     loader: &dyn HnswLoader
    // ) -> Result<Self, DataFusionError> {
    //     // TODO: Provide a way to
    //     //       1. obtain a `serialize::HnswLoader` instance, or
    //     //       2. load from a DataFusion DataFrame
    //     todo!() // TODO
    // }

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
