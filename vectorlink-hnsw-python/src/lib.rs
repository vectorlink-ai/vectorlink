#![allow(unexpected_cfgs)]

use ::vectorlink_hnsw::{
    comparator::CosineDistance1536, hnsw, index, layer, params, serialize,
    util, vectors,
};
use arrow::pyarrow::PyArrowType;
use datafusion::arrow::{
    array::{Array, ArrayData},
    ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream},
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyIndexError, PyRuntimeError},
    prelude::*,
    types::PyCapsule,
};

// This function defines a Python module. Its name MUST match the the `lib.name`
// settings in `Cargo.toml`, else Python will not be able to import the module.
#[pymodule]
fn vectorlink_hnsw(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_class::<BuildParams>()?;
    m.add_class::<OptimizationParams>()?;
    m.add_class::<SearchParams>()?;
    m.add("SerializeError", m.py().get_type::<SerializeError>())?;
    Ok(())
}

#[pyclass(module = "vectorlink_hnsw")]
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
        let arrow_stream = arrow_stream.pointer() as *mut FFI_ArrowArrayStream;
        let reader =
            unsafe { ArrowArrayStreamReader::from_raw(arrow_stream).unwrap() };
        vectors::Vectors::from_arrow(reader, number_of_records)
            .map(Self)
            .map_err(|err| SerializeError::new_err(format!("{err:?}")))
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
            let msg = format!("Index {index} is out of bounds");
            Err(PyIndexError::new_err(msg))
        }
    }

    pub fn store(&self, dirpath: &str, identity: &str) -> PyResult<()> {
        self.0.store(dirpath, identity)?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(dirpath: &str, identity: &str) -> PyResult<Self> {
        Ok(Self(vectors::Vectors::load(dirpath, identity)?))
    }

    pub fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }
}

// PyO3 / Maturin require that no generics are used at FFI boundaries.
// This means that for types like `vectorlink_hnsw::util::SimdAlignedAllocation<E>`,
// the SIMD element type parameter `E` needs to be filled out rather than be
// left up to the user.  This macro does that by providing that `$element` while
// also wrapping the generic type into a new `SimdAlignedAllocation*` type for
// each `$element`.
// For example, for `$element = u8`, it generates the `SimdAlignedAllocationU8`
// type, which wraps `vectorlink_hnsw::util::SimdAlignedAllocation<u8>`.
macro_rules! wrap_SimdAlignedAllocation_type_for_element {
    ($($element:ty),*) => {
        paste::paste! {
            $(
                #[pyclass(module = "vectorlink_hnsw")]
                #[derive(Clone, Debug)]
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

                    pub fn __repr__(&self) -> String {
                        format!("{:#?}", self.0)
                    }
                }
            )*
        }
    }
}

// These wrapper types for `SimdAlignedAllocation<_>`
// are needed for `self::Vectors::new()`
wrap_SimdAlignedAllocation_type_for_element![u8];

#[allow(unused)]
#[pyclass(module = "vectorlink_hnsw")]
pub struct LayerMetadata(serialize::LayerMetadata);

#[pymethods]
impl LayerMetadata {
    pub fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }
}

#[pyclass(module = "vectorlink_hnsw")]
#[derive(Clone)]
pub struct Layer(::vectorlink_hnsw::layer::Layer);

#[pymethods]
impl Layer {
    #[staticmethod]
    pub fn from_arrow(
        arrow_stream: Bound<'_, PyCapsule>,
        number_of_records: usize,
    ) -> PyResult<Self> {
        let arrow_stream = arrow_stream.pointer() as *mut FFI_ArrowArrayStream;
        let reader =
            unsafe { ArrowArrayStreamReader::from_raw(arrow_stream).unwrap() };
        layer::Layer::from_arrow(reader, number_of_records)
            .map(Self)
            .map_err(|err| SerializeError::new_err(format!("{err:?}")))
    }

    pub fn to_arrow_array(&self) -> PyResult<PyArrowType<ArrayData>> {
        let array = self.0.to_arrow_array();
        array
            .map(|a| PyArrowType(a.into_data()))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
        Ok(Self(::vectorlink_hnsw::layer::Layer::load(
            dirpath,
            layer_index,
        )?))
    }

    pub fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }
}


#[pyclass(module = "vectorlink_hnsw")]
pub struct HnswMetadata(serialize::HnswMetadata);

#[pymethods]
impl HnswMetadata {
    pub fn layer_count(&self) -> usize {
        self.0.layer_count()
    }

    pub fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }
}

#[pyclass(module = "vectorlink_hnsw")]
pub struct Hnsw(hnsw::Hnsw);

fn ensure_1536_vector_size(vecs: &Vectors) -> PyResult<()> {
    const F32_SIZE: usize = std::mem::size_of::<f32>();
    let vector_byte_size = vecs.0.vector_byte_size();
    if vector_byte_size != 1536 * F32_SIZE {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Vector dimensionality expected to be 1536, but found {}",
            vector_byte_size / F32_SIZE
        )));
    }

    Ok(())
}

#[pymethods]
impl Hnsw {
    #[new]
    pub fn new(layers: Vec<Layer>) -> Self {
        // TODO To make this work, `Layer: Clone` is needed - which likely means
        //      that the FFI boundary actuyally clones the damn layers.
        //      Fix that.
        let layers: Vec<_> = layers.into_iter().map(|layer| layer.0).collect();
        Self(hnsw::Hnsw::new(layers))
    }

    #[staticmethod]
    pub fn generate_with_cosine_distance_1536(
        bp: BuildParams,
        vecs: &Vectors,
    ) -> PyResult<Self> {
        ensure_1536_vector_size(vecs)?;
        let default_comparator = CosineDistance1536::new(&vecs.0);
        Ok(Self(hnsw::Hnsw::generate(
            &bp.into_raw(),
            &default_comparator,
        )))
    }

    pub fn optimize_with_cosine_distance_1536(
        &mut self,
        op: OptimizationParams,
        vecs: &Vectors,
    ) -> PyResult<f32> {
        ensure_1536_vector_size(vecs)?;
        let default_comparator = CosineDistance1536::new(&vecs.0);
        Ok(self.0.optimize(&op.into_raw(), &default_comparator))
    }

    #[staticmethod]
    pub fn from_arrow(
        arrow_stream: Bound<'_, PyCapsule>,
        number_of_records: usize,
    ) -> PyResult<Self> {
        let arrow_stream = arrow_stream.pointer() as *mut FFI_ArrowArrayStream;
        let reader =
            unsafe { ArrowArrayStreamReader::from_raw(arrow_stream).unwrap() };
        let layer = layer::Layer::from_arrow(reader, number_of_records)
            .map(Layer)
            .map_err(|err| SerializeError::new_err(format!("{err:?}")))?;
        Ok(Self::new(vec![layer]))
    }

    pub fn layer_count(&self) -> usize {
        self.0.layer_count()
    }

    pub fn layer_arrow_array(
        &self,
        index: usize,
    ) -> PyResult<PyArrowType<ArrayData>> {
        self.0.layers()[index]
            .to_arrow_array()
            .map(|a| PyArrowType(a.into_data()))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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

    pub fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }
}

#[allow(unused)]
#[pyclass(module = "vectorlink_hnsw")]
pub struct HnswType(serialize::HnswType);

impl HnswType {
    pub fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }
}

macro_rules! wrap_index_type {
    ($($type:ident),* $(,)?) => {
        $(
            #[pyclass(module = "vectorlink_hnsw")]
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

                pub fn __repr__(&self) -> String {
                    format!("{:#?}", self.0)
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

create_exception!(
    vectorlink_hnsw,
    SerializeError,
    PyException,
    "Failed to serialize"
);

#[derive(Clone, Copy, Debug)]
#[pyclass(module = "vectorlink_hnsw")]
pub struct BuildParams {
    #[pyo3(get, set)]
    pub order: usize,
    #[pyo3(get, set)]
    pub neighborhood_size: usize,
    #[pyo3(get, set)]
    pub bottom_neighborhood_size: usize,
    #[pyo3(get, set)]
    pub optimization_params: OptimizationParams,
}

impl BuildParams {
    fn into_raw(self) -> params::BuildParams {
        params::BuildParams {
            order: self.order,
            neighborhood_size: self.neighborhood_size,
            bottom_neighborhood_size: self.bottom_neighborhood_size,
            optimization_params: self.optimization_params.into_raw(),
        }
    }
}

#[pymethods]
impl BuildParams {
    #[staticmethod]
    pub fn default() -> Self {
        Self::from(params::BuildParams::default())
    }

    pub fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

impl From<params::BuildParams> for BuildParams {
    fn from(p: params::BuildParams) -> Self {
        Self {
            order: p.order,
            neighborhood_size: p.neighborhood_size,
            bottom_neighborhood_size: p.bottom_neighborhood_size,
            optimization_params: p.optimization_params.into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[pyclass(module = "vectorlink_hnsw")]
pub struct OptimizationParams {
    #[pyo3(get, set)]
    pub search_params: SearchParams,
    #[pyo3(get, set)]
    pub improvement_threshold: f32,
    #[pyo3(get, set)]
    pub recall_target: f32,
}

#[pymethods]
impl OptimizationParams {
    #[staticmethod]
    pub fn default() -> Self {
        Self::from(params::OptimizationParams::default())
    }

    pub fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

impl OptimizationParams {
    fn into_raw(self) -> params::OptimizationParams {
        params::OptimizationParams {
            search_params: self.search_params.into_raw(),
            improvement_threshold: self.improvement_threshold,
            recall_target: self.recall_target,
        }
    }
}

impl From<params::OptimizationParams> for OptimizationParams {
    fn from(p: params::OptimizationParams) -> Self {
        Self {
            search_params: p.search_params.into(),
            improvement_threshold: p.improvement_threshold,
            recall_target: p.recall_target,
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[pyclass(module = "vectorlink_hnsw")]
pub struct SearchParams {
    #[pyo3(get, set)]
    pub parallel_visit_count: usize,
    #[pyo3(get, set)]
    pub visit_queue_len: usize,
    #[pyo3(get, set)]
    pub search_queue_len: usize,
    #[pyo3(get, set)]
    pub circulant_parameter_count: usize,
}

#[pymethods]
impl SearchParams {
    #[staticmethod]
    pub fn default() -> Self {
        Self::from(params::SearchParams::default())
    }

    pub fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

impl SearchParams {
    fn into_raw(self) -> params::SearchParams {
        params::SearchParams {
            parallel_visit_count: self.parallel_visit_count,
            visit_queue_len: self.visit_queue_len,
            search_queue_len: self.search_queue_len,
            circulant_parameter_count: self.circulant_parameter_count,
        }
    }
}

impl From<params::SearchParams> for SearchParams {
    fn from(p: params::SearchParams) -> Self {
        Self {
            parallel_visit_count: p.parallel_visit_count,
            visit_queue_len: p.visit_queue_len,
            search_queue_len: p.search_queue_len,
            circulant_parameter_count: p.circulant_parameter_count,
        }
    }
}
