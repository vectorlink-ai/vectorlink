use ::hnsw_redux::{
    serialize,
    util,
    vectors,
};
use ::pyo3::{
    create_exception,
    exceptions::{PyException, PyIndexError, PyTypeError},
    prelude::*,
    pyclass::PyClass,
    types::IntoPyDict,
};


// This function defines a Python module. Its name MUST match the the `lib.name`
// settings in `Cargo.toml`, else Python will not be able to import the module.
#[pymodule]
fn hnsw_redux(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IndexError", py.get_type::<IndexError>())?;
    m.add_class::<SimdAlignedAllocationU8>()?;
    m.add_class::<Vectors>()?;
    Ok(())
}


#[allow(unused)]
#[pyclass(module = "hnsw_redux")]
pub struct Vectors(vectors::Vectors);

#[pymethods]
impl Vectors {
    // #[staticmethod]
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

    // #[staticmethod]
    // pub fn from_reader<R: Read>(
    //     mut reader: R,
    //     num_vecs: usize,
    //     vector_byte_size: usize,
    // ) -> PyResult<Self> {
    //     todo!() // TODO: fill out the type param
    // }

    pub fn num_vecs(&self) -> usize {
        self.0.num_vecs()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn vector_byte_size(&self) -> usize {
        self.0.vector_byte_size()
    }

    // pub fn get<T>(&self, index: usize) -> Option<&T> {
    //     todo!() // TODO: fill out the type param
    // }

    // pub fn get_mut<T>(&mut self, index: usize) -> &mut T {
    //     todo!() // TODO: fill out the type param
    // }

    // pub fn get_mut_f32_slice(&mut self, index: usize) -> &mut [f32] {
    //     todo!() // TODO: fill out the type param
    // }

    pub fn data(&self) -> &[u8] {
        self.0.data()
    }

    // pub fn iter(&self) -> impl Iterator<Item = &[u8]> + '_ {
    //     todo!() // TODO deal with `impl Trait` generics
    // }

    // pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &[u8]> + '_ {
    //     todo!() // TODO deal with `impl Trait` generics
    // }

    pub fn normalize(&mut self) {
        self.0.normalize()
    }

    pub fn __getitem__(&self, key: usize) -> PyResult<&[u8]> {
        if key < self.num_vecs() {
            Ok(&self.0[key])
        } else {
            Err(PyIndexError::new_err(format!("index {key} is out of bounds")))
        }
    }

    // TODO: is __setitem__ needed?
    // pub fn __setitem__(&mut self, key: usize, val: &[u8]) {
    //     let min_len = std::cmp::min(self.0[key].len(), val.len());
    //     // Since a [u8] value cannot be assigned as such, write
    //     // one eleement at a time:
    //     for i in 0..min_len {
    //         self.0[key][i] = val[i]
    //     }
    // }


    #[staticmethod]
    pub async fn from_loader(l: &dyn self::VectorsLoader) -> PyResult<Self> {
        // TODO: Provide a way to obtain a `serialize::VectorsLoader` instance
        vectors::Vectors::from_loader(l).await
            .map(Self)
            .map_err(|datafusion_err| {
                let msg = format!("DataFusion error: {datafusion_err}");
                PyErr::new::<PyTypeError, _>(msg)
            })
    }

    pub fn store(&self, dirpath: &str, identity: &str) -> PyResult<()> {
        self.0.store(dirpath, identity)?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(dirpath: &str, identity: &str) -> PyResult<Self> {
        Ok(Self(vectors::Vectors::load(dirpath, identity)?))
    }
}

create_exception!(hnsw_redux, IndexError, PyException);


pub trait VectorsLoader: serialize::VectorsLoader + PyClass {}


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



pub struct LayerMetadata(serialize::LayerMetadata);



// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn it_works() {
//         todo!("Blah")
//     }
// }
