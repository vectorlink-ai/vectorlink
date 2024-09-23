#![feature(portable_simd)]
#![feature(f16)]
pub mod bitmap;
pub mod comparator;
pub mod hnsw;
pub mod layer;
pub mod memoize;
pub mod ring_queue;
pub mod vecmath;
pub mod vectors;

#[cfg(test)]
mod tests {}
