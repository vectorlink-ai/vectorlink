#![feature(portable_simd)]
#![feature(f16)]
#![allow(clippy::field_reassign_with_default)]
pub mod bitmap;
pub mod comparator;
pub mod hnsw;
pub mod index;
pub mod layer;
pub mod memoize;
pub mod optimize;
pub mod params;
pub mod peaks;
pub mod pq;
pub mod queue_view;
pub mod ring_queue;
pub mod serialize;
pub mod test_util;
pub mod util;
pub mod vecmath;
pub mod vectors;

#[cfg(test)]
mod tests {}
