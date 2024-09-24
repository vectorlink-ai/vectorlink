#![feature(test)]
extern crate test;
use hnsw_redux::{
    comparator::EuclideanDistance8x8,
    hnsw::{BuildParams, Hnsw},
    test_util::random_8_vectors,
};
use test::Bencher;

#[bench]
fn bench_centroid_construction(b: &mut Bencher) {
    let number_of_vecs = u16::MAX as usize;
    let vecs = random_8_vectors(number_of_vecs, 0x533D);
    let comparator = EuclideanDistance8x8::new(&vecs);
    let bp = BuildParams::default();

    b.iter(|| {
        let _ = Hnsw::generate(&bp, &comparator);
    });
}
