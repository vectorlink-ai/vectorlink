#![feature(test)]
extern crate test;
use hnsw_redux::{
    comparator::CosineDistance1024,
    hnsw::{BuildParams, Hnsw},
    test_util::random_vectors,
};
use test::Bencher;

#[bench]
fn bench_unquantized_1024_construction(b: &mut Bencher) {
    let number_of_vecs = 10000;
    let vecs = random_vectors(number_of_vecs, 1024, 0x533D);
    let comparator = CosineDistance1024::new(&vecs);
    let bp = BuildParams::default();

    b.iter(|| {
        let _ = Hnsw::generate(&bp, &comparator);
    });
}
