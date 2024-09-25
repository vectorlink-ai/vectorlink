#![feature(test)]
extern crate test;
use hnsw_redux::{
    comparator::EuclideanDistance8x8,
    hnsw::{BuildParams, Hnsw},
    layer::SearchParams,
    test_util::random_vectors,
    vectors::Vector,
};
use test::Bencher;

#[bench]
fn bench_centroid_construction(b: &mut Bencher) {
    let number_of_vecs = u16::MAX as usize;
    let vecs = random_vectors(number_of_vecs, 8, 0x533D);
    let comparator = EuclideanDistance8x8::new(&vecs);
    let bp = BuildParams::default();

    b.iter(|| {
        let _ = Hnsw::generate(&bp, &comparator);
    });
}

#[bench]
fn bench_symmetrize(b: &mut Bencher) {
    let number_of_vecs = u16::MAX as usize;
    let vecs = random_vectors(number_of_vecs, 8, 0x533D);
    let comparator = EuclideanDistance8x8::new(&vecs);
    let bp = BuildParams::default();
    let mut hnsw = Hnsw::generate(&bp, &comparator);
    let layer = hnsw.get_layer_mut(hnsw.layer_count() - 1);

    b.iter(|| layer.symmetrize(&comparator));
}

#[bench]
fn bench_improve(b: &mut Bencher) {
    let number_of_vecs = u16::MAX as usize;
    let vecs = random_vectors(number_of_vecs, 8, 0x533D);
    let comparator = EuclideanDistance8x8::new(&vecs);
    let bp = BuildParams::default();
    let mut hnsw = Hnsw::generate(&bp, &comparator);
    let sp = SearchParams::default();
    b.iter(|| hnsw.improve_neighbors_in_all_layers(&sp, &comparator));
}

#[bench]
fn bench_centroid_search(b: &mut Bencher) {
    let number_of_vecs = u16::MAX as usize;
    let vecs = random_vectors(number_of_vecs, 8, 0x533D);
    let comparator = EuclideanDistance8x8::new(&vecs);
    let bp = BuildParams::default();
    let mut hnsw = Hnsw::generate(&bp, &comparator);
    let sp = SearchParams::default();
    let vec = &random_vectors(1, 8, 0x12345)[0];
    b.iter(|| hnsw.search_from_initial(Vector::Slice(vec), &sp, &comparator));
}
