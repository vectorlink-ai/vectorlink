#![feature(test)]
extern crate test;
use hnsw_redux::{
    comparator::CosineDistance1024,
    hnsw::Hnsw,
    params::{BuildParams, SearchParams},
    test_util::random_vectors,
    vectors::Vector,
};
use test::Bencher;

#[bench]
fn bench_unquantized_1024_construction(b: &mut Bencher) {
    let number_of_vecs = 10000;
    let vecs = random_vectors(number_of_vecs, 1024, 0x533D);
    let comparator = CosineDistance1024::new(&vecs);
    let bp = BuildParams {
        order: 24,
        neighborhood_size: 24,
        bottom_neighborhood_size: 48,
        optimize_sp: SearchParams {
            parallel_visit_count: 1,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 0,
        },
    };

    b.iter(|| {
        let _ = Hnsw::generate(&bp, &comparator);
    });
}

#[bench]
fn bench_unquantized_1024_search(b: &mut Bencher) {
    let number_of_vecs = 10000;
    let vecs = random_vectors(number_of_vecs, 1024, 0x533D);
    let comparator = CosineDistance1024::new(&vecs);
    let bp = BuildParams {
        order: 24,
        neighborhood_size: 24,
        bottom_neighborhood_size: 48,
        optimize_sp: SearchParams {
            parallel_visit_count: 1,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 0,
        },
    };

    let hnsw = Hnsw::generate(&bp, &comparator);
    let sp = SearchParams {
        parallel_visit_count: 12,
        visit_queue_len: 100,
        search_queue_len: 30,
        circulant_parameter_count: 8,
    };
    let vec = &random_vectors(1, 1024, 0x12345)[0];
    b.iter(|| hnsw.search_from_initial(Vector::Slice(vec), &sp, &comparator));
}
