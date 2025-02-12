#![feature(test)]

extern crate test;
use test::Bencher;
use vectorlink_hnsw::{
    comparator::{
        NewDotProductCentroidDistanceCalculator8, NewEuclideanDistance8x8,
        NewMemoizedComparator128,
    },
    index::{Index, IndexConfiguration, Pq1024x8},
    params::{BuildParams, SearchParams},
    pq::{create_pq, VectorRangeIndexableForVectors},
    test_util::random_vectors,
    vectors::Vector,
};

#[bench]
fn bench_pq_construction(b: &mut Bencher) {
    let number_of_vecs = 10_000;
    let vecs = random_vectors(number_of_vecs, 1024, 0x533D);
    let vector_indexable = VectorRangeIndexableForVectors(&vecs);
    let centroid_count = u16::MAX as usize;
    let centroid_byte_size = 8 * std::mem::size_of::<f32>();
    let centroid_build_params = BuildParams::default();
    let quantized_build_params = BuildParams::default();
    let quantizer_search_params = SearchParams::default();
    b.iter(|| {
        let vector_stream = vecs.iter();
        let _pq = create_pq::<
            NewEuclideanDistance8x8,
            NewMemoizedComparator128,
            NewDotProductCentroidDistanceCalculator8,
            _,
            _,
        >(
            &vector_indexable,
            vector_stream,
            centroid_count,
            centroid_byte_size,
            &centroid_build_params,
            &quantized_build_params,
            quantizer_search_params,
            0x533D,
        );
    });
}

#[bench]
fn bench_pq_search(b: &mut Bencher) {
    let number_of_vecs = 100_000;
    let vecs = random_vectors(number_of_vecs, 1024, 0x533D);
    let vector_indexable = VectorRangeIndexableForVectors(&vecs);
    let vector_stream = vecs.iter();
    let centroid_count = u16::MAX as usize;
    let centroid_byte_size = 8 * std::mem::size_of::<f32>();
    let centroid_build_params = BuildParams::default();
    let quantized_build_params = BuildParams::default();
    let quantizer_search_params = SearchParams::default();

    let pq = create_pq::<
        NewEuclideanDistance8x8,
        NewMemoizedComparator128,
        NewDotProductCentroidDistanceCalculator8,
        _,
        _,
    >(
        &vector_indexable,
        vector_stream,
        centroid_count,
        centroid_byte_size,
        &centroid_build_params,
        &quantized_build_params,
        quantizer_search_params,
        0x533D,
    );
    let index = IndexConfiguration::Pq1024x8(Pq1024x8::new(
        "dummy".to_string(),
        pq,
        vecs,
    ));
    let vec = &random_vectors(1, 1024, 0x12345)[0];
    let sp = SearchParams::default();
    b.iter(|| index.search(Vector::Slice(vec), &sp));
}
