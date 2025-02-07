#![feature(test)]

extern crate test;
use vectorlink_hnsw::{
    comparator::{NewEuclideanDistance8x8, VectorComparatorConstructor},
    hnsw::Hnsw,
    params::{BuildParams, SearchParams},
    pq::{centroid_finder, Quantizer, VectorRangeIndexableForVectors},
    test_util::random_vectors,
};
use test::Bencher;

#[bench]
fn bench_quantization(b: &mut Bencher) {
    let number_of_vecs = 10000;
    let vectors = random_vectors(number_of_vecs, 1024, 0x533D);
    let vector_indexable = VectorRangeIndexableForVectors(&vectors);
    let centroid_count = u16::MAX as usize;
    let centroid_byte_size = 8 * std::mem::size_of::<f32>();
    let centroid_build_params = BuildParams::default();
    let quantizer_search_params = SearchParams {
        parallel_visit_count: 8,
        visit_queue_len: 30,
        search_queue_len: 10,
        circulant_parameter_count: 8,
    };
    let centroids = centroid_finder(
        &vector_indexable,
        centroid_count,
        centroid_byte_size,
        0xD335,
    );

    let centroid_comparator = NewEuclideanDistance8x8::new_from_vecs(&centroids);
    let centroid_hnsw = Hnsw::generate(&centroid_build_params, &centroid_comparator);
    let quantizer = Quantizer::new(centroid_hnsw, quantizer_search_params);
    b.iter(move || {
        let vector_stream = vectors.iter();
        quantizer.quantize_all(vectors.num_vecs(), vector_stream, &centroid_comparator)
    });
}
