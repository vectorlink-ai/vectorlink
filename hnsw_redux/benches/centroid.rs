#![feature(test)]
extern crate test;
use hnsw_redux::{
    comparator::EuclideanDistance8x8,
    hnsw::Hnsw,
    params::{BuildParams, OptimizationParams, SearchParams},
    test_util::random_vectors,
    vectors::Vector,
};
use test::Bencher;

#[bench]
fn bench_centroid_construction(b: &mut Bencher) {
    let number_of_vecs = u16::MAX as usize;
    let vecs = random_vectors(number_of_vecs, 8, 0x533D);
    let comparator = EuclideanDistance8x8::new(&vecs);
    let bp = BuildParams {
        order: 24,
        neighborhood_size: 24,
        bottom_neighborhood_size: 48,
        optimization_params: OptimizationParams {
            search_params: SearchParams {
                parallel_visit_count: 1,
                visit_queue_len: 100,
                search_queue_len: 30,
                circulant_parameter_count: 8,
            },
            improvement_threshold: 0.01,
            recall_target: 1.0,
        },
    };

    b.iter(|| {
        let _ = Hnsw::generate(&bp, &comparator);
    });
}

#[bench]
fn bench_symmetrize(b: &mut Bencher) {
    let number_of_vecs = u16::MAX as usize;
    let vecs = random_vectors(number_of_vecs, 8, 0x533D);
    let comparator = EuclideanDistance8x8::new(&vecs);
    let bp = BuildParams {
        order: 24,
        neighborhood_size: 24,
        bottom_neighborhood_size: 48,
        optimization_params: OptimizationParams {
            search_params: SearchParams {
                parallel_visit_count: 1,
                visit_queue_len: 100,
                search_queue_len: 30,
                circulant_parameter_count: 8,
            },
            improvement_threshold: 0.01,
            recall_target: 1.0,
        },
    };
    let mut hnsw = Hnsw::generate(&bp, &comparator);
    let layer = hnsw.get_layer_mut(hnsw.layer_count() - 1);
    let mut distances = layer.neighborhood_distances(&comparator);
    let mut optimizer = layer.get_optimizer(&mut distances);

    b.iter(|| optimizer.symmetrize());
}

#[bench]
fn bench_improve(b: &mut Bencher) {
    let number_of_vecs = u16::MAX as usize;
    let vecs = random_vectors(number_of_vecs, 8, 0x533D);
    let comparator = EuclideanDistance8x8::new(&vecs);
    let bp = BuildParams {
        order: 24,
        neighborhood_size: 24,
        bottom_neighborhood_size: 48,
        optimization_params: OptimizationParams {
            search_params: SearchParams {
                parallel_visit_count: 1,
                visit_queue_len: 100,
                search_queue_len: 30,
                circulant_parameter_count: 8,
            },
            improvement_threshold: 0.01,
            recall_target: 1.0,
        },
    };
    let mut hnsw = Hnsw::generate(&bp, &comparator);
    let op = OptimizationParams {
        search_params: SearchParams {
            parallel_visit_count: 12,
            visit_queue_len: 100,
            search_queue_len: 30,
            circulant_parameter_count: 8,
        },
        improvement_threshold: 0.01,
        recall_target: 1.0,
    };

    b.iter(|| hnsw.optimize(&op, &comparator));
}

#[bench]
fn bench_centroid_search(b: &mut Bencher) {
    let number_of_vecs = u16::MAX as usize;
    let vecs = random_vectors(number_of_vecs, 8, 0x533D);
    let comparator = EuclideanDistance8x8::new(&vecs);
    let bp = BuildParams {
        order: 24,
        neighborhood_size: 24,
        bottom_neighborhood_size: 48,
        optimization_params: OptimizationParams {
            search_params: SearchParams {
                parallel_visit_count: 1,
                visit_queue_len: 100,
                search_queue_len: 30,
                circulant_parameter_count: 8,
            },
            improvement_threshold: 0.01,
            recall_target: 1.0,
        },
    };

    let hnsw = Hnsw::generate(&bp, &comparator);
    let sp = SearchParams {
        parallel_visit_count: 12,
        visit_queue_len: 100,
        search_queue_len: 30,
        circulant_parameter_count: 8,
    };

    let vec = &random_vectors(1, 8, 0x12345)[0];
    b.iter(|| hnsw.search_from_initial(Vector::Slice(vec), &sp, &comparator));
}
