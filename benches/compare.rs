#![feature(test)]

extern crate test;
use hnsw_redux::vecmath;
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use test::Bencher;

#[bench]
fn bench_dot_product_8(b: &mut Bencher) {
    const VECS: usize = 10000;
    let mut rng = StdRng::seed_from_u64(2024);
    let range = Uniform::from(-1.0..1.0);
    let v1: Vec<f32> = (0..1024).map(|_| rng.sample(range)).collect();
    let vs2: Vec<f32> = (0..1024 * VECS).map(|_| rng.sample(range)).collect();

    b.iter(|| {
        for i in 0..VECS {
            let v2 = &vs2[(i * 1024)..((i + 1) * 1024)];
            vecmath::dot_product_1024_8(&v1, v2);
        }
    });
}

#[bench]
fn bench_dot_product_16(b: &mut Bencher) {
    const VECS: usize = 10000;
    let mut rng = StdRng::seed_from_u64(2024);
    let range = Uniform::from(-1.0..1.0);
    let v1: Vec<f32> = (0..1024).map(|_| rng.sample(range)).collect();
    let vs2: Vec<f32> = (0..1024 * VECS).map(|_| rng.sample(range)).collect();

    b.iter(|| {
        for i in 0..VECS {
            let v2 = &vs2[(i * 1024)..((i + 1) * 1024)];
            vecmath::dot_product_1024_16(&v1, v2);
        }
    });
}

#[bench]
fn bench_dot_product_32(b: &mut Bencher) {
    const VECS: usize = 10000;
    let mut rng = StdRng::seed_from_u64(2024);
    let range = Uniform::from(-1.0..1.0);
    let v1: Vec<f32> = (0..1024).map(|_| rng.sample(range)).collect();
    let vs2: Vec<f32> = (0..1024 * VECS).map(|_| rng.sample(range)).collect();

    b.iter(|| {
        for i in 0..VECS {
            let v2 = &vs2[(i * 1024)..((i + 1) * 1024)];
            vecmath::dot_product_1024_32(&v1, v2);
        }
    });
}

#[bench]
fn bench_dot_product_64(b: &mut Bencher) {
    const VECS: usize = 10000;
    let mut rng = StdRng::seed_from_u64(2024);
    let range = Uniform::from(-1.0..1.0);
    let v1: Vec<f32> = (0..1024).map(|_| rng.sample(range)).collect();
    let vs2: Vec<f32> = (0..1024 * VECS).map(|_| rng.sample(range)).collect();

    b.iter(|| {
        for i in 0..VECS {
            let v2 = &vs2[(i * 1024)..((i + 1) * 1024)];
            vecmath::dot_product_1024_64(&v1, v2);
        }
    });
}

enum Indirection {
    Size1024,
    Size1536,
}

impl Indirection {
    #[inline(always)]
    fn dot_product(&self, v1: &[f32], v2: &[f32]) -> f32 {
        match self {
            Indirection::Size1024 => vecmath::dot_product_1024_16(v1, v2),
            Indirection::Size1536 => vecmath::dot_product_1536_16(v1, v2),
        }
    }
}

#[bench]
fn bench_indirect_dot_product(b: &mut Bencher) {
    const VECS: usize = 10000;
    let mut rng = StdRng::seed_from_u64(2024);
    let range = Uniform::from(-1.0..1.0);
    let v1: Vec<f32> = (0..1024).map(|_| rng.sample(range)).collect();
    let vs2: Vec<f32> = (0..1024 * VECS).map(|_| rng.sample(range)).collect();

    let indirection = Indirection::Size1024;

    b.iter(|| {
        for i in 0..VECS {
            let v2 = &vs2[(i * 1024)..((i + 1) * 1024)];
            indirection.dot_product(&v1, v2);
        }
    });
}

trait DotProduct {
    fn dot_product(&self, v1: &[f32], v2: &[f32]) -> f32;
}

struct DoctProduct1024;
impl DotProduct for DoctProduct1024 {
    #[inline(always)]
    fn dot_product(&self, v1: &[f32], v2: &[f32]) -> f32 {
        vecmath::dot_product_1024_16(v1, v2)
    }
}

#[bench]
fn bench_trait_obj_dot_product(b: &mut Bencher) {
    const VECS: usize = 10000;
    let mut rng = StdRng::seed_from_u64(2024);
    let range = Uniform::from(-1.0..1.0);
    let v1: Vec<f32> = (0..1024).map(|_| rng.sample(range)).collect();
    let vs2: Vec<f32> = (0..1024 * VECS).map(|_| rng.sample(range)).collect();

    let trait_obj: Box<dyn DotProduct> = Box::new(DoctProduct1024);

    b.iter(|| {
        for i in 0..VECS {
            let v2 = &vs2[(i * 1024)..((i + 1) * 1024)];
            trait_obj.dot_product(&v1, v2);
        }
    });
}
