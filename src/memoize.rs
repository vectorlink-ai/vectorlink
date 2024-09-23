use rayon::prelude::*;

// i < j, i != j
#[inline]
fn index_to_offset(n: usize, i: usize, j: usize) -> usize {
    let i_f64 = i as f64;
    let j_f64 = j as f64;
    let n_f64 = n as f64;
    let correction = (i_f64 + 2.0) * (i_f64 + 1.0) / 2.0;
    (i_f64 * n_f64 + j_f64 - correction) as usize
}

// offset = i*n - (i + 2) * (i + 1) / 2 + j
//
fn offset_to_index(n: usize, offset: usize) -> (usize, usize) {
    let d = (2 * n - 1).pow(2) - 8 * offset;
    let i2 = (2 * n - 1) as f64 - (d as f64).sqrt();
    let i = (i2 / 2.0) as usize;
    let triangle = (i * (n - 1)) - ((i + 1) * i) / 2;
    let j = offset + 1 - triangle;
    (i, j)
}

#[inline]
fn triangle_lookup_length(n: usize) -> usize {
    index_to_offset(n, n - 2, n - 1) + 1
}

pub trait CentroidDistanceCalculator: Sync {
    fn num_centroids(&self) -> usize;
    fn calculate_centroid_distance(&self, c1: u16, c2: u16) -> f16;
    fn calculate_centroid_norm(&self, c: u16) -> f16;
}

pub struct MemoizedCentroidDistances {
    distances: Vec<f16>,
    norms: Vec<f16>,
    size: usize,
}

impl MemoizedCentroidDistances {
    pub fn new<C: CentroidDistanceCalculator>(calculator: &C) -> Self {
        let size = calculator.num_centroids();
        eprintln!("constructing memoized");
        let memoized_array_length = triangle_lookup_length(size);
        eprintln!("for size {} we figured {memoized_array_length}", size);
        let mut distances: Vec<f16> = Vec::with_capacity(memoized_array_length);
        {
            let partial_distances_uninit = distances.spare_capacity_mut();
            partial_distances_uninit
                .par_iter_mut()
                .enumerate()
                .for_each(|(c, elt)| {
                    let (i, j) = offset_to_index(size, c);
                    if i > 65535 || j > 65535 {
                        panic!("oh no {i} {j}");
                    }
                    elt.write(calculator.calculate_centroid_distance(i as u16, j as u16));
                });
        }
        unsafe {
            distances.set_len(memoized_array_length);
        }
        let norms: Vec<_> = (0..size)
            .map(|i| calculator.calculate_centroid_norm(i as u16))
            .collect();
        Self {
            distances,
            norms,
            size,
        }
    }

    pub fn lookup_centroid_distance(&self, i: u16, j: u16) -> f16 {
        let offset = match i.cmp(&j) {
            std::cmp::Ordering::Equal => {
                // Early bail
                return self.lookup_centroid_norm(i);
            }
            std::cmp::Ordering::Less => index_to_offset(self.size, i as usize, j as usize),
            std::cmp::Ordering::Greater => index_to_offset(self.size, j as usize, i as usize),
        };
        let distance: f16 = self.distances[offset];
        distance
    }

    pub fn lookup_centroid_norm(&self, i: u16) -> f16 {
        self.norms[i as usize]
    }
}

#[cfg(test)]
mod offsettest {
    use super::*;
    use rand::prelude::*;
    #[test]
    fn test_triangle_offsets() {
        let n = 100;
        let mut expected_index = 0;
        for i in 0..n {
            for j in 0..n {
                if i < j {
                    let actual = index_to_offset(n, i, j);
                    assert_eq!(expected_index, actual);
                    expected_index += 1;
                }
            }
        }
        assert_eq!(expected_index, triangle_lookup_length(n));
    }

    #[test]
    #[ignore]
    fn roundtrip() {
        let n = 65535;
        (0..triangle_lookup_length(n))
            .into_par_iter()
            .for_each(|i| {
                let (a, b) = offset_to_index(n, i);
                if a >= n {
                    eprintln!("Yikes: a: {a}, b: {b}, n: {n}");
                }
                if b >= n {
                    eprintln!("Yikes: a: {a}, b: {b}, n: {n}");
                }
                assert!(a < n);
                assert!(b < n);

                if a == 0 && b == 0 {
                    panic!("Failure at {i}: a: {a}, b: {b}, n: {n}");
                }
                let i2 = index_to_offset(n, a, b);
                if i != i2 {
                    panic!("Failure n: {n}, a: {a}, b: {b}, i: {i}, i2: {i2}");
                }
                assert_eq!(i, i2);
            });
    }

    #[test]
    #[ignore]
    fn roundtrip_backwards() {
        let n = 65535;
        (0..n)
            .into_par_iter()
            .flat_map(|a| (0..n).into_par_iter().map(move |b| (a, b)))
            .for_each(|(a, b)| {
                if a >= b {
                    return;
                }
                let i = index_to_offset(n, a, b);
                let (a2, b2) = offset_to_index(n, i);
                if a2 != a || b2 != b {
                    panic!("omfg: a: {a}, b: {b}, a2: {a2}, b2: {b2}, i: {i}")
                }
            });
    }

    struct IndexProductDistanceCalculator;

    fn scaled_multiple(left: usize, right: usize) -> f16 {
        const SCALE: f32 = (u32::MAX as usize + 1) as f32 / 16.0;
        ((left * right) as f32 / SCALE) as f16
    }

    impl CentroidDistanceCalculator for IndexProductDistanceCalculator {
        fn num_centroids(&self) -> usize {
            (u16::MAX as usize) + 1
        }
        fn calculate_centroid_distance(&self, left: u16, right: u16) -> f16 {
            scaled_multiple(left as usize, right as usize)
        }
        fn calculate_centroid_norm(&self, vec: u16) -> f16 {
            let vec = vec as usize;
            scaled_multiple(vec, vec)
        }
    }

    #[test]
    fn distances_are_mapped_right() {
        let distances = MemoizedCentroidDistances::new(&IndexProductDistanceCalculator);
        let mut rng = StdRng::seed_from_u64(2024);
        let mut set1: Vec<_> = (0..=u16::MAX).collect();
        set1.shuffle(&mut rng);
        set1.truncate(1000);
        set1.shrink_to_fit();
        let mut set2: Vec<_> = (0..=u16::MAX).collect();
        set2.shuffle(&mut rng);
        set2.truncate(1000);
        set2.shrink_to_fit();
        set1.into_par_iter()
            .flat_map(|a| set2.par_iter().map(move |b| (a, *b)))
            .for_each(|(a, b)| {
                if a == b {
                    return;
                }
                let distance = distances.lookup_centroid_distance(a, b) as f32;
                let calculated_distance = scaled_multiple(a as usize, b as usize) as f32;
                let error = (distance - calculated_distance).abs();
                if error > 0.1 {
                    panic!("{a},{b} gave distance {distance} and not {calculated_distance}");
                }
            });
    }
}
