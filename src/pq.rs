use crate::hnsw::Hnsw;

pub struct Pq<'a, FC, CC, QC> {
    full_comparator: &'a FC,
    centroid_comparator: &'a CC,
    quantized_comparator: &'a QC,
    centroid_hnsw: Hnsw,
    quantized_hnsw: Hnsw,
}
