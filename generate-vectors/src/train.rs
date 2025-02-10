use crate::graph::CompareGraph;
use argmin::core::{CostFunction, Error, Gradient};
use vectorlink_hnsw::layer::VectorComparator;
use vectorlink_hnsw::{comparator::CosineDistance1536, vectors::Vectors};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

pub fn sigmoid(z: &DVector<f32>) -> DVector<f32> {
    z.map(|x| 1.0 / (1.0 + (-x).exp()))
}

pub struct MatchClassifier {
    pub features: DMatrix<f32>,
    pub answers: DVector<f32>,
}

impl CostFunction for MatchClassifier {
    type Param = DVector<f32>;
    type Output = f32;

    fn cost(&self, w: &Self::Param) -> Result<Self::Output, Error> {
        let x = &self.features;
        let xw = x * w;
        let sigmoid_wx: DVector<f32> = sigmoid(&xw);
        let f = self
            .answers
            .zip_map(&sigmoid_wx, |y, swx_i| {
                y * swx_i.ln() + (1.0 - y) * (1.0 - swx_i).ln()
            })
            .sum();

        Ok(-f)
    }
}

impl Gradient for MatchClassifier {
    type Param = DVector<f32>;
    type Gradient = DVector<f32>;

    fn gradient(&self, w: &Self::Param) -> Result<Self::Gradient, Error> {
        let (n, _) = self.features.shape();
        let x = &self.features;
        let y = &self.answers;
        let xw = x * w;
        let dy = sigmoid(&xw) - y;
        let g: DVector<f32> = x.transpose() * dy / n as f32;
        Ok(g)
    }
}

pub fn predict(x: &DMatrix<f32>, coeff: &DVector<f32>) -> DVector<f32> {
    let y_hat = x * coeff;
    let sigmoid_y_hat = sigmoid(&y_hat);
    sigmoid_y_hat.map(|v| if v > 0.5 { 1.0 } else { 0.0 })
}

pub fn build_test_and_train<'a>(
    proportion_for_test: f32,
    comparison_fields: Vec<String>,
    all_answers: HashMap<String, Vec<String>>,
    source_compare_graph: &CompareGraph<'a>,
    target_compare_graph: &CompareGraph<'a>,
    candidates: Vec<(Vec<u32>, Vec<u32>)>,
) -> (
    Vec<String>,
    DMatrix<f32>,
    DVector<f32>,
    DMatrix<f32>,
    DVector<f32>,
) {
    // unweighted.. we want the raw X without Beta so we can estimate
    let weights: Vec<(String, f32)> = comparison_fields
        .iter()
        .map(|s| (s.to_string(), 1.0))
        .collect();

    let count = candidates
        .iter()
        .map(|(a, b)| (a.len() * b.len()) as f32)
        .sum::<f32>();

    let record_max = (count * proportion_for_test) as u32;
    eprintln!("record_max: {record_max}");
    let mut train_features = Vec::new();
    let mut train_answers = Vec::new();
    let mut test_features = Vec::new();
    let mut test_answers = Vec::new();
    for (sources, targets) in candidates.iter() {
        for source in sources {
            for target in targets.iter() {
                let training = *source < record_max || *target < record_max;
                let mut distances = compare_record_distances(
                    source_compare_graph,
                    target_compare_graph,
                    *source,
                    *target,
                    &weights,
                );
                // Extend with dummy for intercept...
                distances.push(1.0);
                if training {
                    train_features.push(distances);
                } else {
                    test_features.push(distances);
                }
                let source_id = source_compare_graph.graph.record_id_field_value(*source);
                let target_id = target_compare_graph.graph.record_id_field_value(*target);
                if let Some(targets) = all_answers.get(source_id) {
                    if targets.iter().any(|s| s == target_id) {
                        if training {
                            train_answers.push(1.0);
                        } else {
                            test_answers.push(1.0);
                        }
                    } else if training {
                        train_answers.push(0.0);
                    } else {
                        test_answers.push(0.0);
                    }
                } else if training {
                    train_answers.push(0.0)
                } else {
                    test_answers.push(0.0)
                }
            }
        }
    }
    let train_count = train_answers.len();
    let test_count = test_answers.len();
    let feature_len = comparison_fields.len() + 1; // includes intercept dummy
    let mut feature_names: Vec<String> = weights.iter().map(|(s, _)| s.to_string()).collect();
    feature_names.push("__INTERCEPT__".to_string());
    (
        feature_names,
        DMatrix::from_row_iterator(
            train_count,
            feature_len,
            train_features.iter().flat_map(|v| v.iter().copied()),
        ),
        DVector::from(train_answers),
        DMatrix::from_row_iterator(
            test_count,
            feature_len, // includes intercept dummy
            test_features.iter().flat_map(|v| v.iter().copied()),
        ),
        DVector::from(test_answers),
    )
}

pub fn compare_record_distances(
    source: &CompareGraph,
    target: &CompareGraph,
    source_record: u32,
    target_record: u32,
    weights: &Vec<(String, f32)>,
) -> Vec<f32> {
    let mut results: Vec<f32> = Vec::with_capacity(weights.len());
    for (field, _) in weights {
        if field == "__INTERCEPT__" {
            results.push(1.0);
            continue;
        }
        let source_value_id = source
            .graph
            .get(field)
            .expect("field missing on source graph")
            .record_id_to_value_id(source_record);
        let target_value_id = target
            .graph
            .get(field)
            .expect("field missing on source graph")
            .record_id_to_value_id(target_record);
        if let (Some(source_vector_id), Some(target_vector_id)) = (source_value_id, target_value_id)
        {
            let source_vec = &source.vecs[field][source_vector_id as usize];
            let target_vec = &target.vecs[field][target_vector_id as usize];
            let dummy = Vectors::empty(6144);
            let comparator = CosineDistance1536::new(&dummy);
            let distance = comparator.compare_vec_unstored(source_vec, target_vec);
            results.push(distance);
        } else {
            results.push(0.5); // NOTE: This may be too unprincipled
        }
    }
    if results.is_empty() {
        panic!("No overlap between records - incomparable");
    }
    results
}
