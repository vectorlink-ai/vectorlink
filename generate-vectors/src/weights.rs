use std::{collections::HashMap, fs::File, path::Path};

use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor, Gradient},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use argmin_observer_slog::SlogLogger;

use anyhow::Context;

use clap::Parser;
use hnsw_redux::{
    index::{Index, IndexConfiguration},
    params::SearchParams,
    vectors::{Vector, Vectors},
};
use nalgebra::{DMatrix, DVector};
use smartcore::metrics::roc_auc_score;

use crate::{
    compare::compare_record_distances,
    graph::{CompareGraph, FullGraph},
    model::EmbedderMetadata,
};

#[derive(Parser)]
pub struct WeightsCommand {
    /// The indexed graph in which to search for comparison
    #[arg(short, long)]
    target_graph_dir: String,

    /// The unindexed graph from which to search
    #[arg(short, long)]
    source_graph_dir: String,

    #[arg(short, long)]
    /// Field on which to perform the filter
    filter_field: String,

    #[arg(short, long, num_args = 1..)]
    /// Field on which to perform the filter
    comparison_fields: Vec<String>,

    #[arg(short, long)]
    /// The initial filter threshold to determine what to test
    initial_threshold: f32,

    #[arg(short, long)]
    /// Correct answers in CSV organized as: source,target
    answers_file: String,

    #[arg(short, long, default_value_t = 1.0)]
    /// Weights the F-score towards precision by a factor of beta
    beta: f32,

    #[arg(short, long, default_value_t = 0.33)]
    proportion_for_test: f32,
}

pub fn sigmoid(z: &DVector<f32>) -> DVector<f32> {
    z.map(|x| 1.0 / (1.0 + (-x).exp()))
}

struct MatchClassifier {
    features: DMatrix<f32>,
    answers: DVector<f32>,
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

fn build_test_and_train<'a>(
    proportion_for_test: f32,
    comparison_fields: Vec<String>,
    all_answers: HashMap<String, Vec<String>>,
    source_compare_graph: CompareGraph<'a>,
    target_compare_graph: CompareGraph<'a>,
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

    let count = usize::min(
        source_compare_graph.graph.record_count(),
        target_compare_graph.graph.record_count(),
    );
    let record_max = (count as f32 * proportion_for_test) as u32;
    let mut train_features = Vec::new();
    let mut train_answers = Vec::new();
    let mut test_features = Vec::new();
    let mut test_answers = Vec::new();
    for (sources, targets) in candidates.iter() {
        for source in sources {
            for target in targets.iter() {
                let training = *source < record_max || *target < record_max;
                let mut distances = compare_record_distances(
                    &source_compare_graph,
                    &target_compare_graph,
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

impl WeightsCommand {
    pub async fn execute(&self, _config: &EmbedderMetadata) -> Result<(), anyhow::Error> {
        let target_graph_dir_path = Path::new(&self.target_graph_dir);
        let source_graph_dir_path = Path::new(&self.source_graph_dir);
        let source_graph_path = source_graph_dir_path.join("aggregated.graph");
        let target_graph_path = target_graph_dir_path.join("aggregated.graph");
        let source_graph_file =
            File::open(&source_graph_path).context("source file could not be loaded")?;
        eprintln!("source_graph_path: {:?}", &source_graph_path);
        let source_graph: FullGraph =
            serde_json::from_reader(source_graph_file).context("Unable to load source graph")?;
        let target_graph_file =
            File::open(target_graph_path).context("target file could not be loaded")?;
        let target_graph: FullGraph =
            serde_json::from_reader(target_graph_file).context("Unable to load target graph")?;

        let hnsw_root_directory =
            target_graph_dir_path.join(format!("{}.hnsw", &self.filter_field));
        let hnsw: IndexConfiguration = IndexConfiguration::load(
            &self.filter_field,
            &hnsw_root_directory,
            target_graph_dir_path,
        )?;

        let source_vectors = Vectors::load(source_graph_dir_path, &self.filter_field)
            .context("Unable to load vector file")?;

        let target_field_graph = target_graph
            .get(&self.filter_field)
            .expect("No target field graph found");

        let source_field_graph = source_graph
            .get(&self.filter_field)
            .expect("No target field graph found");

        // Source Value id is position, and results are target value ids.
        let results: Vec<Vec<u32>> = source_vectors
            .iter()
            .map(|query_vec| {
                hnsw.search(Vector::Slice(query_vec), &SearchParams::default())
                    .iter()
                    .filter_map(|(target_value_id, distance)| {
                        if distance < self.initial_threshold {
                            Some(target_value_id)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        let candidates_for_compare: Vec<(Vec<u32>, Vec<u32>)> = results
            .iter()
            .enumerate()
            .map(|(source_value_id, target_value_ids)| {
                let all_target_record_ids: Vec<u32> = target_value_ids
                    .iter()
                    .flat_map(|id| target_field_graph.value_id_to_record_ids(*id))
                    .copied()
                    .collect();
                let all_source_record_ids: Vec<u32> = source_field_graph
                    .value_id_to_record_ids(source_value_id as u32)
                    .to_vec();
                (all_source_record_ids, all_target_record_ids)
            })
            .collect();

        let source_vecs: HashMap<String, Vectors> = source_graph.load_vecs(source_graph_dir_path);
        let target_vecs: HashMap<String, Vectors> = target_graph.load_vecs(target_graph_dir_path);

        let answers_file = File::open(&self.answers_file).context("Unable to open answers file")?;
        let mut rdr = csv::Reader::from_reader(answers_file);
        let mut answers: HashMap<String, Vec<String>> = HashMap::new();
        for result in rdr.records() {
            let record = result.expect("Unable to parse csv field");
            if let Some(result) = answers.get_mut(&record[0]) {
                result.push(record[1].to_string());
            } else {
                answers.insert(record[0].to_string(), vec![record[1].to_string()]);
            }
        }

        let source_compare_graph = CompareGraph::new(&source_graph, source_vecs);
        let target_compare_graph = CompareGraph::new(&target_graph, target_vecs);
        let proportion_for_test = 0.33;
        let comparison_fields = self.comparison_fields.to_vec();
        let (feature_names, train_features, train_answers, test_features, test_answers) =
            build_test_and_train(
                proportion_for_test,
                comparison_fields,
                answers,
                source_compare_graph,
                target_compare_graph,
                candidates_for_compare,
            );

        // Define our cost function
        let cost = MatchClassifier {
            features: train_features,
            answers: train_answers,
        };
        let field_width = self.comparison_fields.len();
        // The final value is the intercept (and not a weight) which we also want to learn!
        let init_param: DVector<f32> = DVector::from(vec![1.0; field_width + 1]);

        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;

        // Set up solver
        let solver = LBFGS::new(linesearch, 7);

        /*
        Matrix<f32, Dyn, Const<1>, VecStorage<f32, Dyn, Const<1>>>:
           argmin_math::ArgminSub<Matrix<f32, Dyn, Const<1>, VecStorage<f32, Dyn, Const<1>>>, Matrix<f32, Dyn, Const<1>, VecStorage<f32, Dyn, Const<1>>>>
        */

        // Run solver
        let res = Executor::new(cost, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;
        let betas: DVector<f32> = res
            .state()
            .best_param
            .clone()
            .context("Could not estimate parameters")?;

        let y: Vec<f32> = test_answers.into_iter().copied().collect();
        let y_hat_as_nalgebra = predict(&test_features, &betas);
        let y_hat: Vec<f32> = y_hat_as_nalgebra.into_iter().copied().collect();
        let score = roc_auc_score(&y, &y_hat);

        let weights: HashMap<String, f32> = feature_names
            .into_iter()
            .zip(betas.data.as_vec().iter().copied())
            .collect();
        eprintln!("ROC AUC {}", score);
        let weights_str = serde_json::to_string(&weights).context("Could not serialize weights")?;
        eprintln!("{weights_str}");
        Ok(())
    }
}
