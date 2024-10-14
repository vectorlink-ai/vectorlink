use std::{collections::HashMap, fs::File, path::Path};

use argmin::{
    core::{observers::ObserverMode, Executor},
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
use nalgebra::DVector;
use smartcore::metrics::roc_auc_score;

use crate::{
    graph::{CompareGraph, FullGraph},
    model::EmbedderMetadata,
    train::{build_test_and_train, compare_record_distances, predict, MatchClassifier},
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

    #[arg(short, long, default_value_t = 0.33)]
    proportion_for_test: f32,
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
                &source_compare_graph,
                &target_compare_graph,
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
