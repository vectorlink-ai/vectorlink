use std::{collections::HashMap, fs::File, path::Path};

use anyhow::Context;

use clap::Parser;
use csv::Writer;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use vectorlink_hnsw::{
    comparator::CosineDistance1536, index::IndexConfiguration,
    layer::VectorComparator, vectors::Vectors,
};

use crate::{
    graph::{CompareGraph, FullGraph},
    model::EmbedderMetadata,
    train::sigmoid,
};

#[derive(Parser)]
pub struct SelfCompareCommand {
    /// The directory containing the indexed graph
    #[arg(short, long)]
    graph_dir: String,

    #[arg(short, long)]
    /// Field on which to perform the filter
    filter_field: String,

    #[arg(short = 't', long)]
    /// The initial filter threshold to determine what to test
    initial_threshold: f32,

    #[arg(short = 'm', long, default_value_t = 0.99)]
    /// The threshold probability
    match_threshold: f32,

    /// Path to output csv
    #[arg(short, long)]
    output_file: String,

    /// Path to output csv
    #[arg(short, long)]
    weights: String,

    #[arg(short = 's', long)]
    include_self: bool,
}

pub fn self_compare_record_distances(
    graph: &CompareGraph,
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
        let source_value_id = graph
            .graph
            .get(field)
            .expect("field missing on source graph")
            .record_id_to_value_id(source_record);
        let target_value_id = graph
            .graph
            .get(field)
            .expect("field missing on source graph")
            .record_id_to_value_id(target_record);
        if let (Some(source_vector_id), Some(target_vector_id)) =
            (source_value_id, target_value_id)
        {
            let source_vec = &graph.vecs[field][source_vector_id as usize];
            let target_vec = &graph.vecs[field][target_vector_id as usize];
            let dummy = Vectors::empty(6144);
            let comparator = CosineDistance1536::new(&dummy);
            let distance =
                comparator.compare_vec_unstored(source_vec, target_vec);
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

pub fn self_compare_records(
    graph: &CompareGraph,
    source_record: u32,
    target_record: u32,
    weights: &Vec<(String, f32)>,
) -> f32 {
    let results = self_compare_record_distances(
        graph,
        source_record,
        target_record,
        weights,
    );
    let betas: DVector<f32> =
        DVector::from(weights.iter().map(|(_, w)| *w).collect::<Vec<_>>());
    let x: DMatrix<f32> = DMatrix::from_row_iterator(1, results.len(), results);
    let y_hat = x * betas;
    let sigmoid_y_hat = sigmoid(&y_hat);
    sigmoid_y_hat[0]
}

impl SelfCompareCommand {
    pub async fn execute(
        &self,
        _config: &EmbedderMetadata,
    ) -> Result<(), anyhow::Error> {
        let weights_map: HashMap<String, f32> =
            serde_json::from_str(&self.weights)
                .context("Could not parse as weights")?;
        let weights: Vec<(String, f32)> = weights_map
            .iter()
            .map(|(s, f)| (s.to_string(), *f))
            .collect();

        let graph_dir_path = Path::new(&self.graph_dir);
        let graph_path = graph_dir_path.join("aggregated.graph");
        let graph_file =
            File::open(graph_path).context("graph file could not be opened")?;
        let graph: FullGraph = serde_json::from_reader(graph_file)
            .context("Unable to load graph")?;

        let hnsw_root_directory =
            graph_dir_path.join(format!("{}.hnsw", &self.filter_field));
        let hnsw: IndexConfiguration = IndexConfiguration::load(
            &self.filter_field,
            &hnsw_root_directory,
            graph_dir_path,
        )?;

        let field_graph =
            graph.get(&self.filter_field).expect("No field graph found");

        // Source Value id is position, and results are target value ids.
        const K_NEIGHBORS: usize = 20;
        let results: Vec<(u32, Vec<u32>)> = hnsw
            .knn(K_NEIGHBORS)
            .map(|(centre_id, distances)| {
                (
                    centre_id,
                    distances
                        .into_iter()
                        .map(|(match_id, _)| match_id)
                        .collect(),
                )
            })
            .collect();

        let candidates_for_compare: Vec<(Vec<u32>, Vec<u32>)> = results
            .iter()
            .map(|(source_value_id, target_value_ids)| {
                let all_target_record_ids: Vec<u32> = target_value_ids
                    .iter()
                    .flat_map(|id| field_graph.value_id_to_record_ids(*id))
                    .copied()
                    .collect();
                let all_source_record_ids: Vec<u32> = field_graph
                    .value_id_to_record_ids(*source_value_id as u32)
                    .to_vec();
                (all_source_record_ids, all_target_record_ids)
            })
            .collect();

        let field_vecs: HashMap<String, Vectors> =
            graph.load_vecs(graph_dir_path);
        let compare_graph = CompareGraph::new(&graph, field_vecs);

        let mut wtr = Writer::from_path(&self.output_file)?;
        wtr.write_record(["source_id", "target_id", "probability"])?;
        for (sources, targets) in candidates_for_compare {
            for source in sources {
                for target in targets.iter() {
                    if !self.include_self && source == *target {
                        continue;
                    }
                    let probability = self_compare_records(
                        &compare_graph,
                        source,
                        *target,
                        &weights,
                    );
                    if probability > self.match_threshold {
                        let source_id = graph.record_id_field_value(source);
                        let target_id = graph.record_id_field_value(*target);
                        wtr.write_record([
                            source_id,
                            target_id,
                            &format!("{}", probability),
                        ])?;
                    }
                }
            }
        }
        Ok(())
    }
}
