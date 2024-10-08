use std::{collections::HashMap, fs::File, path::Path};

use anyhow::Context;

use clap::Parser;
use csv::Writer;
use either::IntoEither;
use hnsw_redux::{
    comparator::CosineDistance1536,
    index::{Index, IndexConfiguration},
    layer::VectorComparator,
    params::SearchParams,
    vectors::{Vector, Vectors},
};

use crate::{
    graph::{CompareGraph, FullGraph},
    model::EmbedderMetadata,
};

#[derive(Parser)]
pub struct CompareCommand {
    /// The indexed graph in which to search for comparison
    #[arg(short, long)]
    target_graph_dir: String,

    /// The unindexed graph from which to search
    #[arg(short, long)]
    source_graph_dir: String,

    #[arg(short, long)]
    /// Field on which to perform the filter
    filter_field: String,

    #[arg(short, long)]
    /// Field on which to return matches
    id_field: String,

    #[arg(short, long)]
    /// The initial filter threshold to determine what to test
    initial_threshold: f32,

    #[arg(short, long)]
    /// The threshold for the weighted comparison
    match_threshold: f32,

    /// Path to output csv
    #[arg(short, long)]
    output_file: String,
}

fn compare_records(
    source: &CompareGraph,
    target: &CompareGraph,
    source_record: u32,
    target_record: u32,
) -> f32 {
    let mut results: Vec<f32> = Vec::new();
    for field in source.vecs.keys() {
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
        }
    }
    if results.is_empty() {
        panic!("No overlap between records");
    }
    results.iter().sum::<f32>() / results.len() as f32
}

impl CompareCommand {
    pub async fn execute(&self, _config: &EmbedderMetadata) -> Result<(), anyhow::Error> {
        let target_graph_dir_path = Path::new(&self.target_graph_dir);
        let source_graph_dir_path = Path::new(&self.source_graph_dir);
        let source_graph_path = source_graph_dir_path.join("csv.graph");
        let target_graph_path = target_graph_dir_path.join("csv.graph");
        let source_graph_file =
            File::open(source_graph_path).context("source file could not be loaded")?;
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

        let target_id_graph = target_graph
            .get(&self.id_field)
            .expect("No target field graph found");

        let source_id_graph = source_graph
            .get(&self.id_field)
            .expect("No target field graph found");

        let source_vecs: HashMap<String, Vectors> = source_graph
            .fields()
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    Vectors::load(source_graph_dir_path, name)
                        .unwrap_or_else(|_| panic!("Unable to load vector file for {name}")),
                )
            })
            .collect();
        let target_vecs: HashMap<String, Vectors> = target_graph
            .fields()
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    Vectors::load(target_graph_dir_path, name)
                        .unwrap_or_else(|_| panic!("Unable to load vector file for {name}")),
                )
            })
            .collect();
        let source_compare_graph = CompareGraph::new(&source_graph, source_vecs);
        let target_compare_graph = CompareGraph::new(&target_graph, target_vecs);

        let mut wtr = Writer::from_path(&self.output_file)?;
        wtr.write_record(["source_id", "target_id", "distance"])?;
        for (sources, targets) in candidates_for_compare {
            for source in sources {
                for target in targets.iter() {
                    let distance = compare_records(
                        &source_compare_graph,
                        &target_compare_graph,
                        source,
                        *target,
                    );
                    if distance < self.match_threshold {
                        let source_id = source_id_graph.record_id_to_value(source).unwrap_or("");
                        let target_id = target_id_graph.record_id_to_value(*target).unwrap_or("");
                        wtr.write_record([source_id, target_id, &format!("{}", distance)])?;
                    }
                }
            }
        }
        Ok(())
    }
}
