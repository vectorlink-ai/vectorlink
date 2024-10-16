use anyhow::Context;
use rayon::prelude::*;
use std::{fs::File, path::Path};

use clap::Parser;

use csv::Writer;
use hnsw_redux::{
    index::{Index, IndexConfiguration},
    params::FindPeaksParams,
    vectors::Vectors,
};

use crate::{graph::FullGraph, line_index::lookup_record, model::EmbedderMetadata};

#[derive(Parser)]
pub struct GenerateMatchesCommand {
    /// The indexed graph in which to search for comparison
    #[arg(short, long)]
    target_graph_dir: String,

    /// The unindexed graph from which to search
    #[arg(short, long)]
    source_graph_dir: String,

    /// Original record file (Either CSV or JSON-Lines)
    #[arg(short, long)]
    record_file: String,

    /// Line index for record file
    #[arg(short('i'), long)]
    record_file_index: String,

    /// Line index for record file
    #[arg(short, long, default_value_t = 300)]
    match_target_count: usize,

    #[arg(short, long)]
    /// Field on which to perform the filter
    filter_field: String,

    #[arg(short, long)]
    /// Correct answers output to CSV organized as: source,target
    answers_file: String,
}

fn read_y_n() -> Result<bool, anyhow::Error> {
    let mut s = String::new();
    std::io::stdin()
        .read_line(&mut s)
        .context("Could not read user input")?;
    if let Some('\n') = s.chars().next_back() {
        s.pop();
    }
    if let Some('\r') = s.chars().next_back() {
        s.pop();
    }
    let s = s.to_lowercase();
    if s == "y" || s == "yes" {
        Ok(true)
    } else {
        Ok(false)
    }
}

impl GenerateMatchesCommand {
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

        let record_file_path = Path::new(&self.record_file);
        let record_file =
            File::open(record_file_path).context("Could not open the original records file")?;
        let record_index_file_path = Path::new(&self.record_file_index);
        let record_index_file = File::open(record_index_file_path)
            .context("Could not open the index file for records")?;

        let find_peaks_params = FindPeaksParams::default();
        let peaks = hnsw.find_distance_transitions(find_peaks_params);
        let first_peak = if peaks.is_empty() {
            panic!("What do we do with no peak?");
        } else {
            peaks[0].0
        };

        let mut csv_wtr = Writer::from_path(&self.answers_file)?;
        csv_wtr.write_record(["source_id", "target_id"])?;

        let mut total_records = 0;
        let k = 20;
        let mut non_match_count = 0;
        let mut match_count = 0;
        const TOTAL_SEARCH_SIZE: usize = 10_000;

        let knn_results: Vec<_> = hnsw.knn(k).take_any(TOTAL_SEARCH_SIZE).collect();
        for (left_vector_id, neighbors) in hnsw.knn(k) {
            let i = neighbors.partition_point(|(_, d)| d < first_peak);
            // Let's take only candidates in which there is a transition
            if i != 0 && i != neighbors.len() {
                let (right_vector_id, distance) = if match_count > non_match_count {
                    // should be bigger than i, how do we calculate...
                    neighbors[i]
                } else {
                    neighbors[0]
                };
                let left = lookup_record(left_vector_id, &record_file, &record_index_file)?;
                let right = lookup_record(right_vector_id, &record_file, &record_index_file)?;
                println!("Are the following records referring to the same entity?:");
                println!("1. {left}");
                println!("2. {right}");
                let matches = read_y_n().context("Could not read input from user!")?;
                if matches {
                    let source_id = source_graph.record_id_field_value(left_vector_id);
                    let target_id = target_graph.record_id_field_value(right_vector_id);
                    csv_wtr.write_record(&[source_id, target_id]);
                    match_count += 1;
                } else {
                    non_match_count += 1;
                }
            }

            if match_count + non_match_count > &self.match_target_count {
                break;
            }
        }
        Ok(())
    }
}
