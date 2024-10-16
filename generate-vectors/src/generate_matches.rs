use std::{fs::File, path::Path};

use anyhow::Context;

use clap::Parser;
use hnsw_redux::{index::IndexConfiguration, vectors::Vectors};

use crate::{graph::FullGraph, model::EmbedderMetadata};

#[derive(Parser)]
pub struct GenerateMatchesCommand {
    /// The indexed graph in which to search for comparison
    #[arg(short, long)]
    target_graph_dir: String,

    /// The unindexed graph from which to search
    #[arg(short, long)]
    source_graph_dir: String,

    /// Original record file (Either CSV or JSON-Lines
    #[arg(short, long)]
    record_file: String,

    #[arg(short, long)]
    /// Field on which to perform the filter
    filter_field: String,

    #[arg(short, long)]
    /// Correct answers output to CSV organized as: source,target
    answers_file: String,
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

        todo!();
    }
}
