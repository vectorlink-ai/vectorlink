use std::io;

use clap::Parser;
use hnsw_redux::{
    index::{Index, IndexConfiguration},
    params::SearchParams,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Command {
    #[arg(long)]
    vector_directory: String,
    #[arg(long)]
    hnsw_root_directory: String,
    #[arg(long)]
    name: String,
    #[arg(long)]
    cluster_file: String,
}

fn main() -> io::Result<()> {
    let args = Command::parse();
    eprintln!("About to index vectors: {}", &args.name);
    let hnsw = IndexConfiguration::load(
        &args.name,
        &args.hnsw_root_directory,
        &args.vector_directory,
    )?;
    let sp = SearchParams::default();
    hnsw.knn(20, &sp, &args.cluster_file);

    Ok(())
}
