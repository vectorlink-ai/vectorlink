use std::{io, time::SystemTime};

use clap::Parser;
use vectorlink_hnsw::{
    index::{Index, IndexConfiguration},
    params::OptimizationParams,
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
    params: Option<String>,
}

fn main() -> io::Result<()> {
    let args = Command::parse();
    let start = SystemTime::now();
    let mut hnsw: IndexConfiguration = IndexConfiguration::load(
        &args.name,
        &args.hnsw_root_directory,
        &args.vector_directory,
    )?;
    let op = args.params.map_or_else(
        OptimizationParams::default,
        |sp_string| -> OptimizationParams {
            serde_json::from_str(&sp_string).expect("Unable to parse search params")
        },
    );

    hnsw.optimize_and_save(&args.hnsw_root_directory, &op)?;

    eprintln!(
        "{}: done improving neighbors",
        start.elapsed().unwrap().as_secs()
    );

    Ok(())
}
