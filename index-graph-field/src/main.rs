use std::{io, path::Path, time::SystemTime};

use clap::Parser;
use hnsw_redux::{
    index::{Hnsw1024, Hnsw1536, Index, IndexConfiguration},
    params::{BuildParams, OptimizationParams},
    vectors::Vectors,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Command {
    #[arg(long)]
    graph_directory: String,
    #[arg(long)]
    field: String,
}

fn main() -> io::Result<()> {
    let start = SystemTime::now();
    let args = Command::parse();
    let graph_path = Path::new(&args.graph_directory);
    eprintln!(
        "About to index field {} in graph directory: {:?}",
        &args.field, &graph_path
    );

    let vectors = Vectors::load(&args.graph_directory, &args.field)?;
    // We need some logic switching on which vector dimension we are
    let mut hnsw: IndexConfiguration =
        Hnsw1536::generate(args.field.to_string(), vectors, &BuildParams::default()).into();
    eprintln!("{}: done generating", start.elapsed().unwrap().as_secs());

    eprintln!("storing..");
    let hnsw_path = graph_path.join(format!("{}.hnsw", &args.field));
    hnsw.store_hnsw(&hnsw_path)?;

    eprintln!("improving neighbors..");
    let op = OptimizationParams::default();

    hnsw.optimize(&op);

    eprintln!("storing after optimize..");
    hnsw.store_hnsw(&hnsw_path)?;

    Ok(())
}
