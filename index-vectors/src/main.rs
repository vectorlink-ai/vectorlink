use std::{io, time::SystemTime};

use clap::Parser;
use hnsw_redux::{
    index::{Hnsw1024, Index, IndexConfiguration},
    params::{BuildParams, OptimizationParams},
    vectors::Vectors,
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
}

fn main() -> io::Result<()> {
    let args = Command::parse();
    eprintln!("About to index vectors: {}", &args.name);
    let start = SystemTime::now();
    let vectors = Vectors::load(&args.vector_directory, &args.name)?;

    let mut hnsw: IndexConfiguration =
        Hnsw1024::generate(args.name, vectors, &BuildParams::default()).into();
    eprintln!("{}: done generating", start.elapsed().unwrap().as_secs());

    eprintln!("storing..");
    hnsw.store_hnsw(&args.hnsw_root_directory)?;

    eprintln!("improving neighbors..");
    let op = OptimizationParams::default();

    hnsw.optimize(&op);

    Ok(())
}
