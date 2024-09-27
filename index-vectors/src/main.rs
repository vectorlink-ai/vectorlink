use std::io;

use clap::Parser;
use hnsw_redux::{
    index::{Hnsw1024, Index, IndexConfiguration},
    params::{BuildParams, SearchParams},
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
    let vectors = Vectors::load(&args.vector_directory, &args.name)?;

    let mut hnsw: IndexConfiguration =
        Hnsw1024::generate(args.name, vectors, &BuildParams::default()).into();

    let sp = SearchParams::default();
    let mut seed = 0x533D;
    let mut recall = hnsw.test_recall(&sp, seed);
    eprintln!("recall: {recall}");

    eprintln!("storing..");
    hnsw.store_hnsw(&args.hnsw_root_directory)?;

    eprintln!("improving neighbors..");
    let mut improvement = 1.0;
    while recall < 1.0 && improvement > 0.001 {
        seed += 1;
        hnsw.improve_neighbors_in_all_layers(&sp);
        let new_recall = hnsw.test_recall(&sp, seed);
        improvement = new_recall - recall;
        recall = new_recall;
        eprintln!("recall: {recall}, improvement: {improvement}");

        eprintln!("storing..");
        hnsw.store_hnsw(&args.hnsw_root_directory)?;
    }

    Ok(())
}
