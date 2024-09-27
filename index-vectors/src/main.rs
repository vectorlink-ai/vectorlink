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

    hnsw.optimize(&sp, 0x533D);
    hnsw.store_hnsw(&args.hnsw_root_directory)?;

    let recall = hnsw.test_recall(&sp, 0x533D);
    eprintln!("recall: {recall}");

    Ok(())
}
