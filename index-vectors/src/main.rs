use std::io;

use clap::Parser;
use hnsw_redux::{
    index::{Hnsw1024, IndexConfiguration},
    params::BuildParams,
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

    let vectors = Vectors::load(&args.vector_directory, &args.name)?;

    let hnsw: IndexConfiguration =
        Hnsw1024::generate(args.name, vectors, &BuildParams::default()).into();

    hnsw.store_hnsw(&args.hnsw_root_directory)?;

    Ok(())
}
