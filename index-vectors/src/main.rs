use std::io;

use clap::Parser;
use hnsw_redux::{
    index::{Hnsw1024, Index, IndexConfiguration},
    params::BuildParams,
    vectors::Vectors,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Command {
    #[arg(long)]
    file: String,
}

fn main() -> io::Result<()> {
    let args = Command::parse();
    let vectors = Vectors::from_file(&args.file, 4096)?;

    let hnsw: IndexConfiguration = Hnsw1024::generate(vectors, &BuildParams::default()).into();

    Ok(())
}
