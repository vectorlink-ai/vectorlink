use std::io;

use clap::Parser;
use hnsw_redux::vectors::Vectors;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Command {
    #[arg(long)]
    vector_directory: String,
    #[arg(long)]
    source_name: String,
    #[arg(long)]
    destination_name: String,
}
fn main() -> io::Result<()> {
    let args = Command::parse();

    let mut vectors = Vectors::load(&args.vector_directory, &args.source_name)?;
    vectors.normalize();

    vectors.store(&args.vector_directory, &args.destination_name)?;

    Ok(())
}
